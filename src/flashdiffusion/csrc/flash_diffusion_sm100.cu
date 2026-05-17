/*
 * flash_diffusion_sm100.cu  — SM100 UMMA v5
 * ==========================================
 * Native Blackwell tcgen05.mma kernel for FlashDiffusion.
 * B200 / B300 (SM100 / SM103).
 *
 * Key fixes from v4:
 *   - BN=256 (not 128) — from mma_shape_B: ((_256,_16),_1,_4)
 *   - SMEM layouts via UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<T>{}, mma_shape)
 *   - mma_shape_A/B computed via partition_shape_A/B(tiled_mma, make_shape(M,K))
 *   - Kernel templated on SMEM layouts (matches example pattern)
 *   - bK = tile_size<2>(tiled_mma) * Int<4>{} = 64
 *
 * Build:
 *   -arch=sm_100a -std=c++17 -O3 --use_fast_math
 *   -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__
 *   --expt-relaxed-constexpr --expt-extended-lambda
 *   -DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1
 *   -DCUTE_ARCH_MMA_SM100A_ENABLED=1
 *   -I /path/to/cutlass_main/include
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cute/tensor.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/algorithm/cooperative_copy.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/half.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

using namespace cute;

// ---------------------------------------------------------------------------
// Sq-norm kernel (always compiled)
// ---------------------------------------------------------------------------
__global__ void sq_norm_kernel_sm100(
    const __half* __restrict__ X,
    float*        __restrict__ sq,
    int N, int d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s = 0.f;
    for (int k = 0; k < d; ++k) { float v = __half2float(X[i*d+k]); s += v*v; }
    sq[i] = s;
}

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ float fast_exp(float x) { return __expf(x); }

// ---------------------------------------------------------------------------
// UMMA configuration — derived from tiled_mma at host side
// bM=128, bN=256, bK=64 for SM100_MMA_F16BF16_SS<half,half,float,128,256,K,K>
// ---------------------------------------------------------------------------
static constexpr int BM       = 128;
static constexpr int BN       = 256;   // ← must match mma_shape_B first dim
static constexpr int BK_MMA   = 64;    // bK = tile_size<2>(tiled_mma) * 4
static constexpr int NTHREADS = 128;

using TypeA = cutlass::half_t;
using TypeB = cutlass::half_t;
using TypeC = float;

using MMA_Atom_Op = SM100_MMA_F16BF16_SS<
    TypeA, TypeB, TypeC,
    128, 256,            // M=128, N=256
    UMMA::Major::K, UMMA::Major::K
>;
using TiledMMA_t = decltype(make_tiled_mma(MMA_Atom_Op{}));

// ---------------------------------------------------------------------------
// SharedStorage — templated on SMEM layouts (computed by host)
// ---------------------------------------------------------------------------
template <class ASmemLayout, class BSmemLayout>
struct SharedStorage {
    alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> sA;
    alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout>> sB;
    alignas(16)  cute::uint64_t mma_barrier;
    alignas(16)  cute::uint32_t tmem_base_ptr;

    CUTE_DEVICE auto tensor_sA() {
        return make_tensor(make_smem_ptr(sA.begin()), ASmemLayout{});
    }
    CUTE_DEVICE auto tensor_sB() {
        return make_tensor(make_smem_ptr(sB.begin()), BSmemLayout{});
    }
};

// ---------------------------------------------------------------------------
// Prepass kernel
// ---------------------------------------------------------------------------
template <class SharedStorage_t>
__global__ void __launch_bounds__(NTHREADS)
prepass_kernel_sm100(
    const TypeA*  __restrict__ X,
    const float*  __restrict__ w,
    float*        __restrict__ D,
    const float*  __restrict__ sq_X,
    int N, int d, float beta)
{
    extern __shared__ char smem_raw[];
    SharedStorage_t& smem = *reinterpret_cast<SharedStorage_t*>(smem_raw);

    int tid   = threadIdx.x;
    int i0    = blockIdx.x * BM;
    int i1    = min(i0 + BM, N);
    int bm    = i1 - i0;

    uint32_t elect_one_thr  = cute::elect_one_sync();
    uint32_t elect_one_warp = (tid / 32 == 0);

    using TmemAllocator = cute::TMEM::Allocator1Sm;
    TmemAllocator tmem_alloc{};
    if (elect_one_warp) {
        tmem_alloc.allocate(TmemAllocator::Sm100TmemCapacityColumns,
                            &smem.tmem_base_ptr);
    }
    if (elect_one_warp && elect_one_thr) {
        cute::initialize_barrier(smem.mma_barrier, 1);
    }
    int mma_phase = 0;
    __syncthreads();

    TiledMMA_t tiled_mma;
    ThrMMA cta_mma = tiled_mma.get_slice(0);

    Tensor tCsA = smem.tensor_sA();
    Tensor tCsB = smem.tensor_sB();
    Tensor tCrA = cta_mma.make_fragment_A(tCsA);
    Tensor tCrB = cta_mma.make_fragment_B(tCsB);

    Tensor tCtAcc = cta_mma.make_fragment_C(
        make_tensor(make_gmem_ptr((TypeC*)nullptr),
                    make_layout(make_shape(Int<BM>{}, Int<BN>{}))));
    tCtAcc.data() = smem.tmem_base_ptr;

    float D_acc = 0.f;

    for (int j0 = 0; j0 < N; j0 += BN) {
        int j1 = min(j0 + BN, N);
        int bn = j1 - j0;

        tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

        // k-tile loop over coordinate dimension
        for (int k0 = 0; k0 < d; k0 += BK_MMA) {
            int klen = min(k0 + BK_MMA, d) - k0;

            // load Xi k-tile into sA (BM × BK_MMA), zero-pad
            for (int idx = tid; idx < BM * BK_MMA; idx += NTHREADS) {
                int r = idx / BK_MMA, c = idx % BK_MMA;
                smem.sA.begin()[idx] = (r < bm && c < klen)
                    ? X[(long long)(i0+r)*d + k0+c]
                    : TypeA(0.f);
            }
            // load Xj k-tile into sB (BN × BK_MMA), zero-pad
            for (int idx = tid; idx < BN * BK_MMA; idx += NTHREADS) {
                int r = idx / BK_MMA, c = idx % BK_MMA;
                smem.sB.begin()[idx] = (r < bn && c < klen)
                    ? X[(long long)(j0+r)*d + k0+c]
                    : TypeB(0.f);
            }
            __syncthreads();

            if (elect_one_warp) {
                gemm(tiled_mma, tCrA, tCrB, tCtAcc);
                tiled_mma.accumulate_ = UMMA::ScaleOut::One;
                cutlass::arch::umma_arrive(&smem.mma_barrier);
            }
            cute::wait_barrier(smem.mma_barrier, mma_phase);
            mma_phase ^= 1;
            __syncthreads();
        }

        // epilogue — scalar dist2 computation (correct v1)
        if (tid < bm) {
            float sqi = sq_X[i0 + tid];
            for (int j = 0; j < bn; ++j) {
                float dot = 0.f;
                for (int k = 0; k < d; ++k) {
                    int ki = k % BK_MMA;
                    float ai = __half2float(*reinterpret_cast<const __half*>(
                                   &smem.sA.begin()[tid * BK_MMA + ki]));
                    float bj = __half2float(*reinterpret_cast<const __half*>(
                                   &smem.sB.begin()[j   * BK_MMA + ki]));
                    dot += ai * bj;
                }
                float dist2 = fmaxf(sqi + sq_X[j0+j] - 2.f*dot, 0.f);
                D_acc += fast_exp(-beta * dist2) * w[j0+j];
            }
        }
        __syncthreads();
    }

    __syncthreads();
    if (elect_one_warp) {
        tmem_alloc.release_allocation_lock();
        tmem_alloc.free(smem.tmem_base_ptr,
                        TmemAllocator::Sm100TmemCapacityColumns);
    }

    if (tid < bm) D[i0 + tid] = D_acc;
}

// ---------------------------------------------------------------------------
// Matvec kernel
// ---------------------------------------------------------------------------
template <class SharedStorage_t>
__global__ void __launch_bounds__(NTHREADS)
matvec_kernel_sm100(
    const TypeA*  __restrict__ X,
    const float*  __restrict__ u,
    float*        __restrict__ out,
    const float*  __restrict__ sq_X,
    const float*  __restrict__ rscale,
    int N, int d, float beta)
{
    extern __shared__ char smem_raw[];
    SharedStorage_t& smem = *reinterpret_cast<SharedStorage_t*>(smem_raw);

    int tid   = threadIdx.x;
    int i0    = blockIdx.x * BM;
    int i1    = min(i0 + BM, N);
    int bm    = i1 - i0;

    uint32_t elect_one_thr  = cute::elect_one_sync();
    uint32_t elect_one_warp = (tid / 32 == 0);

    using TmemAllocator = cute::TMEM::Allocator1Sm;
    TmemAllocator tmem_alloc{};
    if (elect_one_warp) {
        tmem_alloc.allocate(TmemAllocator::Sm100TmemCapacityColumns,
                            &smem.tmem_base_ptr);
    }
    if (elect_one_warp && elect_one_thr) {
        cute::initialize_barrier(smem.mma_barrier, 1);
    }
    int mma_phase = 0;
    __syncthreads();

    TiledMMA_t tiled_mma;
    ThrMMA cta_mma = tiled_mma.get_slice(0);

    Tensor tCsA = smem.tensor_sA();
    Tensor tCsB = smem.tensor_sB();
    Tensor tCrA = cta_mma.make_fragment_A(tCsA);
    Tensor tCrB = cta_mma.make_fragment_B(tCsB);

    Tensor tCtAcc = cta_mma.make_fragment_C(
        make_tensor(make_gmem_ptr((TypeC*)nullptr),
                    make_layout(make_shape(Int<BM>{}, Int<BN>{}))));
    tCtAcc.data() = smem.tmem_base_ptr;

    float acc = 0.f;

    for (int j0 = 0; j0 < N; j0 += BN) {
        int j1 = min(j0 + BN, N);
        int bn = j1 - j0;

        tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

        for (int k0 = 0; k0 < d; k0 += BK_MMA) {
            int klen = min(k0 + BK_MMA, d) - k0;

            for (int idx = tid; idx < BM * BK_MMA; idx += NTHREADS) {
                int r = idx / BK_MMA, c = idx % BK_MMA;
                smem.sA.begin()[idx] = (r < bm && c < klen)
                    ? X[(long long)(i0+r)*d + k0+c] : TypeA(0.f);
            }
            for (int idx = tid; idx < BN * BK_MMA; idx += NTHREADS) {
                int r = idx / BK_MMA, c = idx % BK_MMA;
                smem.sB.begin()[idx] = (r < bn && c < klen)
                    ? X[(long long)(j0+r)*d + k0+c] : TypeB(0.f);
            }
            __syncthreads();

            if (elect_one_warp) {
                gemm(tiled_mma, tCrA, tCrB, tCtAcc);
                tiled_mma.accumulate_ = UMMA::ScaleOut::One;
                cutlass::arch::umma_arrive(&smem.mma_barrier);
            }
            cute::wait_barrier(smem.mma_barrier, mma_phase);
            mma_phase ^= 1;
            __syncthreads();
        }

        if (tid < bm) {
            float sqi = sq_X[i0 + tid];
            for (int j = 0; j < bn; ++j) {
                float dot = 0.f;
                for (int k = 0; k < d; ++k) {
                    int ki = k % BK_MMA;
                    float ai = __half2float(*reinterpret_cast<const __half*>(
                                   &smem.sA.begin()[tid * BK_MMA + ki]));
                    float bj = __half2float(*reinterpret_cast<const __half*>(
                                   &smem.sB.begin()[j   * BK_MMA + ki]));
                    dot += ai * bj;
                }
                float dist2 = fmaxf(sqi + sq_X[j0+j] - 2.f*dot, 0.f);
                acc += fast_exp(-beta * dist2) * u[j0+j];
            }
        }
        __syncthreads();
    }

    __syncthreads();
    if (elect_one_warp) {
        tmem_alloc.release_allocation_lock();
        tmem_alloc.free(smem.tmem_base_ptr,
                        TmemAllocator::Sm100TmemCapacityColumns);
    }

    if (tid < bm) out[i0 + tid] = rscale[i0 + tid] * acc;
}

// ---------------------------------------------------------------------------
// Host-side launcher — compute SMEM layouts using CUTLASS API
// ---------------------------------------------------------------------------
void launch_prepass(
    const TypeA* X, const float* w, float* D, const float* sq_X,
    int N, int d, float beta, cudaStream_t stream)
{
    TiledMMA_t tiled_mma;

    auto bM = tile_size<0>(tiled_mma);
    auto bN = tile_size<1>(tiled_mma);
    auto bK = tile_size<2>(tiled_mma) * Int<4>{};

    auto mma_shape_A = partition_shape_A(tiled_mma, make_shape(bM, bK));
    auto mma_shape_B = partition_shape_B(tiled_mma, make_shape(bN, bK));

    auto sA_layout = UMMA::tile_to_mma_shape(
        UMMA::Layout_K_SW128_Atom<TypeA>{}, mma_shape_A);
    auto sB_layout = UMMA::tile_to_mma_shape(
        UMMA::Layout_K_SW128_Atom<TypeB>{}, mma_shape_B);

    using SmemA_t  = decltype(sA_layout);
    using SmemB_t  = decltype(sB_layout);
    using Storage_t = SharedStorage<SmemA_t, SmemB_t>;

    int smem_bytes = sizeof(Storage_t);
    int grid       = (N + BM - 1) / BM;

    cudaFuncSetAttribute(
        prepass_kernel_sm100<Storage_t>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    prepass_kernel_sm100<Storage_t><<<grid, NTHREADS, smem_bytes, stream>>>(
        X, w, D, sq_X, N, d, beta);
}

void launch_matvec(
    const TypeA* X, const float* u, float* out,
    const float* sq_X, const float* rscale,
    int N, int d, float beta, cudaStream_t stream)
{
    TiledMMA_t tiled_mma;

    auto bM = tile_size<0>(tiled_mma);
    auto bN = tile_size<1>(tiled_mma);
    auto bK = tile_size<2>(tiled_mma) * Int<4>{};

    auto mma_shape_A = partition_shape_A(tiled_mma, make_shape(bM, bK));
    auto mma_shape_B = partition_shape_B(tiled_mma, make_shape(bN, bK));

    auto sA_layout = UMMA::tile_to_mma_shape(
        UMMA::Layout_K_SW128_Atom<TypeA>{}, mma_shape_A);
    auto sB_layout = UMMA::tile_to_mma_shape(
        UMMA::Layout_K_SW128_Atom<TypeB>{}, mma_shape_B);

    using SmemA_t  = decltype(sA_layout);
    using SmemB_t  = decltype(sB_layout);
    using Storage_t = SharedStorage<SmemA_t, SmemB_t>;

    int smem_bytes = sizeof(Storage_t);
    int grid       = (N + BM - 1) / BM;

    cudaFuncSetAttribute(
        matvec_kernel_sm100<Storage_t>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    matvec_kernel_sm100<Storage_t><<<grid, NTHREADS, smem_bytes, stream>>>(
        X, u, out, sq_X, rscale, N, d, beta);
}

#endif  // CUTLASS_ARCH_MMA_SM100_SUPPORTED

// ---------------------------------------------------------------------------
// Python entry points
// ---------------------------------------------------------------------------

torch::Tensor compute_sq_norms(torch::Tensor X_fp16) {
    int N = X_fp16.size(0), d = X_fp16.size(1);
    auto sq = torch::empty({N}, X_fp16.options().dtype(torch::kFloat32));
    sq_norm_kernel_sm100<<<(N+255)/256, 256, 0,
                           at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(X_fp16.data_ptr<at::Half>()),
        sq.data_ptr<float>(), N, d);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return sq;
}

torch::Tensor prepass(
    torch::Tensor X_fp16, torch::Tensor w,
    torch::Tensor sq_X,   float beta)
{
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    int N = X_fp16.size(0), d = X_fp16.size(1);
    auto D = torch::zeros({N}, w.options());
    launch_prepass(
        reinterpret_cast<const TypeA*>(X_fp16.data_ptr<at::Half>()),
        w.data_ptr<float>(), D.data_ptr<float>(),
        sq_X.data_ptr<float>(), N, d, beta,
        at::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return D;
#else
    TORCH_CHECK(false, "Rebuild with -DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1");
#endif
}

torch::Tensor matvec(
    torch::Tensor X_fp16, torch::Tensor v,
    torch::Tensor rscale, torch::Tensor sq_X, float beta)
{
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    int N = X_fp16.size(0), d = X_fp16.size(1);
    auto u   = rscale * v;
    auto out = torch::empty({N}, v.options());
    launch_matvec(
        reinterpret_cast<const TypeA*>(X_fp16.data_ptr<at::Half>()),
        u.data_ptr<float>(), out.data_ptr<float>(),
        sq_X.data_ptr<float>(), rscale.data_ptr<float>(),
        N, d, beta, at::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
#else
    TORCH_CHECK(false, "Rebuild with -DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1");
#endif
}

PYBIND11_MODULE(flash_diffusion_sm100, m) {
    m.doc() = "FlashDiffusion SM100 native UMMA tcgen05.mma v5 (B200/B300)";
    m.def("compute_sq_norms", &compute_sq_norms, py::arg("X_fp16"));
    m.def("prepass",  &prepass,
          py::arg("X_fp16"), py::arg("w"), py::arg("sq_X"), py::arg("beta"));
    m.def("matvec",   &matvec,
          py::arg("X_fp16"), py::arg("v"), py::arg("rscale"),
          py::arg("sq_X"), py::arg("beta"));
}
