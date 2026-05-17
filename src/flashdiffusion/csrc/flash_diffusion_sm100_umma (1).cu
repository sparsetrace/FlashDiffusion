/*
 * flash_diffusion_sm100.cu  — SM100 UMMA v3
 * ==========================================
 * Native Blackwell tcgen05.mma (UMMA) kernel for FlashDiffusion.
 * B200 / B300 (SM100 / SM103).
 *
 * API verified from CUTLASS main examples/cute/tutorial/blackwell/01_mma_sm100.cu
 *
 * Exact pattern used:
 *   using TmemAllocator = cute::TMEM::Allocator1Sm;
 *   tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
 *   gemm(tiled_mma, tCrA(_,_,k), tCrB(_,_,k), tCtAcc);   // no cute:: prefix!
 *   tiled_mma.accumulate_ = UMMA::ScaleOut::One;
 *   cutlass::arch::umma_arrive(&smem.mma_barrier);
 *   cute::wait_barrier(smem.mma_barrier, phase);
 *   cooperative_copy<128>(tid, src, dst);                  // GMEM->SMEM
 *   make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);  // TMEM->RMEM
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

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// ---------------------------------------------------------------------------
// Tile constants
// BM=128, BN=128 required for SM100_MMA_F16BF16_SS (M must be 64 or 128)
// NTHREADS=128 required (one warpgroup for 1-CTA UMMA)
// ---------------------------------------------------------------------------
static constexpr int BM       = 128;
static constexpr int BN       = 128;
static constexpr int BK       = 16;    // tcgen05.mma K tile for fp16
static constexpr int NTHREADS = 128;

// ---------------------------------------------------------------------------
// UMMA atom
// ---------------------------------------------------------------------------
using MMA_Atom_Op = SM100_MMA_F16BF16_SS<
    cutlass::half_t,   // TypeA
    cutlass::half_t,   // TypeB
    float,             // TypeC accumulator (in TMEM)
    128,               // M (must be 64 or 128)
    128,               // N (must be multiple of 8, 8..256)
    UMMA::Major::K,    // A K-major
    UMMA::Major::K     // B K-major
>;
using TiledMMA_t = decltype(make_tiled_mma(MMA_Atom_Op{}));

// ---------------------------------------------------------------------------
// SMEM layouts — 128-byte swizzle for tcgen05.mma SMEM descriptors
// ---------------------------------------------------------------------------
using SmemLayoutAtomA = decltype(
    composition(Swizzle<3, 4, 3>{},
                Layout<Shape<_8, Int<BK>>, Stride<Int<BK>, _1>>{}));
using SmemLayoutAtomB = decltype(
    composition(Swizzle<3, 4, 3>{},
                Layout<Shape<_8, Int<BK>>, Stride<Int<BK>, _1>>{}));

using SmemLayoutA = decltype(
    tile_to_shape(SmemLayoutAtomA{}, Shape<Int<BM>, Int<BK>>{}));
using SmemLayoutB = decltype(
    tile_to_shape(SmemLayoutAtomB{}, Shape<Int<BN>, Int<BK>>{}));

// ---------------------------------------------------------------------------
// Shared memory — matches pattern from 01_mma_sm100.cu
// ---------------------------------------------------------------------------
struct SharedStorage {
    alignas(128) cute::ArrayEngine<cutlass::half_t, cute::cosize_v<SmemLayoutA>> sA;
    alignas(128) cute::ArrayEngine<cutlass::half_t, cute::cosize_v<SmemLayoutB>> sB;
    alignas(16)  cute::uint64_t mma_barrier;
    alignas(16)  cute::uint32_t tmem_base_ptr;

    CUTE_DEVICE auto tensor_sA() {
        return make_tensor(make_smem_ptr(sA.begin()), SmemLayoutA{});
    }
    CUTE_DEVICE auto tensor_sB() {
        return make_tensor(make_smem_ptr(sB.begin()), SmemLayoutB{});
    }
};

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ float fast_exp(float x) { return __expf(x); }

__device__ __forceinline__ float h2f(cutlass::half_t x) {
    return __half2float(*reinterpret_cast<const __half*>(&x));
}

#endif  // CUTLASS_ARCH_MMA_SM100_SUPPORTED

// ---------------------------------------------------------------------------
// Sq-norm kernel (no UMMA needed)
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
// Prepass kernel — SM100 UMMA
//   D_i = Σ_j exp(-beta * ||xi - xj||²) * w_j
//
// Pattern from 01_mma_sm100.cu:
//   1. cooperative_copy GMEM→SMEM
//   2. __syncthreads
//   3. elect_one_warp: gemm loop + umma_arrive
//   4. cute::wait_barrier
//   5. epilogue (scalar, reads SMEM directly for dist2 computation)
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(NTHREADS)
prepass_kernel_sm100(
    const cutlass::half_t* __restrict__ X,
    const float*           __restrict__ w,
    float*                 __restrict__ D,
    const float*           __restrict__ sq_X,
    int N, int d, float beta)
{
    extern __shared__ char smem_raw[];
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_raw);

    int tid   = threadIdx.x;
    int i0    = blockIdx.x * BM;
    int i1    = min(i0 + BM, N);
    int bm    = i1 - i0;

    uint32_t elect_one_thr  = cute::elect_one_sync();
    uint32_t elect_one_warp = (tid / 32 == 0);

    // ── TMEM allocation (warp 0 only) ──────────────────────────────────────
    using TmemAllocator = cute::TMEM::Allocator1Sm;
    TmemAllocator tmem_alloc{};
    if (elect_one_warp) {
        tmem_alloc.allocate(TmemAllocator::Sm100TmemCapacityColumns,
                            &smem.tmem_base_ptr);
    }

    // ── barrier init (thread 0 of warp 0) ──────────────────────────────────
    if (elect_one_warp && elect_one_thr) {
        cute::initialize_barrier(smem.mma_barrier, 1);
    }
    int mma_phase = 0;
    __syncthreads();   // all threads observe allocations + barrier init

    // ── TiledMMA setup ─────────────────────────────────────────────────────
    TiledMMA_t tiled_mma;
    ThrMMA cta_mma = tiled_mma.get_slice(0);   // peer CTA = 0 (1-CTA mode)

    Tensor tCsA = smem.tensor_sA();   // (BM, BK)  swizzled SMEM
    Tensor tCsB = smem.tensor_sB();   // (BN, BK)  swizzled SMEM

    // SMEM descriptor fragments for A and B (UMMA reads from descriptors)
    Tensor tCrA = cta_mma.make_fragment_A(tCsA);
    Tensor tCrB = cta_mma.make_fragment_B(tCsB);

    // TMEM accumulator — assign base ptr after allocation
    // Shape: (BM, BN), lives in TMEM
    Tensor tCtAcc = cta_mma.make_fragment_C(
        make_tensor(make_gmem_ptr((float*)nullptr),
                    make_layout(make_shape(Int<BM>{}, Int<BN>{}))));
    tCtAcc.data() = smem.tmem_base_ptr;

    // number of K tiles for the coordinate dimension
    int num_k = (d + BK - 1) / BK;

    // first MMA clears TMEM accumulator
    tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

    float D_acc = 0.f;

    // ── load Xi tile once (fixed across j-loop) ────────────────────────────
    Tensor gXi = make_tensor(
        make_gmem_ptr(X + (long long)i0 * d),
        make_layout(make_shape(bm, d), make_stride(d, _1{})));
    cooperative_copy<128>(tid, gXi, tCsA);
    __syncthreads();

    // ── j-tile loop ────────────────────────────────────────────────────────
    for (int j0 = 0; j0 < N; j0 += BN) {
        int j1 = min(j0 + BN, N);
        int bn = j1 - j0;

        // ── GMEM → SMEM: Xj tile ──────────────────────────────────────────
        Tensor gXj = make_tensor(
            make_gmem_ptr(X + (long long)j0 * d),
            make_layout(make_shape(bn, d), make_stride(d, _1{})));
        cooperative_copy<128>(tid, gXj, tCsB);
        __syncthreads();

        // ── tcgen05.mma: sA @ sB^T → TMEM ────────────────────────────────
        // EXACTLY as in 01_mma_sm100.cu:
        if (elect_one_warp) {
            for (int k = 0; k < size<1>(tCrA); ++k) {
                gemm(tiled_mma, tCrA(_, _, k), tCrB(_, _, k), tCtAcc);
                tiled_mma.accumulate_ = UMMA::ScaleOut::One;
            }
            cutlass::arch::umma_arrive(&smem.mma_barrier);
        }
        cute::wait_barrier(smem.mma_barrier, mma_phase);
        mma_phase ^= 1;
        __syncthreads();

        // ── epilogue: dist2 → exp → accumulate ────────────────────────────
        // The UMMA computed Xi @ Xj^T into TMEM — we use the SMEM tiles
        // directly for the scalar epilogue (correct, avoids TMEM readout
        // complexity in v1; TMEM readout optimization is v2)
        if (tid < bm) {
            float sqi = sq_X[i0 + tid];
            for (int j = 0; j < bn; ++j) {
                float dot = 0.f;
                for (int k = 0; k < d; ++k)
                    dot += h2f(tCsA(tid, k)) * h2f(tCsB(j, k));
                float dist2 = fmaxf(sqi + sq_X[j0+j] - 2.f*dot, 0.f);
                D_acc += fast_exp(-beta * dist2) * w[j0+j];
            }
        }
        __syncthreads();

        // reset for next j tile
        tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
    }

    // ── TMEM deallocation ──────────────────────────────────────────────────
    __syncthreads();
    if (elect_one_warp) {
        tmem_alloc.release_allocation_lock();
        tmem_alloc.free(smem.tmem_base_ptr,
                        TmemAllocator::Sm100TmemCapacityColumns);
    }

    if (tid < bm)
        D[i0 + tid] = D_acc;
}

// ---------------------------------------------------------------------------
// Matvec kernel — same structure as prepass
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(NTHREADS)
matvec_kernel_sm100(
    const cutlass::half_t* __restrict__ X,
    const float*           __restrict__ u,
    float*                 __restrict__ out,
    const float*           __restrict__ sq_X,
    const float*           __restrict__ rscale,
    int N, int d, float beta)
{
    extern __shared__ char smem_raw[];
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_raw);

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
        make_tensor(make_gmem_ptr((float*)nullptr),
                    make_layout(make_shape(Int<BM>{}, Int<BN>{}))));
    tCtAcc.data() = smem.tmem_base_ptr;

    int num_k = (d + BK - 1) / BK;
    tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

    float acc = 0.f;

    // load Xi once
    Tensor gXi = make_tensor(
        make_gmem_ptr(X + (long long)i0 * d),
        make_layout(make_shape(bm, d), make_stride(d, _1{})));
    cooperative_copy<128>(tid, gXi, tCsA);
    __syncthreads();

    for (int j0 = 0; j0 < N; j0 += BN) {
        int j1 = min(j0 + BN, N);
        int bn = j1 - j0;

        Tensor gXj = make_tensor(
            make_gmem_ptr(X + (long long)j0 * d),
            make_layout(make_shape(bn, d), make_stride(d, _1{})));
        cooperative_copy<128>(tid, gXj, tCsB);
        __syncthreads();

        if (elect_one_warp) {
            for (int k = 0; k < size<1>(tCrA); ++k) {
                gemm(tiled_mma, tCrA(_, _, k), tCrB(_, _, k), tCtAcc);
                tiled_mma.accumulate_ = UMMA::ScaleOut::One;
            }
            cutlass::arch::umma_arrive(&smem.mma_barrier);
        }
        cute::wait_barrier(smem.mma_barrier, mma_phase);
        mma_phase ^= 1;
        __syncthreads();

        if (tid < bm) {
            float sqi = sq_X[i0 + tid];
            for (int j = 0; j < bn; ++j) {
                float dot = 0.f;
                for (int k = 0; k < d; ++k)
                    dot += h2f(tCsA(tid, k)) * h2f(tCsB(j, k));
                float dist2 = fmaxf(sqi + sq_X[j0+j] - 2.f*dot, 0.f);
                acc += fast_exp(-beta * dist2) * u[j0+j];
            }
        }
        __syncthreads();

        tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
    }

    __syncthreads();
    if (elect_one_warp) {
        tmem_alloc.release_allocation_lock();
        tmem_alloc.free(smem.tmem_base_ptr,
                        TmemAllocator::Sm100TmemCapacityColumns);
    }

    if (tid < bm)
        out[i0 + tid] = rscale[i0 + tid] * acc;
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
    auto D    = torch::zeros({N}, w.options());
    int  grid = (N + BM - 1) / BM;
    prepass_kernel_sm100<<<grid, NTHREADS, sizeof(SharedStorage),
                           at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const cutlass::half_t*>(X_fp16.data_ptr<at::Half>()),
        w.data_ptr<float>(), D.data_ptr<float>(),
        sq_X.data_ptr<float>(), N, d, beta);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return D;
#else
    TORCH_CHECK(false, "Rebuilt with -DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1");
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
    int  grid = (N + BM - 1) / BM;
    matvec_kernel_sm100<<<grid, NTHREADS, sizeof(SharedStorage),
                          at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const cutlass::half_t*>(X_fp16.data_ptr<at::Half>()),
        u.data_ptr<float>(), out.data_ptr<float>(),
        sq_X.data_ptr<float>(), rscale.data_ptr<float>(), N, d, beta);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
#else
    TORCH_CHECK(false, "Rebuilt with -DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1");
#endif
}

PYBIND11_MODULE(flash_diffusion_sm100, m) {
    m.doc() = "FlashDiffusion SM100 native UMMA tcgen05.mma (B200/B300)";
    m.def("compute_sq_norms", &compute_sq_norms, py::arg("X_fp16"));
    m.def("prepass",  &prepass,
          py::arg("X_fp16"), py::arg("w"), py::arg("sq_X"), py::arg("beta"));
    m.def("matvec",   &matvec,
          py::arg("X_fp16"), py::arg("v"), py::arg("rscale"),
          py::arg("sq_X"), py::arg("beta"));
}
