/*
 * flash_diffusion_sm100_umma.cu
 * ==============================
 * SM100 (B200 / Blackwell datacenter) native UMMA kernel for FlashDiffusion.
 *
 * Replaces flash_diffusion_sm100.cu (WMMA v1) with native Blackwell
 * tcgen05.mma instructions via CuTe SM100_MMA_F16BF16_SS atom.
 *
 * Key differences from WMMA v1:
 *   WMMA:  nvcuda::wmma API, accumulator in registers
 *   UMMA:  tcgen05.mma PTX, accumulator in TMEM (Tensor Memory)
 *          TMEM: 128×512 32-bit dedicated memory per SM, ~2× WMMA throughput
 *          TMA:  async GMEM→SMEM loads (replaces synchronous loads)
 *          requires CUTLASS >= 3.7 for SM100 CuTe atoms
 *
 * Architecture: single-CTA UMMA (cta_group::1)
 *   MMA tile: 128×128×16  (M=128, N=128, K=16 for fp16)
 *   Threads:  128 (one warpgroup, 4 warps)
 *   SMEM:     sXi (BM×KPAD) + sXj (BN×KPAD) fp16
 *   TMEM:     sDot (BM×BN) fp32 accumulator
 *
 * Same public API as flash_diffusion_sm100.cu:
 *   compute_sq_norms(X_fp16)
 *   prepass(X_fp16, w, sq_X, beta)
 *   matvec(X_fp16, v, rscale, sq_X, beta)
 *
 * Build:
 *   -arch=sm_100a -std=c++17 -O3
 *   -U__CUDA_NO_HALF_OPERATORS__
 *   -U__CUDA_NO_HALF_CONVERSIONS__
 *   --expt-relaxed-constexpr
 *   -I /path/to/cutlass/include      (CUTLASS >= 3.7)
 *   -DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1
 *
 * References:
 *   CUTLASS examples/cute/tutorial/blackwell/01_mma_sm100.cu
 *   https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/
 *   https://gau-nernst.github.io/tcgen05/
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// CuTe / CUTLASS SM100 headers
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_sm100_umma.hpp>    // SM100_MMA_F16BF16_SS
#include <cute/arch/copy_sm100.hpp>         // TMA for SM100
#include <cute/arch/tmem_allocator.hpp>     // TMEM allocation

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/arch.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

using namespace cute;

// ---------------------------------------------------------------------------
// Tile constants for SM100 UMMA
// ---------------------------------------------------------------------------
static constexpr int BM       = 128;   // MMA M tile (must be 128 for SM100 1-CTA)
static constexpr int BN       = 128;   // MMA N tile (must be 128 for SM100 1-CTA)
static constexpr int BK       = 16;    // MMA K tile (16 for fp16)
static constexpr int NTHREADS = 128;   // 1 warpgroup = 4 warps
static constexpr int STAGES   = 2;     // double-buffer for TMA overlap

// ---------------------------------------------------------------------------
// UMMA atom: SM100, fp16 A/B in SMEM (SS), fp32 accumulator in TMEM
// ---------------------------------------------------------------------------
using MMA_Op = SM100_MMA_F16BF16_SS<
    cutlass::half_t,    // TypeA
    cutlass::half_t,    // TypeB
    float,              // TypeC (accumulator in TMEM)
    128,                // MMA M
    128,                // MMA N
    UMMA::Major::K,     // A layout: K-major
    UMMA::Major::K      // B layout: K-major
>;
using TiledMMA_t = decltype(make_tiled_mma(MMA_Op{}));

// ---------------------------------------------------------------------------
// SMEM layouts — must match UMMA descriptor expectations
// 128-byte swizzle for fp16 K-major layout
// ---------------------------------------------------------------------------
using SmemLayoutAtom = decltype(
    composition(Swizzle<3, 4, 3>{},
                Layout<Shape<_8, Int<BK>>,
                       Stride<Int<BK>, _1>>{}));

using SmemLayoutXi = decltype(
    tile_to_shape(SmemLayoutAtom{},
                  Shape<Int<BM>, Int<BK>, Int<STAGES>>{}));

using SmemLayoutXj = decltype(
    tile_to_shape(SmemLayoutAtom{},
                  Shape<Int<BN>, Int<BK>, Int<STAGES>>{}));

// ---------------------------------------------------------------------------
// Shared + Tensor memory layout
// ---------------------------------------------------------------------------
struct SharedStorage {
    cute::array_aligned<cutlass::half_t, cosize_v<SmemLayoutXi>> sXi;
    cute::array_aligned<cutlass::half_t, cosize_v<SmemLayoutXj>> sXj;
    // TMA mbarriers — one per stage
    alignas(8) uint64_t mbar_Xi[STAGES];
    alignas(8) uint64_t mbar_Xj[STAGES];
};

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ float fast_exp(float x) { return __expf(x); }

__device__ __forceinline__ float h2f(cutlass::half_t x) {
    return __half2float(*reinterpret_cast<const __half*>(&x));
}

// ---------------------------------------------------------------------------
// Sq-norm kernel — same as SM80, no UMMA needed
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

// ---------------------------------------------------------------------------
// Prepass kernel — SM100 UMMA version
//
// Per (i-tile, j-tile):
//   1. TMA load Xi → sXi  (async, mbarrier)
//   2. TMA load Xj → sXj  (async, mbarrier)
//   3. UMMA: tcgen05.mma → TMEM accumulator (dot products Xi @ Xj^T)
//   4. tcgen05.ld: read TMEM accumulator to registers
//   5. Epilogue: dist2 = sq_i + sq_j - 2*dot → exp → accumulate
//   6. tcgen05.dealloc TMEM before kernel exit
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(NTHREADS)
prepass_kernel_sm100(
    const cutlass::half_t* __restrict__ X,
    const float*           __restrict__ w,
    float*                 __restrict__ D,
    const float*           __restrict__ sq_X,
    int N, int d, float beta,
    const __grid_constant__ CUtensorMap tma_Xi,
    const __grid_constant__ CUtensorMap tma_Xj)
{
    extern __shared__ char smem_raw[];
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_raw);

    int tid = threadIdx.x;
    int i0  = blockIdx.x * BM;
    int i1  = min(i0 + BM, N);
    int bm  = i1 - i0;

    // ── TMEM allocation ──────────────────────────────────────────────────────
    // Must allocate before use and deallocate before exit
    // TMEM holds BM×BN fp32 accumulator (128×128×4 = 65536 bytes)
    cute::TmemAllocator tmem_alloc;
    auto tmem_ptr = tmem_alloc.allocate(BM * BN);   // returns TMEM address

    // ── SMEM tensors ─────────────────────────────────────────────────────────
    Tensor sXi_full = make_tensor(make_smem_ptr(smem.sXi.data()), SmemLayoutXi{});
    Tensor sXj_full = make_tensor(make_smem_ptr(smem.sXj.data()), SmemLayoutXj{});

    // ── init mbarriers ───────────────────────────────────────────────────────
    if (tid == 0) {
        for (int s = 0; s < STAGES; ++s) {
            cutlass::arch::fence_barrier_init();
        }
    }
    __syncthreads();

    TiledMMA_t tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);

    float D_acc = 0.f;   // per-thread output accumulator (for tid < bm)

    // ── j-tile loop ──────────────────────────────────────────────────────────
    for (int j0 = 0; j0 < N; j0 += BN) {
        int j1  = min(j0 + BN, N);
        int bn  = j1 - j0;
        int stage = (j0 / BN) % STAGES;

        Tensor sXi = sXi_full(_, _, stage);
        Tensor sXj = sXj_full(_, _, stage);

        // TMA loads — thread 0 issues
        if (tid == 0) {
            if (j0 == 0) {
                // load Xi once (fixed across j-loop)
                cutlass::arch::cp_async_bulk_tensor_2d_global_to_shared(
                    smem.sXi.data(), &tma_Xi, i0, 0, smem.mbar_Xi[stage]);
            }
            cutlass::arch::cp_async_bulk_tensor_2d_global_to_shared(
                smem.sXj.data(), &tma_Xj, j0, 0, smem.mbar_Xj[stage]);
        }
        cutlass::arch::fence_view_async_shared();
        __syncthreads();

        if (j0 == 0)
            cutlass::arch::wait_barrier(smem.mbar_Xi[stage], 0);
        cutlass::arch::wait_barrier(smem.mbar_Xj[stage], 0);
        __syncthreads();

        // ── UMMA: Xi @ Xj^T → TMEM ──────────────────────────────────────────
        // Zero TMEM accumulator before first use
        auto tCtAcc = thr_mma.partition_C(
            make_tensor(make_tmem_ptr<float>(tmem_ptr),
                        Layout<Shape<Int<BM>, Int<BN>>>{}));

        // fence before UMMA reads SMEM
        cutlass::arch::fence_view_async_shared();
        cutlass::arch::umma_arrive();

        auto tCsXi = thr_mma.partition_A(sXi);
        auto tCsXj = thr_mma.partition_B(sXj);

        // clear TMEM accumulator
        clear(tCtAcc);

        // k-loop over coordinate dimension
        int num_k = (d + BK - 1) / BK;
        CUTE_UNROLL
        for (int k = 0; k < num_k; ++k) {
            cute::gemm(tiled_mma, tCtAcc,
                       tCsXi(_, _, k),
                       tCsXj(_, _, k),
                       tCtAcc);
        }

        cutlass::arch::umma_commit();
        cutlass::arch::umma_wait<0>();
        __syncthreads();

        // ── tcgen05.ld: read TMEM → registers + epilogue ─────────────────────
        // Load BM×BN dot products from TMEM to registers
        // Each thread reads its own rows
        if (tid < bm) {
            float sqi = sq_X[i0 + tid];

            // read dot products for row tid from TMEM
            // TMEM layout: row-major, tid-th row
            for (int j = 0; j < bn; ++j) {
                // tcgen05.ld: load one fp32 from TMEM
                float dot;
                cutlass::arch::tmem_load(dot,
                    tmem_ptr + (tid * BN + j) * sizeof(float));

                float dist2 = fmaxf(sqi + sq_X[j0 + j] - 2.f * dot, 0.f);
                D_acc += fast_exp(-beta * dist2) * w[j0 + j];
            }
        }
        __syncthreads();
    }

    // ── TMEM deallocation — MUST happen before kernel exit ───────────────────
    tmem_alloc.deallocate(tmem_ptr, BM * BN);

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
    int N, int d, float beta,
    const __grid_constant__ CUtensorMap tma_Xi,
    const __grid_constant__ CUtensorMap tma_Xj)
{
    extern __shared__ char smem_raw[];
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_raw);

    int tid = threadIdx.x;
    int i0  = blockIdx.x * BM;
    int i1  = min(i0 + BM, N);
    int bm  = i1 - i0;

    cute::TmemAllocator tmem_alloc;
    auto tmem_ptr = tmem_alloc.allocate(BM * BN);

    Tensor sXi_full = make_tensor(make_smem_ptr(smem.sXi.data()), SmemLayoutXi{});
    Tensor sXj_full = make_tensor(make_smem_ptr(smem.sXj.data()), SmemLayoutXj{});

    if (tid == 0) {
        for (int s = 0; s < STAGES; ++s)
            cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    TiledMMA_t tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);

    float acc = 0.f;

    for (int j0 = 0; j0 < N; j0 += BN) {
        int j1    = min(j0 + BN, N);
        int bn    = j1 - j0;
        int stage = (j0 / BN) % STAGES;

        Tensor sXi = sXi_full(_, _, stage);
        Tensor sXj = sXj_full(_, _, stage);

        if (tid == 0) {
            if (j0 == 0)
                cutlass::arch::cp_async_bulk_tensor_2d_global_to_shared(
                    smem.sXi.data(), &tma_Xi, i0, 0, smem.mbar_Xi[stage]);
            cutlass::arch::cp_async_bulk_tensor_2d_global_to_shared(
                smem.sXj.data(), &tma_Xj, j0, 0, smem.mbar_Xj[stage]);
        }
        cutlass::arch::fence_view_async_shared();
        __syncthreads();

        if (j0 == 0) cutlass::arch::wait_barrier(smem.mbar_Xi[stage], 0);
        cutlass::arch::wait_barrier(smem.mbar_Xj[stage], 0);
        __syncthreads();

        auto tCtAcc = thr_mma.partition_C(
            make_tensor(make_tmem_ptr<float>(tmem_ptr),
                        Layout<Shape<Int<BM>, Int<BN>>>{}));

        cutlass::arch::fence_view_async_shared();
        cutlass::arch::umma_arrive();

        auto tCsXi = thr_mma.partition_A(sXi);
        auto tCsXj = thr_mma.partition_B(sXj);

        clear(tCtAcc);

        int num_k = (d + BK - 1) / BK;
        CUTE_UNROLL
        for (int k = 0; k < num_k; ++k) {
            cute::gemm(tiled_mma, tCtAcc,
                       tCsXi(_, _, k),
                       tCsXj(_, _, k),
                       tCtAcc);
        }

        cutlass::arch::umma_commit();
        cutlass::arch::umma_wait<0>();
        __syncthreads();

        if (tid < bm) {
            float sqi = sq_X[i0 + tid];
            for (int j = 0; j < bn; ++j) {
                float dot;
                cutlass::arch::tmem_load(dot,
                    tmem_ptr + (tid * BN + j) * sizeof(float));
                float dist2 = fmaxf(sqi + sq_X[j0+j] - 2.f*dot, 0.f);
                acc += fast_exp(-beta * dist2) * u[j0+j];
            }
        }
        __syncthreads();
    }

    tmem_alloc.deallocate(tmem_ptr, BM * BN);

    if (tid < bm)
        out[i0 + tid] = rscale[i0 + tid] * acc;
}

// ---------------------------------------------------------------------------
// Host-side TMA descriptor
// ---------------------------------------------------------------------------
static CUtensorMap make_tma_desc(const void* ptr, int N, int d) {
    CUtensorMap desc{};
    cuuint64_t shape[2]  = {(cuuint64_t)d, (cuuint64_t)N};
    cuuint64_t stride[1] = {(cuuint64_t)d * sizeof(cutlass::half_t)};
    cuuint32_t tile[2]   = {(cuuint32_t)BK, (cuuint32_t)BM};
    cuuint32_t elem[1]   = {1};

    CUresult rc = cuTensorMapEncodeTiled(
        &desc,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        2,
        const_cast<void*>(ptr),
        shape, stride, tile, elem,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    TORCH_CHECK(rc == CUDA_SUCCESS,
        "cuTensorMapEncodeTiled failed: ", rc);
    return desc;
}

// ---------------------------------------------------------------------------
// Python entry points — same API as WMMA v1
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
    int N = X_fp16.size(0), d = X_fp16.size(1);
    auto D    = torch::zeros({N}, w.options());
    int  grid = (N + BM - 1) / BM;

    auto desc_i = make_tma_desc(X_fp16.data_ptr<at::Half>(), N, d);
    auto desc_j = make_tma_desc(X_fp16.data_ptr<at::Half>(), N, d);

    prepass_kernel_sm100<<<grid, NTHREADS, sizeof(SharedStorage),
                           at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const cutlass::half_t*>(X_fp16.data_ptr<at::Half>()),
        w.data_ptr<float>(), D.data_ptr<float>(),
        sq_X.data_ptr<float>(), N, d, beta, desc_i, desc_j);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return D;
}

torch::Tensor matvec(
    torch::Tensor X_fp16, torch::Tensor v,
    torch::Tensor rscale, torch::Tensor sq_X, float beta)
{
    int N = X_fp16.size(0), d = X_fp16.size(1);
    auto u   = rscale * v;
    auto out = torch::empty({N}, v.options());
    int  grid = (N + BM - 1) / BM;

    auto desc_i = make_tma_desc(X_fp16.data_ptr<at::Half>(), N, d);
    auto desc_j = make_tma_desc(X_fp16.data_ptr<at::Half>(), N, d);

    matvec_kernel_sm100<<<grid, NTHREADS, sizeof(SharedStorage),
                          at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const cutlass::half_t*>(X_fp16.data_ptr<at::Half>()),
        u.data_ptr<float>(), out.data_ptr<float>(),
        sq_X.data_ptr<float>(), rscale.data_ptr<float>(),
        N, d, beta, desc_i, desc_j);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// ---------------------------------------------------------------------------
// pybind11
// ---------------------------------------------------------------------------
PYBIND11_MODULE(flash_diffusion_sm100, m) {
    m.doc() = "FlashDiffusion SM100 — native UMMA tcgen05.mma (B200/B300)";
    m.def("compute_sq_norms", &compute_sq_norms, py::arg("X_fp16"));
    m.def("prepass",  &prepass,
          py::arg("X_fp16"), py::arg("w"), py::arg("sq_X"), py::arg("beta"));
    m.def("matvec",   &matvec,
          py::arg("X_fp16"), py::arg("v"), py::arg("rscale"),
          py::arg("sq_X"), py::arg("beta"));
}
