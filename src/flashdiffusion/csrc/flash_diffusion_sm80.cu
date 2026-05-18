/*
 * flash_diffusion_sm80.cu  — WMMA v2
 * ===================================
 * FlashDiffusion WMMA tensor core kernel.
 * Targets: SM80 (A100), SM86 (RTX 3090), SM89 (RTX 4090),
 *          SM120 (RTX 5090 / RTX PRO 6000) via fallback
 *
 * v2 over v1:
 *   Fixed SMEM size via k-chunking — supports arbitrary d (including d=784).
 *   v1 allocated BM×KPAD SMEM which exceeded 96KB A100 limit for d > ~400.
 *
 * SMEM layout (fixed, independent of d):
 *   sA   = BM × KCHUNK × 2  = 64 × 32 × 2 =  4,096 bytes
 *   sB   = BN × KCHUNK × 2  = 64 × 32 × 2 =  4,096 bytes
 *   sDot = BM × BN    × 4   = 64 × 64  × 4 = 16,384 bytes
 *   Total = 24,576 bytes = 24KB  ✓  (well under 96KB A100 limit)
 *
 * K-chunking: loop over d in steps of KCHUNK=32.
 *   Each chunk: load sA[BM×32], sB[BN×32], run 1 WMMA call per warp,
 *   accumulate into sDot (load fc from sDot on non-first chunks).
 *
 * Build:
 *   SM80/86/89: -arch=sm_80  -std=c++17 -O3 --use_fast_math
 *   -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__
 *   --expt-relaxed-constexpr
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

using namespace nvcuda;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr int WMMA_M  = 16;
static constexpr int WMMA_N  = 16;
static constexpr int WMMA_K  = 16;

static constexpr int BM      = 64;
static constexpr int BN      = 64;
static constexpr int KCHUNK  = 32;    // process d in 32-wide chunks (2 WMMA_K)

// 16 warps: each covers one 16×16 WMMA tile
static constexpr int WARPS_M  = BM / WMMA_M;        // 4
static constexpr int WARPS_N  = BN / WMMA_N;        // 4
static constexpr int NWARPS   = WARPS_M * WARPS_N;  // 16
static constexpr int NTHREADS = NWARPS * 32;          // 512

// Fixed SMEM — 24KB, works for any d
static constexpr size_t SMEM_BYTES =
    (size_t)BM * KCHUNK * sizeof(half)    // sA
  + (size_t)BN * KCHUNK * sizeof(half)    // sB
  + (size_t)BM * BN     * sizeof(float);  // sDot

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------------
// Squared norms
// ---------------------------------------------------------------------------
__global__ void sq_norm_kernel(
    const half* __restrict__ X, float* __restrict__ sq, int N, int d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s = 0.f;
    for (int k = 0; k < d; ++k) { float v = __half2float(X[i*d+k]); s += v*v; }
    sq[i] = s;
}

// ---------------------------------------------------------------------------
// SMEM loaders for a k-chunk [k0, k0+klen)
// sA: [BM, KCHUNK] row-major   → matrix_a row_major
// sB: [BN, KCHUNK] col-major   → matrix_b col_major (= B^T)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void load_sA_chunk(
    half* sA, const half* X, int i0, int bm, int d,
    int k0, int klen, int tid)
{
    for (int idx = tid; idx < BM * KCHUNK; idx += NTHREADS) {
        int r = idx / KCHUNK, c = idx % KCHUNK;
        sA[idx] = (r < bm && c < klen)
            ? X[(long long)(i0+r)*d + k0+c]
            : __float2half(0.f);
    }
}

__device__ __forceinline__ void load_sB_chunk(
    half* sB, const half* X, int j0, int bn, int d,
    int k0, int klen, int tid)
{
    for (int idx = tid; idx < BN * KCHUNK; idx += NTHREADS) {
        int col = idx / KCHUNK, c = idx % KCHUNK;
        sB[idx] = (col < bn && c < klen)
            ? X[(long long)(j0+col)*d + k0+c]
            : __float2half(0.f);
    }
}

// ---------------------------------------------------------------------------
// WMMA accumulate: sA (BM×KCHUNK, row) @ sB^T (KCHUNK×BN, col) → sDot
// Each warp covers one 16×16 WMMA tile.
// first_chunk=true:  initialize fc=0
// first_chunk=false: load fc from sDot first (accumulate across k-chunks)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void wmma_accumulate(
    const half* sA, const half* sB, float* sDot,
    bool first_chunk, int tid)
{
    const int warp = tid >> 5;
    const int wm   = warp / WARPS_N;
    const int wn   = warp % WARPS_N;
    const int row16 = wm * WMMA_M;
    const int col16 = wn * WMMA_N;

    wmma::fragment<wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K,
                   half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K,
                   half, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fc;

    if (first_chunk) {
        wmma::fill_fragment(fc, 0.f);
    } else {
        wmma::load_matrix_sync(fc,
            sDot + row16 * BN + col16, BN, wmma::mem_row_major);
    }

    for (int kk = 0; kk < KCHUNK; kk += WMMA_K) {
        wmma::load_matrix_sync(fa, sA + row16 * KCHUNK + kk, KCHUNK);
        wmma::load_matrix_sync(fb, sB + col16 * KCHUNK + kk, KCHUNK);
        wmma::mma_sync(fc, fa, fb, fc);
    }

    wmma::store_matrix_sync(
        sDot + row16 * BN + col16, fc, BN, wmma::mem_row_major);
}

// ---------------------------------------------------------------------------
// Prepass kernel: D_i = Σ_j exp(-β||xi-xj||²) * w_j
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(NTHREADS)
prepass_kernel(
    const half*  __restrict__ X,
    const float* __restrict__ w,
    float*       __restrict__ D,
    const float* __restrict__ sq_X,
    int N, int d, float beta)
{
    extern __shared__ unsigned char smem[];
    half*  sA   = reinterpret_cast<half*>(smem);
    half*  sB   = sA + BM * KCHUNK;
    float* sDot = reinterpret_cast<float*>(sB + BN * KCHUNK);

    const int tid = threadIdx.x;
    const int i0  = blockIdx.x * BM;
    const int bm  = min(BM, N - i0);
    float acc = 0.f;

    for (int j0 = 0; j0 < N; j0 += BN) {
        const int bn = min(BN, N - j0);

        // k-chunk loop: accumulate dot products into sDot
        for (int k0 = 0; k0 < d; k0 += KCHUNK) {
            const int klen = min(KCHUNK, d - k0);
            const bool first = (k0 == 0);

            load_sA_chunk(sA, X, i0, bm, d, k0, klen, tid);
            load_sB_chunk(sB, X, j0, bn, d, k0, klen, tid);
            __syncthreads();

            wmma_accumulate(sA, sB, sDot, first, tid);
            __syncthreads();
        }

        // epilogue: dist2 → exp → accumulate
        if (tid < bm) {
            float sqi = sq_X[i0 + tid];
            #pragma unroll 4
            for (int j = 0; j < bn; ++j) {
                float dist2 = fmaxf(sqi + sq_X[j0+j] - 2.f * sDot[tid*BN+j], 0.f);
                acc += __expf(-beta * dist2) * w[j0+j];
            }
        }
        __syncthreads();
    }

    if (tid < bm) D[i0 + tid] = acc;
}

// ---------------------------------------------------------------------------
// Matvec kernel: out_i = rscale_i * Σ_j exp(-β||xi-xj||²) * u_j
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(NTHREADS)
matvec_kernel(
    const half*  __restrict__ X,
    const float* __restrict__ u,
    float*       __restrict__ out,
    const float* __restrict__ sq_X,
    const float* __restrict__ rscale,
    int N, int d, float beta)
{
    extern __shared__ unsigned char smem[];
    half*  sA   = reinterpret_cast<half*>(smem);
    half*  sB   = sA + BM * KCHUNK;
    float* sDot = reinterpret_cast<float*>(sB + BN * KCHUNK);

    const int tid = threadIdx.x;
    const int i0  = blockIdx.x * BM;
    const int bm  = min(BM, N - i0);
    float acc = 0.f;

    for (int j0 = 0; j0 < N; j0 += BN) {
        const int bn = min(BN, N - j0);

        for (int k0 = 0; k0 < d; k0 += KCHUNK) {
            const int klen = min(KCHUNK, d - k0);
            const bool first = (k0 == 0);

            load_sA_chunk(sA, X, i0, bm, d, k0, klen, tid);
            load_sB_chunk(sB, X, j0, bn, d, k0, klen, tid);
            __syncthreads();

            wmma_accumulate(sA, sB, sDot, first, tid);
            __syncthreads();
        }

        if (tid < bm) {
            float sqi = sq_X[i0 + tid];
            #pragma unroll 4
            for (int j = 0; j < bn; ++j) {
                float dist2 = fmaxf(sqi + sq_X[j0+j] - 2.f * sDot[tid*BN+j], 0.f);
                acc += __expf(-beta * dist2) * u[j0+j];
            }
        }
        __syncthreads();
    }

    if (tid < bm) out[i0 + tid] = rscale[i0 + tid] * acc;
}

// ---------------------------------------------------------------------------
// SMEM opt-in
// ---------------------------------------------------------------------------
static void set_smem(const void* fn) {
    int dev; C10_CUDA_CHECK(cudaGetDevice(&dev));
    int maxoptin;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &maxoptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    TORCH_CHECK((int)SMEM_BYTES <= maxoptin,
        "SMEM ", SMEM_BYTES, " > device max ", maxoptin);
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES));
}

// ---------------------------------------------------------------------------
// Python entry points
// ---------------------------------------------------------------------------

torch::Tensor compute_sq_norms(torch::Tensor X_fp16) {
    TORCH_CHECK(X_fp16.is_cuda() && X_fp16.scalar_type() == at::kHalf,
                "X_fp16 must be a CUDA half tensor");
    int N = X_fp16.size(0), d = X_fp16.size(1);
    auto sq = torch::empty({N}, X_fp16.options().dtype(at::kFloat));
    sq_norm_kernel<<<ceil_div(N,256), 256, 0,
                     at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(X_fp16.data_ptr<at::Half>()),
        sq.data_ptr<float>(), N, d);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return sq;
}

torch::Tensor prepass(
    torch::Tensor X_fp16, torch::Tensor w,
    torch::Tensor sq_X,   float beta)
{
    int N = X_fp16.size(0), d = X_fp16.size(1);
    set_smem(reinterpret_cast<const void*>(prepass_kernel));

    auto D = torch::zeros({N}, w.options());
    prepass_kernel<<<ceil_div(N,BM), NTHREADS, SMEM_BYTES,
                     at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(X_fp16.data_ptr<at::Half>()),
        w.data_ptr<float>(), D.data_ptr<float>(),
        sq_X.data_ptr<float>(), N, d, beta);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return D;
}

torch::Tensor matvec(
    torch::Tensor X_fp16, torch::Tensor v,
    torch::Tensor rscale, torch::Tensor sq_X, float beta)
{
    int N = X_fp16.size(0), d = X_fp16.size(1);
    set_smem(reinterpret_cast<const void*>(matvec_kernel));

    auto u   = (rscale * v).contiguous();
    auto out = torch::empty({N}, v.options());
    matvec_kernel<<<ceil_div(N,BM), NTHREADS, SMEM_BYTES,
                    at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(X_fp16.data_ptr<at::Half>()),
        u.data_ptr<float>(), out.data_ptr<float>(),
        sq_X.data_ptr<float>(), rscale.data_ptr<float>(),
        N, d, beta);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashDiffusion WMMA v2 — k-chunked, arbitrary d "
              "(SM80/SM86/SM89 A100/RTX3090/RTX4090)";
    m.def("compute_sq_norms", &compute_sq_norms, py::arg("X_fp16"));
    m.def("prepass",  &prepass,
          py::arg("X_fp16"), py::arg("w"), py::arg("sq_X"), py::arg("beta"));
    m.def("matvec",   &matvec,
          py::arg("X_fp16"), py::arg("v"), py::arg("rscale"),
          py::arg("sq_X"), py::arg("beta"));
}
