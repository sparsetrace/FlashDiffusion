/*
 * flash_diffusion_sm80.cu
 * =======================
 * FlashDiffusion WMMA tensor core kernel.
 * Targets: SM80 (A100), SM86 (RTX 3090), SM89 (RTX 4090),
 *          SM120 (RTX 5090 / RTX PRO 6000 Blackwell consumer)
 *
 * Uses nvcuda::wmma — portable from SM70 (Volta) through SM120.
 * Replaces the scalar flash_diffusion_sm80.cu baseline.
 *
 * Tile: BM=64, BN=64, 16 warps = 512 threads
 * WMMA: 16×16×16 fp16 input, fp32 accumulator
 *
 * Same public API as all FlashDiffusion kernels:
 *   compute_sq_norms(X_fp16)
 *   prepass(X_fp16, w, sq_X, beta)
 *   matvec(X_fp16, v, rscale, sq_X, beta)
 *
 * Build:
 *   SM80/86/89: -arch=sm_80  -std=c++17 -O3 --use_fast_math
 *   SM120/121:  -arch=sm_120 -std=c++17 -O3 --use_fast_math
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
static constexpr int WMMA_M   = 16;
static constexpr int WMMA_N   = 16;
static constexpr int WMMA_K   = 16;
static constexpr int BM       = 64;
static constexpr int BN       = 64;
static constexpr int WARPS_M  = BM / WMMA_M;     // 4
static constexpr int WARPS_N  = BN / WMMA_N;     // 4
static constexpr int NWARPS   = WARPS_M * WARPS_N;  // 16
static constexpr int NTHREADS = NWARPS * 32;      // 512

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }
static inline int round_up(int a, int b) { return ceil_div(a, b) * b; }

// Shared memory: sA[BM×KPAD] + sB[BN×KPAD] + sDot[BM×BN]
static size_t smem_size(int d) {
    int KPAD = round_up(d, WMMA_K);
    return (size_t)BM * KPAD * sizeof(half)
         + (size_t)BN * KPAD * sizeof(half)
         + (size_t)BM * BN   * sizeof(float);
}

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
// SMEM loaders
// sA: [BM, KPAD] row-major    → matrix_a row_major
// sB: [BN, KPAD] stored as col × k  → matrix_b col_major (= B^T row-major)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void load_sA(
    half* sA, const half* X, int i0, int bm, int d, int KPAD, int tid)
{
    for (int idx = tid; idx < BM * KPAD; idx += NTHREADS) {
        int r = idx / KPAD, c = idx % KPAD;
        sA[idx] = (r < bm && c < d) ? X[(long long)(i0+r)*d + c] : __float2half(0.f);
    }
}

__device__ __forceinline__ void load_sB(
    half* sB, const half* X, int j0, int bn, int d, int KPAD, int tid)
{
    for (int idx = tid; idx < BN * KPAD; idx += NTHREADS) {
        int col = idx / KPAD, k = idx % KPAD;
        sB[idx] = (col < bn && k < d) ? X[(long long)(j0+col)*d + k] : __float2half(0.f);
    }
}

// ---------------------------------------------------------------------------
// WMMA dot tile: sA (BM×KPAD, row) @ sB (KPAD×BN, col) → sDot (BM×BN, row)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void wmma_dot_tile(
    const half* sA, const half* sB, float* sDot, int KPAD, int tid)
{
    int warp = tid >> 5;
    int wm   = warp / WARPS_N;
    int wn   = warp % WARPS_N;

    wmma::fragment<wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fc;
    wmma::fill_fragment(fc, 0.f);

    for (int kk = 0; kk < KPAD; kk += WMMA_K) {
        wmma::load_matrix_sync(fa, sA + wm * WMMA_M * KPAD + kk, KPAD);
        wmma::load_matrix_sync(fb, sB + wn * WMMA_N * KPAD + kk, KPAD);
        wmma::mma_sync(fc, fa, fb, fc);
    }
    wmma::store_matrix_sync(sDot + wm * WMMA_M * BN + wn * WMMA_N,
                            fc, BN, wmma::mem_row_major);
}

// ---------------------------------------------------------------------------
// Prepass: D_i = Σ_j exp(-β||xi-xj||²) * w_j
// ---------------------------------------------------------------------------
__global__ void prepass_kernel(
    const half*  __restrict__ X,
    const float* __restrict__ w,
    float*       __restrict__ D,
    const float* __restrict__ sq_X,
    int N, int d, int KPAD, float beta)
{
    extern __shared__ unsigned char smem[];
    half*  sA   = reinterpret_cast<half*>(smem);
    half*  sB   = sA + BM * KPAD;
    float* sDot = reinterpret_cast<float*>(sB + BN * KPAD);

    const int tid = threadIdx.x;
    const int i0  = blockIdx.x * BM;
    const int bm  = min(BM, N - i0);
    float acc = 0.f;

    load_sA(sA, X, i0, bm, d, KPAD, tid);
    __syncthreads();

    for (int j0 = 0; j0 < N; j0 += BN) {
        const int bn = min(BN, N - j0);

        load_sB(sB, X, j0, bn, d, KPAD, tid);
        __syncthreads();

        wmma_dot_tile(sA, sB, sDot, KPAD, tid);
        __syncthreads();

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
// Matvec: out_i = rscale_i * Σ_j exp(-β||xi-xj||²) * u_j
// ---------------------------------------------------------------------------
__global__ void matvec_kernel(
    const half*  __restrict__ X,
    const float* __restrict__ u,
    float*       __restrict__ out,
    const float* __restrict__ sq_X,
    const float* __restrict__ rscale,
    int N, int d, int KPAD, float beta)
{
    extern __shared__ unsigned char smem[];
    half*  sA   = reinterpret_cast<half*>(smem);
    half*  sB   = sA + BM * KPAD;
    float* sDot = reinterpret_cast<float*>(sB + BN * KPAD);

    const int tid = threadIdx.x;
    const int i0  = blockIdx.x * BM;
    const int bm  = min(BM, N - i0);
    float acc = 0.f;

    load_sA(sA, X, i0, bm, d, KPAD, tid);
    __syncthreads();

    for (int j0 = 0; j0 < N; j0 += BN) {
        const int bn = min(BN, N - j0);

        load_sB(sB, X, j0, bn, d, KPAD, tid);
        __syncthreads();

        wmma_dot_tile(sA, sB, sDot, KPAD, tid);
        __syncthreads();

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
// SMEM opt-in helpers
// ---------------------------------------------------------------------------
static void set_smem(const void* fn, size_t bytes) {
    int dev; C10_CUDA_CHECK(cudaGetDevice(&dev));
    int maxoptin;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &maxoptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    TORCH_CHECK((int)bytes <= maxoptin,
        "SMEM request ", bytes, " exceeds device max ", maxoptin);
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes));
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
    int KPAD = round_up(d, WMMA_K);
    size_t sm = smem_size(d);
    set_smem(reinterpret_cast<const void*>(prepass_kernel), sm);

    auto D = torch::zeros({N}, w.options());
    prepass_kernel<<<ceil_div(N,BM), NTHREADS, sm,
                     at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(X_fp16.data_ptr<at::Half>()),
        w.data_ptr<float>(), D.data_ptr<float>(),
        sq_X.data_ptr<float>(), N, d, KPAD, beta);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return D;
}

torch::Tensor matvec(
    torch::Tensor X_fp16, torch::Tensor v,
    torch::Tensor rscale, torch::Tensor sq_X, float beta)
{
    int N = X_fp16.size(0), d = X_fp16.size(1);
    int KPAD = round_up(d, WMMA_K);
    size_t sm = smem_size(d);
    set_smem(reinterpret_cast<const void*>(matvec_kernel), sm);

    auto u   = (rscale * v).contiguous();
    auto out = torch::empty({N}, v.options());
    matvec_kernel<<<ceil_div(N,BM), NTHREADS, sm,
                    at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(X_fp16.data_ptr<at::Half>()),
        u.data_ptr<float>(), out.data_ptr<float>(),
        sq_X.data_ptr<float>(), rscale.data_ptr<float>(),
        N, d, KPAD, beta);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(flash_diffusion_cuda, m) {
    m.doc() = "FlashDiffusion WMMA tensor cores — SM80/SM86/SM89/SM120/SM121";
    m.def("compute_sq_norms", &compute_sq_norms, py::arg("X_fp16"));
    m.def("prepass",  &prepass,
          py::arg("X_fp16"), py::arg("w"), py::arg("sq_X"), py::arg("beta"));
    m.def("matvec",   &matvec,
          py::arg("X_fp16"), py::arg("v"), py::arg("rscale"),
          py::arg("sq_X"), py::arg("beta"));
}
