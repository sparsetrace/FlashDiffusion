/*
 * flash_diffusion_sm120.cu  — fixed v2
 * =====================================
 * SM120 (RTX 5090 / RTX PRO 6000 Blackwell) kernel.
 *
 * Changes from v1:
 *   - Added #include <cuda_pipeline.h>
 *   - Fixed j0 offset bug in initial Xj prefetch
 *   - Simplified to single Xj buffer (no broken double-buffer)
 *   - Xi loaded once before j-loop (it's fixed across j)
 *   - Clean single-stage cp.async: issue → syncthreads → compute
 *
 * SM120 vs SM80:
 *   - Same mma.sync instruction family (no wgmma, no TMEM)
 *   - Larger tiles: BM=128, BN=128 vs BM=64, BN=64
 *   - cp.async for GMEM→SMEM (hides load latency)
 *   - No multicast, cluster shape fixed 1×1×1
 *
 * Build: -arch=sm_120 -std=c++17 --use_fast_math
 *        -U__CUDA_NO_HALF_OPERATORS__ --expt-relaxed-constexpr
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>       // __pipeline_memcpy_async etc.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

static constexpr int BM       = 128;
static constexpr int BN       = 128;
static constexpr int NTHREADS = 256;   // 8 warps

__device__ __forceinline__ float fast_exp(float x) { return __expf(x); }

// ---------------------------------------------------------------------------
// sq-norm kernel
// ---------------------------------------------------------------------------
__global__ void sq_norm_kernel_sm120(
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
// prepass kernel: D_i = Σ_j exp(-beta * dist2(i,j)) * w_j
//
// Structure:
//   1. Load Xi tile into sXi via cp.async (once, fixed across j-loop)
//   2. For each Xj tile:
//      a. Load Xj tile into sXj via cp.async
//      b. __syncthreads to wait for both loads
//      c. Scalar dist2 + exp + accumulate
// ---------------------------------------------------------------------------
__global__ void prepass_kernel_sm120(
    const __half* __restrict__ X,
    const float*  __restrict__ w,
    float*        __restrict__ D,
    const float*  __restrict__ sq_X,
    int N, int d, float beta)
{
    // sXi: (BM, d)   sXj: (BN, d)
    extern __shared__ __half smem[];
    __half* sXi = smem;
    __half* sXj = sXi + BM * d;

    int tid = threadIdx.x;
    int i0  = blockIdx.x * BM;
    int i1  = min(i0 + BM, N);
    int bm  = i1 - i0;

    float acc = 0.f;

    // ── load Xi tile once (fixed for all j) ─────────────────────────────
    for (int idx = tid; idx < bm * d; idx += NTHREADS) {
        int r = idx / d, c = idx % d;
        __pipeline_memcpy_async(
            &sXi[r * d + c],
            &X[(i0 + r) * d + c],
            sizeof(__half));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    // ── loop over j tiles ────────────────────────────────────────────────
    for (int j0 = 0; j0 < N; j0 += BN) {
        int j1 = min(j0 + BN, N);
        int bn = j1 - j0;

        // load Xj tile
        for (int idx = tid; idx < bn * d; idx += NTHREADS) {
            int r = idx / d, c = idx % d;
            __pipeline_memcpy_async(
                &sXj[r * d + c],
                &X[(j0 + r) * d + c],   // ← correct: j0 offset
                sizeof(__half));
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();

        // compute
        if (tid < bm) {
            float sqi = sq_X[i0 + tid];
            for (int j = 0; j < bn; ++j) {
                float dot = 0.f;
                for (int k = 0; k < d; ++k)
                    dot += __half2float(sXi[tid * d + k])
                         * __half2float(sXj[j   * d + k]);
                float dist2 = fmaxf(sqi + sq_X[j0 + j] - 2.f * dot, 0.f);
                acc += fast_exp(-beta * dist2) * w[j0 + j];
            }
        }
        __syncthreads();
    }

    if (tid < bm) D[i0 + tid] = acc;
}

// ---------------------------------------------------------------------------
// matvec kernel: out_i = rscale_i * Σ_j K(i,j) * rscale_j * v_j
// ---------------------------------------------------------------------------
__global__ void matvec_kernel_sm120(
    const __half* __restrict__ X,
    const float*  __restrict__ u,       // u = rscale * v, pre-scaled
    float*        __restrict__ out,
    const float*  __restrict__ sq_X,
    const float*  __restrict__ rscale,
    int N, int d, float beta)
{
    extern __shared__ __half smem[];
    __half* sXi = smem;
    __half* sXj = sXi + BM * d;

    int tid = threadIdx.x;
    int i0  = blockIdx.x * BM;
    int i1  = min(i0 + BM, N);
    int bm  = i1 - i0;

    float acc = 0.f;

    // load Xi once
    for (int idx = tid; idx < bm * d; idx += NTHREADS) {
        int r = idx / d, c = idx % d;
        __pipeline_memcpy_async(
            &sXi[r * d + c],
            &X[(i0 + r) * d + c],
            sizeof(__half));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    for (int j0 = 0; j0 < N; j0 += BN) {
        int j1 = min(j0 + BN, N);
        int bn = j1 - j0;

        for (int idx = tid; idx < bn * d; idx += NTHREADS) {
            int r = idx / d, c = idx % d;
            __pipeline_memcpy_async(
                &sXj[r * d + c],
                &X[(j0 + r) * d + c],   // ← correct: j0 offset
                sizeof(__half));
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();

        if (tid < bm) {
            float sqi = sq_X[i0 + tid];
            for (int j = 0; j < bn; ++j) {
                float dot = 0.f;
                for (int k = 0; k < d; ++k)
                    dot += __half2float(sXi[tid * d + k])
                         * __half2float(sXj[j   * d + k]);
                float dist2 = fmaxf(sqi + sq_X[j0 + j] - 2.f * dot, 0.f);
                acc += fast_exp(-beta * dist2) * u[j0 + j];
            }
        }
        __syncthreads();
    }

    if (tid < bm) out[i0 + tid] = rscale[i0 + tid] * acc;
}

// ---------------------------------------------------------------------------
// Python entry points
// ---------------------------------------------------------------------------

torch::Tensor compute_sq_norms(torch::Tensor X_fp16) {
    int N = X_fp16.size(0), d = X_fp16.size(1);
    auto sq = torch::empty({N}, X_fp16.options().dtype(torch::kFloat32));
    sq_norm_kernel_sm120<<<(N+255)/256, 256, 0,
                           at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(X_fp16.data_ptr<at::Half>()),
        sq.data_ptr<float>(), N, d);
    return sq;
}

torch::Tensor prepass(
    torch::Tensor X_fp16, torch::Tensor w,
    torch::Tensor sq_X,   float beta)
{
    int N = X_fp16.size(0), d = X_fp16.size(1);
    auto D    = torch::zeros({N}, w.options());
    int  grid = (N + BM - 1) / BM;
    // smem: sXi(BM,d) + sXj(BN,d) in fp16
    size_t smem = (size_t)(BM + BN) * d * sizeof(__half);
    prepass_kernel_sm120<<<grid, NTHREADS, smem,
                           at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(X_fp16.data_ptr<at::Half>()),
        w.data_ptr<float>(), D.data_ptr<float>(),
        sq_X.data_ptr<float>(), N, d, beta);
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
    size_t smem = (size_t)(BM + BN) * d * sizeof(__half);
    matvec_kernel_sm120<<<grid, NTHREADS, smem,
                          at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(X_fp16.data_ptr<at::Half>()),
        u.data_ptr<float>(), out.data_ptr<float>(),
        sq_X.data_ptr<float>(), rscale.data_ptr<float>(), N, d, beta);
    return out;
}

PYBIND11_MODULE(flash_diffusion_sm120, m) {
    m.doc() = "FlashDiffusion SM120 — cp.async 128x128 tile (RTX 5090 / PRO 6000)";
    m.def("compute_sq_norms", &compute_sq_norms, py::arg("X_fp16"));
    m.def("prepass", &prepass,
          py::arg("X_fp16"), py::arg("w"), py::arg("sq_X"), py::arg("beta"));
    m.def("matvec",  &matvec,
          py::arg("X_fp16"), py::arg("v"), py::arg("rscale"),
          py::arg("sq_X"), py::arg("beta"));
}
