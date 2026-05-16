/*
 * flash_diffusion_sm120.cu
 * ========================
 * SM120 (RTX 5090 / RTX PRO 6000 Blackwell) kernel for FlashDiffusion.
 *
 * Validated working. No cp.async pipeline (v1 — synchronous loads).
 * cp.async pipeline to be added in v2 once correctness is confirmed.
 *
 * SM120 vs SM80 differences used here:
 *   - Larger tiles: BM=128, BN=128 (vs BM=64, BN=64 on SM80)
 *   - More threads: NTHREADS=256 (8 warps vs 4 warps)
 *   - Same mma.sync instruction family (no wgmma, no TMEM)
 *   - No multicast, cluster shape 1x1x1
 *
 * No online softmax. No gradients. rscale precomputed.
 *
 * Build:
 *   -arch=sm_120 -std=c++17 --use_fast_math
 *   -U__CUDA_NO_HALF_OPERATORS__ --expt-relaxed-constexpr
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

static constexpr int BM       = 128;
static constexpr int BN       = 128;
static constexpr int NTHREADS = 256;

__device__ __forceinline__ float fast_exp(float x) { return __expf(x); }

__global__ void sq_norm_kernel_sm120(
    const __half* __restrict__ X,
    float*        __restrict__ sq,
    int N, int d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s = 0.f;
    for (int k = 0; k < d; ++k) {
        float v = __half2float(X[i * d + k]);
        s += v * v;
    }
    sq[i] = s;
}

__global__ void prepass_kernel_sm120(
    const __half* __restrict__ X,
    const float*  __restrict__ w,
    float*        __restrict__ D,
    const float*  __restrict__ sq_X,
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

    for (int idx = tid; idx < bm * d; idx += NTHREADS) {
        int r = idx / d, c = idx % d;
        sXi[r * d + c] = X[(i0 + r) * d + c];
    }
    __syncthreads();

    for (int j0 = 0; j0 < N; j0 += BN) {
        int j1 = min(j0 + BN, N);
        int bn = j1 - j0;

        for (int idx = tid; idx < bn * d; idx += NTHREADS) {
            int r = idx / d, c = idx % d;
            sXj[r * d + c] = X[(j0 + r) * d + c];
        }
        __syncthreads();

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

__global__ void matvec_kernel_sm120(
    const __half* __restrict__ X,
    const float*  __restrict__ u,
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

    for (int idx = tid; idx < bm * d; idx += NTHREADS) {
        int r = idx / d, c = idx % d;
        sXi[r * d + c] = X[(i0 + r) * d + c];
    }
    __syncthreads();

    for (int j0 = 0; j0 < N; j0 += BN) {
        int j1 = min(j0 + BN, N);
        int bn = j1 - j0;

        for (int idx = tid; idx < bn * d; idx += NTHREADS) {
            int r = idx / d, c = idx % d;
            sXj[r * d + c] = X[(j0 + r) * d + c];
        }
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

torch::Tensor compute_sq_norms(torch::Tensor X_fp16) {
    int N = X_fp16.size(0), d = X_fp16.size(1);
    auto sq = torch::empty({N}, X_fp16.options().dtype(torch::kFloat32));
    sq_norm_kernel_sm120<<<(N + 255) / 256, 256, 0,
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
    m.doc() = "FlashDiffusion SM120 — 128x128 tiles v1 validated "
              "(RTX 5090 / RTX PRO 6000 Blackwell)";
    m.def("compute_sq_norms", &compute_sq_norms, py::arg("X_fp16"));
    m.def("prepass", &prepass,
          py::arg("X_fp16"), py::arg("w"), py::arg("sq_X"), py::arg("beta"));
    m.def("matvec",  &matvec,
          py::arg("X_fp16"), py::arg("v"), py::arg("rscale"),
          py::arg("sq_X"), py::arg("beta"));
}
