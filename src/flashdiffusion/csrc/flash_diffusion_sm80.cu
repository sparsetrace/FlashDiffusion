/*
 * flash_diffusion_sm80.cu
 * =======================
 * SM80 (A100 / Ampere) CUDA kernel for FlashDiffusion.
 *
 * Computes, tile by tile, without materialising the O(N^2) kernel matrix:
 *
 *   PREPASS:
 *     D_i = sum_j  exp(-beta * ||x_i - x_j||^2) * w_j
 *
 *   MATVEC:
 *     out_i = rscale_i * sum_j  exp(-beta * ||x_i - x_j||^2) * rscale_j * v_j
 *
 * No online softmax — rscale is precomputed, epilogue is a scalar multiply.
 * No gradients — we compute the harmonic basis (eigenvectors), not a
 * differentiable layer.
 *
 * v1: scalar loop (validates correctness on A100).
 * v2: replace inner loop with CuTe TiledMMA (tensor cores) once v1 is green.
 *
 * Build (via torch.utils.cpp_extension.load in notebook):
 *   extra_cuda_cflags = [
 *       "-arch=sm_80", "-std=c++17", "--use_fast_math",
 *       "-U__CUDA_NO_HALF_OPERATORS__", "--expt-relaxed-constexpr",
 *   ]
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// ---------------------------------------------------------------------------
// Tile constants
// ---------------------------------------------------------------------------
static constexpr int BM       = 64;    // rows per Q-tile
static constexpr int BN       = 64;    // cols per K-tile
static constexpr int NTHREADS = 128;   // threads per block (4 warps)

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ float fast_exp(float x) { return __expf(x); }

// ---------------------------------------------------------------------------
// Sq-norm kernel:  sq[i] = ||x_i||^2
// ---------------------------------------------------------------------------
__global__ void sq_norm_kernel(
    const __half* __restrict__ X,
    float*        __restrict__ sq,
    int N, int d
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s = 0.f;
    for (int k = 0; k < d; ++k) {
        float v = __half2float(X[i * d + k]);
        s += v * v;
    }
    sq[i] = s;
}

// ---------------------------------------------------------------------------
// Prepass kernel:  D_i = sum_j K(i,j) * w_j
// ---------------------------------------------------------------------------
__global__ void prepass_kernel(
    const __half* __restrict__ X,
    const float*  __restrict__ w,
    float*        __restrict__ D,
    const float*  __restrict__ sq_X,
    int N, int d, float beta
) {
    extern __shared__ char smem[];
    __half* sXi = (__half*)smem;
    __half* sXj = sXi + BM * d;

    int tid = threadIdx.x;
    int i0  = blockIdx.x * BM;
    int i1  = min(i0 + BM, N);
    int bm  = i1 - i0;

    float acc = 0.f;

    // load Xi tile into smem
    for (int idx = tid; idx < bm * d; idx += NTHREADS)
        sXi[(idx / d) * d + idx % d] = X[(i0 + idx / d) * d + idx % d];
    __syncthreads();

    // loop over Xj tiles
    for (int j0 = 0; j0 < N; j0 += BN) {
        int j1 = min(j0 + BN, N);
        int bn = j1 - j0;

        for (int idx = tid; idx < bn * d; idx += NTHREADS)
            sXj[(idx / d) * d + idx % d] = X[(j0 + idx / d) * d + idx % d];
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

    if (tid < bm)
        D[i0 + tid] = acc;
}

// ---------------------------------------------------------------------------
// Matvec kernel:  out_i = rscale_i * sum_j K(i,j) * rscale_j * v_j
// ---------------------------------------------------------------------------
__global__ void matvec_kernel(
    const __half* __restrict__ X,
    const float*  __restrict__ u,        // u = rscale * v, pre-scaled on host
    float*        __restrict__ out,
    const float*  __restrict__ sq_X,
    const float*  __restrict__ rscale,
    int N, int d, float beta
) {
    extern __shared__ char smem[];
    __half* sXi = (__half*)smem;
    __half* sXj = sXi + BM * d;

    int tid = threadIdx.x;
    int i0  = blockIdx.x * BM;
    int i1  = min(i0 + BM, N);
    int bm  = i1 - i0;

    float acc = 0.f;

    for (int idx = tid; idx < bm * d; idx += NTHREADS)
        sXi[(idx / d) * d + idx % d] = X[(i0 + idx / d) * d + idx % d];
    __syncthreads();

    for (int j0 = 0; j0 < N; j0 += BN) {
        int j1 = min(j0 + BN, N);
        int bn = j1 - j0;

        for (int idx = tid; idx < bn * d; idx += NTHREADS)
            sXj[(idx / d) * d + idx % d] = X[(j0 + idx / d) * d + idx % d];
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

    if (tid < bm)
        out[i0 + tid] = rscale[i0 + tid] * acc;
}

// ---------------------------------------------------------------------------
// Python-facing entry points
// ---------------------------------------------------------------------------

// Call once per dataset — cache the result on the Python side
torch::Tensor compute_sq_norms(torch::Tensor X_fp16) {
    int N = X_fp16.size(0), d = X_fp16.size(1);
    auto sq = torch::empty({N}, X_fp16.options().dtype(torch::kFloat32));
    int threads = 256, blocks = (N + threads - 1) / threads;
    sq_norm_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(X_fp16.data_ptr<at::Half>()),
        sq.data_ptr<float>(), N, d);
    return sq;
}

// Pass 1: w = ones(N)  ->  D  (raw row sums)
// Pass 2: w = D^{-alpha}  ->  D_alpha (before * w_i)
torch::Tensor prepass(
    torch::Tensor X_fp16,
    torch::Tensor w,
    torch::Tensor sq_X,
    float beta
) {
    int N = X_fp16.size(0), d = X_fp16.size(1);
    auto D      = torch::zeros({N}, w.options());
    int  grid   = (N + BM - 1) / BM;
    size_t smem = (size_t)(BM + BN) * d * sizeof(__half);
    prepass_kernel<<<grid, NTHREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(X_fp16.data_ptr<at::Half>()),
        w.data_ptr<float>(), D.data_ptr<float>(),
        sq_X.data_ptr<float>(), N, d, beta);
    return D;
}

// Called every Lanczos iteration — X/rscale/sq_X fixed, only v changes
torch::Tensor matvec(
    torch::Tensor X_fp16,
    torch::Tensor v,
    torch::Tensor rscale,
    torch::Tensor sq_X,
    float beta
) {
    int N = X_fp16.size(0), d = X_fp16.size(1);
    auto u      = rscale * v;                    // pre-scale on GPU, O(N)
    auto out    = torch::empty({N}, v.options());
    int  grid   = (N + BM - 1) / BM;
    size_t smem = (size_t)(BM + BN) * d * sizeof(__half);
    matvec_kernel<<<grid, NTHREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(X_fp16.data_ptr<at::Half>()),
        u.data_ptr<float>(), out.data_ptr<float>(),
        sq_X.data_ptr<float>(), rscale.data_ptr<float>(), N, d, beta);
    return out;
}

// ---------------------------------------------------------------------------
// pybind11
// ---------------------------------------------------------------------------
PYBIND11_MODULE(flash_diffusion_cuda, m) {
    m.doc() = "FlashDiffusion SM80 CUDA kernel — scalar v1";
    m.def("compute_sq_norms", &compute_sq_norms,
          "Precompute ||x_i||^2. Call once, cache result.",
          py::arg("X_fp16"));
    m.def("prepass", &prepass,
          "Weighted row sum D_i = sum_j K(i,j)*w_j. Call twice for C-L normalisation.",
          py::arg("X_fp16"), py::arg("w"), py::arg("sq_X"), py::arg("beta"));
    m.def("matvec", &matvec,
          "Matvec out_i = rscale_i * sum_j K(i,j) * rscale_j * v_j. No gradients.",
          py::arg("X_fp16"), py::arg("v"), py::arg("rscale"),
          py::arg("sq_X"), py::arg("beta"));
}
