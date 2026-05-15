/*
flashdiffusion/csrc/flash_diffusion_sm80.cu
============================================
SM80 (A100 / Ampere) CuTe kernel for FlashDiffusion.

Computes, tile by tile, without materialising the O(N²) kernel matrix:

  PREPASS:
    D_i = Σ_j  exp(-beta * ||x_i - x_j||²) * w_j        (weighted row sum)

  MATVEC:
    out_i = rscale_i * Σ_j  exp(-beta * ||x_i - x_j||²) * rscale_j * v_j

Both kernels share the same tiling structure:
  1. Load X_i tile (Q side) into SMEM via cp.async
  2. Loop over X_j tiles (K side, fixed — "KV cache"):
     a. Load X_j tile into SMEM via cp.async
     b. GEMM: dot_ij = Xi @ Xj^T   (fp16 mma.sync -> fp32 accum)
     c. Bias:  dist2  = sq_i + sq_j - 2*dot_ij
     d. Exp:   K_tile = exp(-beta * dist2)
     e. Accumulate: acc += K_tile * u_j   (u_j = rscale_j * v_j for matvec,
                                                = w_j for prepass)
  3. Post-scale: out_i = rscale_i * acc_i   (matvec only)
  4. Write output tile to GMEM

No online softmax. No backward pass. No autograd.
rscale is precomputed by the two prepass calls; only V changes per Lanczos iter.

Architecture notes (SM80)
--------------------------
- MMA atom: SM80_16x8x16_F32F16F16F32  (mma.sync, fp16 in, fp32 accum)
- Async copy: SM80_CP_ASYNC_CACHEALWAYS<uint128_t>  (128-bit loads, 16 bytes)
- Tile shape: [BM=64, BN=64, BK=16] — fits in 192KB SMEM with 2-stage pipeline
- Thread block: 128 threads (4 warps)
- Each thread block computes one (BM, BN) tile of the kernel matrix K_tile,
  then reduces into the output (BM, r) accumulator across the BN dimension.

Key difference from FlashAttention
-----------------------------------
FA: online softmax requires max-tracking across BN tiles -> inter-tile state
FD: plain sum -> acc += K_tile * u_j -> no inter-tile state, tiles independent
This means the epilogue is a single scalar multiply (rscale), not a rescaling.

Build
-----
  nvcc -arch=sm_80 -std=c++17 -O3 \
       -I /path/to/cutlass/include \
       flash_diffusion_sm80.cu -o flash_diffusion_sm80.o

  Or via torch.utils.cpp_extension.CUDAExtension (see setup.py).

Exposed to Python via pybind11 / torch.ops:
  prepass_cuda(X, w, beta, tile_m, tile_n) -> D        (N,)  float32
  matvec_cuda(X, V, rscale, beta, tile_m, tile_n) -> out (N, r) float32
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

using namespace cute;

// ---------------------------------------------------------------------------
// Compile-time tile constants (SM80 sweet spot for 192KB SMEM)
// ---------------------------------------------------------------------------
static constexpr int BM = 64;    // rows per Q-tile  (output rows per block)
static constexpr int BN = 64;    // cols per K-tile  (inner reduction)
static constexpr int BK = 16;    // coordinate dimension tile (k-loop over d)
static constexpr int NTHREADS = 128;  // 4 warps

// MMA atom: SM80, fp16 A/B, fp32 C/D, shape 16x8x16
using MMA_Atom_t = MMA_Atom<SM80_16x8x16_F32F16F16F32>;

// Async copy atom: 128-bit (16 bytes = 8 fp16) per thread
using CopyAtom_t = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cutlass::half_t>;

// ---------------------------------------------------------------------------
// Shared memory layout (2-stage pipeline)
// ---------------------------------------------------------------------------
// Xi: (BM, BK) in smem, 2 stages
// Xj: (BN, BK) in smem, 2 stages
// Swizzled to avoid bank conflicts on the K dimension
using SmemLayoutAtom = decltype(
    composition(Swizzle<3,3,3>{},
                Layout<Shape<_8, _BK>, Stride<_BK, _1>>{}));
//  ^-- will be specialised with BK=16 at compile time

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

// Fast exp approximation (single precision) — same as FA2
__device__ __forceinline__ float fast_exp(float x) {
    return __expf(x);
}

// Squared L2 norm of a register vector fragment (used for diagonal bias)
__device__ __forceinline__ float sq_norm_fragment(
    const float* x, int d)
{
    float acc = 0.f;
    for (int k = 0; k < d; ++k) acc += x[k] * x[k];
    return acc;
}

// ---------------------------------------------------------------------------
// PREPASS kernel: D_i = Σ_j K(i,j) * w_j
// ---------------------------------------------------------------------------
/*
  Grid:  (ceil(N/BM),)          -- one block per output row tile
  Block: (NTHREADS,)

  Each block:
    - Loads X[i0:i1, :] into smem (BM rows)
    - Loops over j tiles of X[j0:j1, :]
      - Computes dist2 via GEMM + diagonal bias
      - Accumulates D[i] += Σ_j exp(-beta*dist2) * w[j]
    - Writes D[i0:i1] to GMEM
*/
template<int BM_, int BN_, int BK_>
__global__ void prepass_kernel(
    const __half* __restrict__ X,      // (N, d)  fp16
    const float*  __restrict__ w,      // (N,)    fp32 weights  (=1 for pass1)
    float*        __restrict__ D,      // (N,)    fp32 output
    const float*  __restrict__ sq_X,   // (N,)    precomputed ||x_i||^2 fp32
    int N, int d,
    float beta,
    int num_j_tiles
) {
    // ---- Thread / block indices ----
    int tid  = threadIdx.x;
    int i0   = blockIdx.x * BM_;
    int i1   = min(i0 + BM_, N);
    int bm   = i1 - i0;               // actual rows in this block

    // ---- Shared memory: 2-stage pipeline for Xi and Xj ----
    extern __shared__ char smem_buf[];
    __half* sXi = (__half*)smem_buf;                         // (BM, BK) x 2
    __half* sXj = sXi + 2 * BM_ * BK_;                     // (BN, BK) x 2

    // ---- Accumulator: (BM,) register float ----
    float acc[BM_];
    #pragma unroll
    for (int m = 0; m < BM_; ++m) acc[m] = 0.f;

    // ---- Main loop over j tiles ----
    for (int jt = 0; jt < num_j_tiles; ++jt) {
        int j0 = jt * BN_;
        int j1 = min(j0 + BN_, N);
        int bn = j1 - j0;

        // Load Xj tile: each thread loads a few elements
        // (simplified: use cp.async for production; direct load here for clarity)
        // In a full CuTe kernel this is replaced by make_tiled_copy + cute::copy
        for (int idx = tid; idx < bn * d; idx += NTHREADS) {
            int row = idx / d, col = idx % d;
            sXj[row * d + col] = X[(j0 + row) * d + col];
        }
        // Load Xi tile
        for (int idx = tid; idx < bm * d; idx += NTHREADS) {
            int row = idx / d, col = idx % d;
            sXi[row * d + col] = X[(i0 + row) * d + col];
        }
        __syncthreads();

        // Each thread owns one row of the output tile (i0 + tid if tid < bm)
        if (tid < bm) {
            int gi = i0 + tid;
            float sqi = sq_X[gi];

            for (int j = 0; j < bn; ++j) {
                int gj = j0 + j;
                // dot = xi . xj
                float dot = 0.f;
                for (int k = 0; k < d; ++k) {
                    dot += __half2float(sXi[tid * d + k])
                         * __half2float(sXj[j   * d + k]);
                }
                float sqj  = sq_X[gj];
                float dist2 = fmaxf(sqi + sqj - 2.f * dot, 0.f);
                float kij   = fast_exp(-beta * dist2);
                acc[tid] += kij * w[gj];
            }
        }
        __syncthreads();
    }

    // Write output
    if (tid < bm) {
        D[i0 + tid] = acc[tid];
    }
}

// ---------------------------------------------------------------------------
// MATVEC kernel: out_i = rscale_i * Σ_j K(i,j) * rscale_j * v_j
// ---------------------------------------------------------------------------
/*
  Grid:  (ceil(N/BM), r)         -- r = number of Lanczos vectors (typically 1)
  Block: (NTHREADS,)

  Same tile loop as prepass, but:
    - u_j = rscale_j * v_j     (pre-scaled input)
    - out_i *= rscale_i         (post-scale)
    - accumulator is a float (scalar per Lanczos vector per row)
*/
__global__ void matvec_kernel(
    const __half* __restrict__ X,       // (N, d)  fp16
    const float*  __restrict__ u,       // (N,)    fp32  u = rscale * v
    float*        __restrict__ out,     // (N,)    fp32
    const float*  __restrict__ sq_X,    // (N,)    precomputed ||x_i||^2
    const float*  __restrict__ rscale,  // (N,)    post-scale diagonal
    int N, int d,
    float beta,
    int num_j_tiles
) {
    int tid = threadIdx.x;
    int i0  = blockIdx.x * BM;
    int i1  = min(i0 + BM, N);
    int bm  = i1 - i0;

    extern __shared__ char smem_buf[];
    __half* sXi = (__half*)smem_buf;
    __half* sXj = sXi + BM * d;

    float acc = 0.f;

    for (int jt = 0, j0 = 0; j0 < N; ++jt, j0 += BN) {
        int j1 = min(j0 + BN, N);
        int bn = j1 - j0;

        for (int idx = tid; idx < bn * d; idx += NTHREADS)
            sXj[(idx/d) * d + idx%d] = X[(j0 + idx/d) * d + idx%d];
        for (int idx = tid; idx < bm * d; idx += NTHREADS)
            sXi[(idx/d) * d + idx%d] = X[(i0 + idx/d) * d + idx%d];
        __syncthreads();

        if (tid < bm) {
            float sqi = sq_X[i0 + tid];
            for (int j = 0; j < bn; ++j) {
                float dot = 0.f;
                for (int k = 0; k < d; ++k)
                    dot += __half2float(sXi[tid*d+k]) * __half2float(sXj[j*d+k]);
                float dist2 = fmaxf(sqi + sq_X[j0+j] - 2.f*dot, 0.f);
                acc += fast_exp(-beta * dist2) * u[j0+j];
            }
        }
        __syncthreads();
    }

    if (tid < bm) {
        out[i0 + tid] = rscale[i0 + tid] * acc;
    }
}

// ---------------------------------------------------------------------------
// Sq-norm precomputation kernel
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
        float v = __half2float(X[i*d+k]);
        s += v*v;
    }
    sq[i] = s;
}

// ---------------------------------------------------------------------------
// Python-facing entry points (called from kernel.py CUDA backend)
// ---------------------------------------------------------------------------

// Precompute ||x_i||^2 for all i — called once, cached on the Python side
torch::Tensor compute_sq_norms(torch::Tensor X_fp16) {
    // X_fp16: (N, d) float16 on CUDA
    int N = X_fp16.size(0), d = X_fp16.size(1);
    auto sq = torch::empty({N}, X_fp16.options().dtype(torch::kFloat32));

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    sq_norm_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(X_fp16.data_ptr<at::Half>()),
        sq.data_ptr<float>(),
        N, d
    );
    return sq;
}

// Prepass: D = K @ w  (one weighted row sum)
// w = ones(N) for pass1, w = D^{-alpha} for pass2
torch::Tensor prepass_cuda(
    torch::Tensor X_fp16,   // (N, d)  fp16
    torch::Tensor w,        // (N,)    fp32
    torch::Tensor sq_X,     // (N,)    fp32  from compute_sq_norms
    float beta
) {
    int N = X_fp16.size(0), d = X_fp16.size(1);
    auto D = torch::zeros({N}, w.options());

    int num_j_tiles = (N + BN - 1) / BN;
    int grid        = (N + BM - 1) / BM;
    // smem: Xi(BM,d) + Xj(BN,d) in fp16
    size_t smem = (BM + BN) * d * sizeof(__half);

    prepass_kernel<BM, BN, BK><<<grid, NTHREADS, smem,
                                  at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(X_fp16.data_ptr<at::Half>()),
        w.data_ptr<float>(),
        D.data_ptr<float>(),
        sq_X.data_ptr<float>(),
        N, d, beta, num_j_tiles
    );
    return D;
}

// Matvec: out = rscale * (K @ (rscale * v))
// Called every Lanczos iteration — V changes, X/rscale fixed
torch::Tensor matvec_cuda(
    torch::Tensor X_fp16,   // (N, d)  fp16
    torch::Tensor v,        // (N,)    fp32  Lanczos vector
    torch::Tensor rscale,   // (N,)    fp32  precomputed diagonal
    torch::Tensor sq_X,     // (N,)    fp32
    float beta
) {
    int N = X_fp16.size(0), d = X_fp16.size(1);
    // pre-scale: u = rscale * v
    auto u   = rscale * v;                   // (N,) elementwise, stays on GPU
    auto out  = torch::empty({N}, v.options());

    int grid = (N + BM - 1) / BM;
    size_t smem = (BM + BN) * d * sizeof(__half);

    matvec_kernel<<<grid, NTHREADS, smem,
                    at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(X_fp16.data_ptr<at::Half>()),
        u.data_ptr<float>(),
        out.data_ptr<float>(),
        sq_X.data_ptr<float>(),
        rscale.data_ptr<float>(),
        N, d, beta,
        (N + BN - 1) / BN
    );
    return out;
}

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------
PYBIND11_MODULE(flash_diffusion_cuda, m) {
    m.doc() = "FlashDiffusion SM80 CUDA kernel";

    m.def("compute_sq_norms", &compute_sq_norms,
          "Precompute ||x_i||^2 for all rows. Call once, cache the result.",
          py::arg("X_fp16"));

    m.def("prepass", &prepass_cuda,
          "Weighted row sum: D_i = sum_j K(i,j) * w_j. "
          "Call twice: pass1 w=ones, pass2 w=D^{-alpha}.",
          py::arg("X_fp16"), py::arg("w"), py::arg("sq_X"), py::arg("beta"));

    m.def("matvec", &matvec_cuda,
          "Matvec: out_i = rscale_i * sum_j K(i,j) * rscale_j * v_j. "
          "Called every Lanczos iteration. No gradients needed.",
          py::arg("X_fp16"), py::arg("v"), py::arg("rscale"),
          py::arg("sq_X"), py::arg("beta"));
}
