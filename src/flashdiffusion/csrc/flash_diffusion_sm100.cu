/*
 * flash_diffusion_sm100.cu
 * ========================
 * First-pass SM100 / B200-class FlashDiffusion kernel.
 *
 * Goals:
 *   - Same public API as flash_diffusion_sm120.cu:
 *       compute_sq_norms(X_fp16)
 *       prepass(X_fp16, w, sq_X, beta)
 *       matvec(X_fp16, v, rscale, sq_X, beta)
 *
 *   - Law-of-cosines distance:
 *       ||xi - xj||^2 = ||xi||^2 + ||xj||^2 - 2 xi·xj
 *
 *   - Force tensor-core path via WMMA:
 *       dot tile = Xi_tile @ Xj_tile.T
 *
 *   - Regular expf, no __expf, no --use_fast_math required.
 *
 * Notes:
 *   - This is a correctness/first-performance pass.
 *   - It uses WMMA, not Blackwell tcgen05/UMMA/TMEM/TMA.
 *   - Feature dimension d is padded to KPAD = ceil(d/16)*16 in shared memory.
 *   - For d=3 Swiss roll this will still use tensor cores, but padding overhead
 *     means it may not beat the scalar kernel. For larger MD feature dimensions,
 *     tensor cores should matter much more.
 *
 * Build suggestion:
 *   -std=c++17 -O3 -arch=sm_100a
 *   -U__CUDA_NO_HALF_OPERATORS__
 *   -U__CUDA_NO_HALF_CONVERSIONS__
 *   --expt-relaxed-constexpr
 *
 * Do NOT add --use_fast_math if you want regular expf semantics.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <sstream>

using namespace nvcuda;

static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

/*
 * Tile choice:
 *   BM=64, BN=64 gives 4 x 4 = 16 WMMA warp tiles.
 *   Therefore NTHREADS = 16 warps = 512 threads.
 *
 * Shared memory:
 *   sA    : BM x KPAD half, row-major
 *   sB    : BN x KPAD half, stored as KPAD x BN column-major
 *   sDot  : BM x BN float
 *
 * For KPAD=16:
 *   sA+sB = (64+64)*16*2 = 4096 bytes
 *   sDot  = 64*64*4      = 16384 bytes
 *   total ≈ 20 KB
 *
 * For KPAD=512:
 *   sA+sB = 128*512*2    = 131 KB
 *   sDot  = 16 KB
 *   total ≈ 147 KB
 */
static constexpr int BM       = 64;
static constexpr int BN       = 64;
static constexpr int WARPS_M  = BM / WMMA_M;  // 4
static constexpr int WARPS_N  = BN / WMMA_N;  // 4
static constexpr int NWARPS   = WARPS_M * WARPS_N;  // 16
static constexpr int NTHREADS = NWARPS * 32;  // 512

static inline int ceil_div_host(int a, int b) {
    return (a + b - 1) / b;
}

static inline int round_up_host(int a, int b) {
    return ceil_div_host(a, b) * b;
}

__device__ __forceinline__ float regular_exp(float x) {
    return expf(x);
}

// -----------------------------------------------------------------------------
// Basic checks
// -----------------------------------------------------------------------------

static void check_cuda_half_2d(const torch::Tensor& X, const char* name) {
    TORCH_CHECK(X.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(X.scalar_type() == at::kHalf, name, " must be float16/half");
    TORCH_CHECK(X.dim() == 2, name, " must be 2D");
    TORCH_CHECK(X.is_contiguous(), name, " must be contiguous");
}

static void check_cuda_float_1d(const torch::Tensor& x, const char* name) {
    TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(x.dim() == 1, name, " must be 1D");
    TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
}

// -----------------------------------------------------------------------------
// Squared norms
// -----------------------------------------------------------------------------

__global__ void sq_norm_kernel_sm100(
    const half* __restrict__ X,
    float*      __restrict__ sq,
    int N,
    int d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float s = 0.0f;
    const half* xi = X + static_cast<long long>(i) * d;

    for (int k = 0; k < d; ++k) {
        float v = __half2float(xi[k]);
        s += v * v;
    }

    sq[i] = s;
}

// -----------------------------------------------------------------------------
// Shared-memory loading helpers
// -----------------------------------------------------------------------------

__device__ __forceinline__ void load_A_tile_rowmajor(
    half*       __restrict__ sA,
    const half* __restrict__ X,
    int i0,
    int bm,
    int N,
    int d,
    int KPAD,
    int tid)
{
    // sA layout: [BM, KPAD] row-major
    int total = BM * KPAD;

    for (int idx = tid; idx < total; idx += NTHREADS) {
        int r = idx / KPAD;
        int k = idx - r * KPAD;

        half val = __float2half(0.0f);
        if (r < bm && k < d) {
            val = X[static_cast<long long>(i0 + r) * d + k];
        }
        sA[r * KPAD + k] = val;
    }
}

__device__ __forceinline__ void load_B_tile_colmajor(
    half*       __restrict__ sB,
    const half* __restrict__ X,
    int j0,
    int bn,
    int N,
    int d,
    int KPAD,
    int tid)
{
    /*
     * We want B = Xj.T, shape [KPAD, BN].
     * WMMA matrix_b is loaded as column-major.
     *
     * Store as:
     *   sB[col * KPAD + k] = X[j0 + col, k]
     *
     * Then pointer for a 16-column block is:
     *   sB + col16 * KPAD + kk
     *
     * with leading dimension KPAD.
     */
    int total = BN * KPAD;

    for (int idx = tid; idx < total; idx += NTHREADS) {
        int col = idx / KPAD;
        int k   = idx - col * KPAD;

        half val = __float2half(0.0f);
        if (col < bn && k < d) {
            val = X[static_cast<long long>(j0 + col) * d + k];
        }
        sB[col * KPAD + k] = val;
    }
}

// -----------------------------------------------------------------------------
// Tensor-core dot tile
// -----------------------------------------------------------------------------

__device__ __forceinline__ void compute_dot_tile_wmma(
    const half* __restrict__ sA,
    const half* __restrict__ sB,
    float*      __restrict__ sDot,
    int KPAD,
    int tid)
{
    int warp_id = tid >> 5;  // 0..15

    int wm = warp_id / WARPS_N;  // 0..3
    int wn = warp_id - wm * WARPS_N;  // 0..3

    int row16 = wm * WMMA_M;
    int col16 = wn * WMMA_N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   half, wmma::row_major> a_frag;

    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   half, wmma::col_major> b_frag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                   float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for (int kk = 0; kk < KPAD; kk += WMMA_K) {
        const half* a_ptr = sA + row16 * KPAD + kk;
        const half* b_ptr = sB + col16 * KPAD + kk;

        wmma::load_matrix_sync(a_frag, a_ptr, KPAD);
        wmma::load_matrix_sync(b_frag, b_ptr, KPAD);

        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // Store 16x16 subtile into sDot row-major, full tile stride BN.
    float* dot_ptr = sDot + row16 * BN + col16;
    wmma::store_matrix_sync(dot_ptr, acc_frag, BN, wmma::mem_row_major);
}

// -----------------------------------------------------------------------------
// Prepass:
//   D_i = Σ_j exp(-beta ||xi-xj||²) * w_j
// -----------------------------------------------------------------------------

__global__ void prepass_kernel_sm100_tc(
    const half*  __restrict__ X,
    const float* __restrict__ w,
    float*       __restrict__ D,
    const float* __restrict__ sq_X,
    int N,
    int d,
    int KPAD,
    float beta)
{
    extern __shared__ unsigned char smem_raw[];

    half*  sA    = reinterpret_cast<half*>(smem_raw);
    half*  sB    = sA + BM * KPAD;
    float* sDot  = reinterpret_cast<float*>(sB + BN * KPAD);

    int tid = threadIdx.x;

    int i0 = blockIdx.x * BM;
    int i1 = min(i0 + BM, N);
    int bm = i1 - i0;

    // One persistent accumulator per output row for threads tid < bm.
    float acc = 0.0f;

    load_A_tile_rowmajor(sA, X, i0, bm, N, d, KPAD, tid);
    __syncthreads();

    for (int j0 = 0; j0 < N; j0 += BN) {
        int j1 = min(j0 + BN, N);
        int bn = j1 - j0;

        load_B_tile_colmajor(sB, X, j0, bn, N, d, KPAD, tid);
        __syncthreads();

        compute_dot_tile_wmma(sA, sB, sDot, KPAD, tid);
        __syncthreads();

        if (tid < bm) {
            float sqi = sq_X[i0 + tid];

            #pragma unroll 4
            for (int j = 0; j < BN; ++j) {
                if (j < bn) {
                    float dot   = sDot[tid * BN + j];
                    float dist2 = sqi + sq_X[j0 + j] - 2.0f * dot;
                    dist2 = fmaxf(dist2, 0.0f);

                    float kij = regular_exp(-beta * dist2);
                    acc += kij * w[j0 + j];
                }
            }
        }

        // Protect sB/sDot before next j tile overwrites shared memory.
        __syncthreads();
    }

    if (tid < bm) {
        D[i0 + tid] = acc;
    }
}

// -----------------------------------------------------------------------------
// Matvec:
//   out_i = rscale_i * Σ_j exp(-beta ||xi-xj||²) * u_j
//   where u_j = rscale_j * v_j is precomputed on the Torch side.
// -----------------------------------------------------------------------------

__global__ void matvec_kernel_sm100_tc(
    const half*  __restrict__ X,
    const float* __restrict__ u,
    float*       __restrict__ out,
    const float* __restrict__ sq_X,
    const float* __restrict__ rscale,
    int N,
    int d,
    int KPAD,
    float beta)
{
    extern __shared__ unsigned char smem_raw[];

    half*  sA    = reinterpret_cast<half*>(smem_raw);
    half*  sB    = sA + BM * KPAD;
    float* sDot  = reinterpret_cast<float*>(sB + BN * KPAD);

    int tid = threadIdx.x;

    int i0 = blockIdx.x * BM;
    int i1 = min(i0 + BM, N);
    int bm = i1 - i0;

    float acc = 0.0f;

    load_A_tile_rowmajor(sA, X, i0, bm, N, d, KPAD, tid);
    __syncthreads();

    for (int j0 = 0; j0 < N; j0 += BN) {
        int j1 = min(j0 + BN, N);
        int bn = j1 - j0;

        load_B_tile_colmajor(sB, X, j0, bn, N, d, KPAD, tid);
        __syncthreads();

        compute_dot_tile_wmma(sA, sB, sDot, KPAD, tid);
        __syncthreads();

        if (tid < bm) {
            float sqi = sq_X[i0 + tid];

            #pragma unroll 4
            for (int j = 0; j < BN; ++j) {
                if (j < bn) {
                    float dot   = sDot[tid * BN + j];
                    float dist2 = sqi + sq_X[j0 + j] - 2.0f * dot;
                    dist2 = fmaxf(dist2, 0.0f);

                    float kij = regular_exp(-beta * dist2);
                    acc += kij * u[j0 + j];
                }
            }
        }

        __syncthreads();
    }

    if (tid < bm) {
        out[i0 + tid] = rscale[i0 + tid] * acc;
    }
}

// -----------------------------------------------------------------------------
// Shared-memory launch helper
// -----------------------------------------------------------------------------

static size_t smem_bytes_for(int d) {
    int KPAD = round_up_host(d, WMMA_K);

    size_t bytes_A   = static_cast<size_t>(BM) * KPAD * sizeof(half);
    size_t bytes_B   = static_cast<size_t>(BN) * KPAD * sizeof(half);
    size_t bytes_dot = static_cast<size_t>(BM) * BN   * sizeof(float);

    return bytes_A + bytes_B + bytes_dot;
}

static void check_and_configure_smem_prepass(size_t smem) {
    int device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&device));

    int max_optin = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &max_optin,
        cudaDevAttrMaxSharedMemoryPerBlockOptin,
        device));

    TORCH_CHECK(
        smem <= static_cast<size_t>(max_optin),
        "Requested dynamic shared memory ", smem,
        " exceeds device opt-in maximum ", max_optin,
        ". Reduce feature dimension d, lower BM/BN, or add a lower-memory kernel."
    );

    C10_CUDA_CHECK(cudaFuncSetAttribute(
        prepass_kernel_sm100_tc,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem)));

    C10_CUDA_CHECK(cudaFuncSetAttribute(
        prepass_kernel_sm100_tc,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared));
}

static void check_and_configure_smem_matvec(size_t smem) {
    int device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&device));

    int max_optin = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &max_optin,
        cudaDevAttrMaxSharedMemoryPerBlockOptin,
        device));

    TORCH_CHECK(
        smem <= static_cast<size_t>(max_optin),
        "Requested dynamic shared memory ", smem,
        " exceeds device opt-in maximum ", max_optin,
        ". Reduce feature dimension d, lower BM/BN, or add a lower-memory kernel."
    );

    C10_CUDA_CHECK(cudaFuncSetAttribute(
        matvec_kernel_sm100_tc,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem)));

    C10_CUDA_CHECK(cudaFuncSetAttribute(
        matvec_kernel_sm100_tc,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared));
}

// -----------------------------------------------------------------------------
// Torch bindings
// -----------------------------------------------------------------------------

torch::Tensor compute_sq_norms(torch::Tensor X_fp16) {
    auto Xc = X_fp16.contiguous();

    check_cuda_half_2d(Xc, "X_fp16");

    int N = static_cast<int>(Xc.size(0));
    int d = static_cast<int>(Xc.size(1));

    auto sq = torch::empty({N}, Xc.options().dtype(at::kFloat));

    int threads = 256;
    int blocks  = ceil_div_host(N, threads);

    sq_norm_kernel_sm100<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(Xc.data_ptr<at::Half>()),
        sq.data_ptr<float>(),
        N,
        d);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return sq;
}

torch::Tensor prepass(
    torch::Tensor X_fp16,
    torch::Tensor w,
    torch::Tensor sq_X,
    float beta)
{
    auto Xc  = X_fp16.contiguous();
    auto wc  = w.contiguous();
    auto sqc = sq_X.contiguous();

    check_cuda_half_2d(Xc, "X_fp16");
    check_cuda_float_1d(wc, "w");
    check_cuda_float_1d(sqc, "sq_X");

    int N = static_cast<int>(Xc.size(0));
    int d = static_cast<int>(Xc.size(1));

    TORCH_CHECK(wc.size(0) == N, "w length must equal N");
    TORCH_CHECK(sqc.size(0) == N, "sq_X length must equal N");

    int KPAD = round_up_host(d, WMMA_K);
    int grid = ceil_div_host(N, BM);

    size_t smem = smem_bytes_for(d);
    check_and_configure_smem_prepass(smem);

    auto D = torch::empty({N}, wc.options().dtype(at::kFloat));

    prepass_kernel_sm100_tc<<<grid, NTHREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(Xc.data_ptr<at::Half>()),
        wc.data_ptr<float>(),
        D.data_ptr<float>(),
        sqc.data_ptr<float>(),
        N,
        d,
        KPAD,
        beta);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return D;
}

torch::Tensor matvec(
    torch::Tensor X_fp16,
    torch::Tensor v,
    torch::Tensor rscale,
    torch::Tensor sq_X,
    float beta)
{
    auto Xc = X_fp16.contiguous();
    auto vc = v.contiguous();
    auto rc = rscale.contiguous();
    auto sqc = sq_X.contiguous();

    check_cuda_half_2d(Xc, "X_fp16");
    check_cuda_float_1d(vc, "v");
    check_cuda_float_1d(rc, "rscale");
    check_cuda_float_1d(sqc, "sq_X");

    int N = static_cast<int>(Xc.size(0));
    int d = static_cast<int>(Xc.size(1));

    TORCH_CHECK(vc.size(0) == N, "v length must equal N");
    TORCH_CHECK(rc.size(0) == N, "rscale length must equal N");
    TORCH_CHECK(sqc.size(0) == N, "sq_X length must equal N");

    int KPAD = round_up_host(d, WMMA_K);
    int grid = ceil_div_host(N, BM);

    size_t smem = smem_bytes_for(d);
    check_and_configure_smem_matvec(smem);

    // u_j = rscale_j * v_j
    auto u = (rc * vc).contiguous();

    auto out = torch::empty({N}, vc.options().dtype(at::kFloat));

    matvec_kernel_sm100_tc<<<grid, NTHREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(Xc.data_ptr<at::Half>()),
        u.data_ptr<float>(),
        out.data_ptr<float>(),
        sqc.data_ptr<float>(),
        rc.data_ptr<float>(),
        N,
        d,
        KPAD,
        beta);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(flash_diffusion_sm100, m) {
    m.doc() = "FlashDiffusion SM100/B200 first-pass WMMA tensor-core kernel";

    m.def("compute_sq_norms", &compute_sq_norms, py::arg("X_fp16"));

    m.def("prepass", &prepass,
          py::arg("X_fp16"),
          py::arg("w"),
          py::arg("sq_X"),
          py::arg("beta"));

    m.def("matvec", &matvec,
          py::arg("X_fp16"),
          py::arg("v"),
          py::arg("rscale"),
          py::arg("sq_X"),
          py::arg("beta"));
}
