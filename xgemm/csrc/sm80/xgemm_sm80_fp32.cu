/*
 * xgemm SM80/Ampere FP32 demonstrator — matrix-free exponential GEMM.
 *
 * Supported GPUs: A100 (sm_80), A10/A10G (sm_86), same sm80 binary family.
 * All computations use FP32 CUDA cores. No WMMA, CUTLASS, torch or tensor cores.
 * Query blocks BM=64, key blocks BN=64; feature chunks KCHUNK=32.
 *
 * Dot:      out = exp(Qs @ Q.T) @ V
 * Gaussian: out = exp(min(Qs @ Q.T - beta*(qi[:,None]+qk[None,:]),0)) @ V
 *           where Qs = 2*beta*Xq, Q = Xk, qi=||Xq||^2, qk=||Xk||^2.
 *
 * All 2D buffers C/row-major. The Python wrapper pads Nq/Nk to multiples of
 * 64 and feature D to a multiple of 32. RHS width <=16 per call.
 *
 * beta is a device-side float32 rank-0 FFI operand, not a compile-time attr.
 * JAX XLA FFI owns the stream; this extension never synchronizes it.
 *
 * Compile with: python tools/build_sm80.py
 */

#include <Python.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

static constexpr int BM = 64;
static constexpr int BN = 64;
static constexpr int KCHUNK = 32;
static constexpr int BV = 16;
static constexpr int THREADS = 128;
static constexpr size_t SMEM_BYTES =
    (size_t)(BM + BN) * KCHUNK * sizeof(float) +
    (size_t)BM * BN * sizeof(float) +
    (size_t)BN * BV * sizeof(float);

// The 36 KiB scratch footprint is within the standard 48 KiB CUDA shared
// memory limit; no dynamic-shared-memory opt-in is necessary.
static_assert(SMEM_BYTES <= 48 * 1024, "SM80 kernel exceeds baseline SMEM budget");

__device__ __forceinline__ float safe_exp(float s) { return __expf(s); }

__device__ __forceinline__ void kahan_tile_sum(float x, float& sum, float& correction) {
    float y = __fsub_rn(x, correction);
    float t = __fadd_rn(sum, y);
    correction = __fsub_rn(__fsub_rn(t, sum), y);
    sum = t;
}

// Bounded dynamic shared scratch; no N-by-N buffers are allocated.
// Each of the first 64 threads owns one query row. All 128 threads cooperate
// in loading query/key/value tiles. Gram is stored as [BN][BM] to make
// per-key writes from consecutive query-row threads contiguous.
template <bool RBF>
__global__ void __launch_bounds__(THREADS)
exp_gemm_kernel(
    const float* __restrict__ Qs,   // [Nq, D], pre-scaled in RBF mode
    const float* __restrict__ Q,    // [Nk, D]
    const float* __restrict__ V,    // [Nk, cols], cols <= 16
    const float* __restrict__ qi,   // [Nq] if RBF, nullptr otherwise
    const float* __restrict__ qk,   // [Nk] if RBF, nullptr otherwise
    const float* __restrict__ beta, // device scalar if RBF, nullptr otherwise
    float* __restrict__ out,       // [Nq, cols]
    int Nq, int Nk, int D, int cols)
{
    extern __shared__ float scratch[];
    float* A = scratch;                       // [BM, KCHUNK]
    float* B = A + BM * KCHUNK;              // [BN, KCHUNK]
    float* G = B + BN * KCHUNK;              // [BN, BM] transposed gram
    float* RHS = G + BN * BM;                // [BN, BV]

    const int tid = threadIdx.x;
    const int i0 = blockIdx.x * BM;
    const bool active = tid < BM;
    const float beta_value = RBF ? beta[0] : 0.f;
    const float norm_i = (RBF && active) ? qi[i0 + tid] : 0.f;

    float acc[BV];
    float corr[BV];
    #pragma unroll
    for (int c = 0; c < BV; ++c) {
        acc[c] = 0.f;
        corr[c] = 0.f;
    }

    for (int j0 = 0; j0 < Nk; j0 += BN) {
        if (active) {
            #pragma unroll
            for (int j = 0; j < BN; ++j) G[j * BM + tid] = 0.f;
        }
        // RHS is independent of the feature dimension, so stage once.
        for (int idx = tid; idx < BN * BV; idx += THREADS) {
            const int j = idx / BV;
            const int c = idx % BV;
            RHS[idx] = (c < cols) ? V[(size_t)(j0 + j) * cols + c] : 0.f;
        }
        __syncthreads();

        for (int k0 = 0; k0 < D; k0 += KCHUNK) {
            for (int idx = tid; idx < BM * KCHUNK; idx += THREADS) {
                const int i = idx / KCHUNK;
                const int k = idx % KCHUNK;
                A[idx] = Qs[(size_t)(i0 + i) * D + k0 + k];
            }
            for (int idx = tid; idx < BN * KCHUNK; idx += THREADS) {
                const int j = idx / KCHUNK;
                const int k = idx % KCHUNK;
                B[idx] = Q[(size_t)(j0 + j) * D + k0 + k];
            }
            __syncthreads();

            if (active) {
                #pragma unroll 4
                for (int j = 0; j < BN; ++j) {
                    float dot = 0.f;
                    #pragma unroll
                    for (int k = 0; k < KCHUNK; ++k) {
                        dot = fmaf(A[tid * KCHUNK + k], B[j * KCHUNK + k], dot);
                    }
                    G[j * BM + tid] += dot;
                }
            }
            // Needed before A/B are overwritten by the next feature chunk.
            __syncthreads();
        }

        if (active) {
            float tile_acc[BV];
            #pragma unroll
            for (int c = 0; c < BV; ++c) tile_acc[c] = 0.f;

            #pragma unroll 4
            for (int j = 0; j < BN; ++j) {
                float s = G[j * BM + tid];
                if (RBF) {
                    s -= beta_value * (norm_i + qk[j0 + j]);
                    s = fminf(s, 0.f);  // squared distances >=0 in exact arithmetic
                }
                const float p = safe_exp(s);
                #pragma unroll
                for (int c = 0; c < BV; ++c) {
                    if (c < cols) {
                        tile_acc[c] = fmaf(p, RHS[j * BV + c], tile_acc[c]);
                    }
                }
            }
            #pragma unroll
            for (int c = 0; c < BV; ++c) {
                if (c < cols) kahan_tile_sum(tile_acc[c], acc[c], corr[c]);
            }
        }
        __syncthreads();
    }

    if (active) {
        #pragma unroll
        for (int c = 0; c < BV; ++c) {
            if (c < cols) out[(size_t)(i0 + tid) * cols + c] = acc[c];
        }
    }
}

static ffi::Error cuda_error(const char* ctx, cudaError_t err) {
    if (err == cudaSuccess) return ffi::Error::Success();
    std::ostringstream os;
    os << ctx << ": " << cudaGetErrorString(err);
    return ffi::Error::Internal(os.str());
}

static ffi::Error validate(int64_t Nq, int64_t Nk, int64_t D, int64_t cols) {
    if (D <= 0 || D % KCHUNK != 0)
        return ffi::Error::InvalidArgument("feature D must be positive and padded to multiple of 32");
    if (Nq % BM != 0 || Nk % BN != 0)
        return ffi::Error::InvalidArgument("query and key N must be padded to multiple of 64");
    if (cols <= 0 || cols > BV)
        return ffi::Error::InvalidArgument("RHS column count must be in [1,16]");
    return ffi::Error::Success();
}

static ffi::Error launch(
    cudaStream_t stream,
    const float* Qs, const float* Q, const float* V,
    const float* qi, const float* qk, const float* beta,
    float* out, int64_t Nq, int64_t Nk, int64_t D, int64_t cols, bool rbf)
{
    if (auto e = validate(Nq, Nk, D, cols); e.failure()) return e;
    if (Nq == 0) return ffi::Error::Success();
    if (Nk == 0) {
        return cuda_error("cudaMemsetAsync", cudaMemsetAsync(out, 0,
            (size_t)Nq * cols * sizeof(float), stream));
    }
    // Avoid interpreting the device pointer as a host scalar. beta is read
    // by the kernel on the GPU, making it a true runtime JAX input.
    if (rbf) {
        exp_gemm_kernel<true><<<Nq / BM, THREADS, SMEM_BYTES, stream>>>(
            Qs, Q, V, qi, qk, beta, out, (int)Nq, (int)Nk, (int)D, (int)cols);
    } else {
        exp_gemm_kernel<false><<<Nq / BM, THREADS, SMEM_BYTES, stream>>>(
            Qs, Q, V, nullptr, nullptr, nullptr, out,
            (int)Nq, (int)Nk, (int)D, (int)cols);
    }
    return cuda_error("exp_gemm_kernel launch", cudaGetLastError());
}

using B1 = ffi::BufferR1<ffi::F32>;
using B2 = ffi::BufferR2<ffi::F32>;
using B0 = ffi::BufferR0<ffi::F32>;
using R1 = ffi::ResultBufferR1<ffi::F32>;
using R2 = ffi::ResultBufferR2<ffi::F32>;

ffi::Error DotVec(cudaStream_t stream, B2 Qs, B2 Q, B1 v, R1 out) {
    auto a = Qs.dimensions(); auto b = Q.dimensions();
    if (a[1] != b[1] || v.dimensions()[0] != b[0] || out->dimensions()[0] != a[0])
        return ffi::Error::InvalidArgument("dot matvec input/output shape mismatch");
    return launch(stream, Qs.typed_data(), Q.typed_data(), v.typed_data(),
                  nullptr, nullptr, nullptr, out->typed_data(), a[0], b[0], a[1], 1, false);
}
ffi::Error DotMat(cudaStream_t stream, B2 Qs, B2 Q, B2 V, R2 out) {
    auto a = Qs.dimensions(); auto b = Q.dimensions();
    auto v = V.dimensions(); auto o = out->dimensions();
    if (a[1] != b[1] || v[0] != b[0] || o[0] != a[0] || o[1] != v[1])
        return ffi::Error::InvalidArgument("dot matmat input/output shape mismatch");
    return launch(stream, Qs.typed_data(), Q.typed_data(), V.typed_data(),
                  nullptr, nullptr, nullptr, out->typed_data(), a[0], b[0], a[1], v[1], false);
}
ffi::Error RbfVec(cudaStream_t stream, B2 Qs, B2 Q, B1 v, B1 qi, B1 qk, B0 beta, R1 out) {
    auto a = Qs.dimensions(); auto b = Q.dimensions();
    if (a[1] != b[1] || v.dimensions()[0] != b[0] ||
        qi.dimensions()[0] != a[0] || qk.dimensions()[0] != b[0] ||
        out->dimensions()[0] != a[0])
        return ffi::Error::InvalidArgument("RBF matvec input/output shape mismatch");
    return launch(stream, Qs.typed_data(), Q.typed_data(), v.typed_data(),
                  qi.typed_data(), qk.typed_data(), beta.typed_data(),
                  out->typed_data(), a[0], b[0], a[1], 1, true);
}
ffi::Error RbfMat(cudaStream_t stream, B2 Qs, B2 Q, B2 V, B1 qi, B1 qk, B0 beta, R2 out) {
    auto a = Qs.dimensions(); auto b = Q.dimensions();
    auto v = V.dimensions(); auto o = out->dimensions();
    if (a[1] != b[1] || v[0] != b[0] || o[0] != a[0] || o[1] != v[1] ||
        qi.dimensions()[0] != a[0] || qk.dimensions()[0] != b[0])
        return ffi::Error::InvalidArgument("RBF matmat input/output shape mismatch");
    return launch(stream, Qs.typed_data(), Q.typed_data(), V.typed_data(),
                  qi.typed_data(), qk.typed_data(), beta.typed_data(),
                  out->typed_data(), a[0], b[0], a[1], v[1], true);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(XgemmSm80DotVec, DotVec,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<B2>().Arg<B2>().Arg<B1>().Ret<R1>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(XgemmSm80DotMat, DotMat,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<B2>().Arg<B2>().Arg<B2>().Ret<R2>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(XgemmSm80RbfVec, RbfVec,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<B2>().Arg<B2>().Arg<B1>().Arg<B1>().Arg<B1>().Arg<B0>().Ret<R1>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(XgemmSm80RbfMat, RbfMat,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<B2>().Arg<B2>().Arg<B2>().Arg<B1>().Arg<B1>().Arg<B0>().Ret<R2>());

// Use Python's C API instead of pybind11. This reduces the standalone
// build dependencies to CUDA, Python headers, and JAX's bundled FFI headers.
static PyObject* registrations(PyObject*, PyObject*) {
    PyObject* d = PyDict_New();
    if (!d) return nullptr;
    struct Target { const char* name; void* handler; };
    Target targets[] = {
        {"xgemm_sm80_fp32_matvec",     reinterpret_cast<void*>(XgemmSm80DotVec)},
        {"xgemm_sm80_fp32_matmat",     reinterpret_cast<void*>(XgemmSm80DotMat)},
        {"xgemm_sm80_fp32_matvec_rbf", reinterpret_cast<void*>(XgemmSm80RbfVec)},
        {"xgemm_sm80_fp32_matmat_rbf", reinterpret_cast<void*>(XgemmSm80RbfMat)},
    };
    for (const Target& t : targets) {
        PyObject* capsule = PyCapsule_New(t.handler, nullptr, nullptr);
        if (!capsule) { Py_DECREF(d); return nullptr; }
        if (PyDict_SetItemString(d, t.name, capsule) < 0) {
            Py_DECREF(capsule); Py_DECREF(d); return nullptr;
        }
        Py_DECREF(capsule);
    }
    return d;
}
static PyMethodDef methods[] = {
    {"registrations", (PyCFunction)registrations, METH_NOARGS, "Return JAX typed-FFI call targets."},
    {nullptr, nullptr, 0, nullptr}
};
static struct PyModuleDef moddef = {
    PyModuleDef_HEAD_INIT, "_sm80", "SM80 FP32 exponential GEMM FFI", -1,
    methods, nullptr, nullptr, nullptr, nullptr
};
PyMODINIT_FUNC PyInit__sm80(void) { return PyModule_Create(&moddef); }
