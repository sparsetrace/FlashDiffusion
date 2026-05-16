"""
flashdiffusion/kernel_cuda_sm100.py
=====================================
Python wrapper for the SM100 (B200 / Blackwell datacenter) kernel.

v1: Uses WMMA (nvcuda::wmma) tensor cores — forward-compatible from Volta,
    works on SM100 but does not use Blackwell-native UMMA (tcgen05.mma).
    Correct results, good performance. UMMA path is v2.

Same interface as kernel_cuda.py (SM80), kernel_cuda_sm90.py, kernel_cuda_sm120.py.

Build:
    FLASHDIFFUSION_BUILD_CUDA=1 TORCH_CUDA_ARCH_LIST="10.0a" pip install -e ".[cuda]"

    Or in notebook:
        -arch=sm_100a -std=c++17 -O3 --use_fast_math
        -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__
        --expt-relaxed-constexpr

Note: sm_100a (architecture-accelerated) required for full SM100 feature set.
      sm_100 (without 'a') also works for WMMA but misses some SM100 features.
"""

from __future__ import annotations
import numpy as np
import torch
import time

_ext = None   # injected by notebook or loaded via _load_ext()


def _load_ext():
    global _ext
    if _ext is None:
        try:
            import flash_diffusion_sm100 as ext
            _ext = ext
        except ImportError as e:
            raise ImportError(
                "flash_diffusion_sm100 not compiled.\n"
                "Build with: FLASHDIFFUSION_BUILD_CUDA=1 "
                "TORCH_CUDA_ARCH_LIST='10.0a' pip install -e '.[cuda]'\n"
                "Requires: CUDA >= 12.0, SM100 GPU (B200/B300)"
            ) from e
    return _ext


def _to_gpu_fp16(X: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(X, dtype=np.float16)).cuda()


class CUDAStateSM100:
    """Cached GPU tensors for SM100. Same interface as SM80/SM90/SM120 states."""

    def __init__(self, X_fp16, sq_X, rscale, D, D_alpha, beta, alpha):
        self.X_fp16  = X_fp16
        self.sq_X    = sq_X
        self.rscale  = rscale
        self.D       = D
        self.D_alpha = D_alpha
        self.beta    = beta
        self.alpha   = alpha
        self.N       = X_fp16.shape[0]
        self.d       = X_fp16.shape[1]


def precompute_sm100(
    X:     np.ndarray,
    beta:  float,
    alpha: float = 0.5,
) -> CUDAStateSM100:
    """
    Two-pass Coifman-Lafon precomputation on SM100 GPU.
    Uses WMMA tensor cores for Xi @ Xj.T dot products.

    Pass 1: D_i      = sum_j exp(-beta * dist2(i,j))
    Pass 2: D_alpha_i = sum_j exp(-beta * dist2(i,j)) * D_j^{-alpha}
    rscale_i = D_alpha_i^{-0.5} * D_i^{-alpha}
    """
    ext = _load_ext()
    N, d = X.shape
    print(f"[precompute_sm100] N={N}  d={d}  beta={beta}  alpha={alpha}")

    X_fp16 = _to_gpu_fp16(X)
    sq_X   = ext.compute_sq_norms(X_fp16)

    t0 = time.perf_counter()

    # pass 1: uniform weights → D
    w1 = torch.ones(N, dtype=torch.float32, device='cuda')
    D  = ext.prepass(X_fp16, w1, sq_X, beta)

    # pass 2: density-corrected weights → D_alpha
    w2      = D.pow(-alpha)
    D_alpha = ext.prepass(X_fp16, w2, sq_X, beta) * w2

    # precomputed rscale = D_alpha^{-1/2} * D^{-alpha}
    rscale = D_alpha.pow(-0.5) * D.pow(-alpha)

    elapsed = time.perf_counter() - t0
    print(f"  precompute done (SM100 WMMA) in {elapsed:.3f}s")

    return CUDAStateSM100(X_fp16, sq_X, rscale, D, D_alpha, beta, alpha)


def matvec_sm100(state: CUDAStateSM100, v: np.ndarray) -> np.ndarray:
    """
    Apply diffusion map operator M to Lanczos vector v.
    M v = rscale * (K @ (rscale * v))

    GPU matvec using WMMA tensor cores.
    Returns numpy float64 for ARPACK compatibility.
    """
    ext = _load_ext()
    v_gpu = torch.from_numpy(v.astype(np.float32)).cuda()
    with torch.no_grad():
        out = ext.matvec(
            state.X_fp16, v_gpu, state.rscale, state.sq_X, state.beta)
    return out.cpu().numpy().astype(np.float64)
