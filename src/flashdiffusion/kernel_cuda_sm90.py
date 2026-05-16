"""
flashdiffusion/kernel_cuda_sm90.py
===================================
Python wrapper for the SM90 (H100 / Hopper) wgmma+TMA kernel.

Same interface as kernel_cuda.py (SM80) and kernel_cuda_sm120.py.
Only the module name and precompute/matvec function names differ.

TMA descriptor construction happens in the .cu host-side code.
The descriptors are rebuilt per-call (cheap — host-side only).
In production, cache them alongside rscale in CUDAStateSM90.
"""

from __future__ import annotations
import numpy as np
import torch
import time
from typing import Optional

_ext = None   # injected by notebook or loaded via _load_ext()


def _load_ext():
    global _ext
    if _ext is None:
        try:
            import flash_diffusion_sm90 as ext
            _ext = ext
        except ImportError as e:
            raise ImportError(
                "flash_diffusion_sm90 not compiled.\n"
                "Build with: FLASHDIFFUSION_BUILD_CUDA=1 "
                "TORCH_CUDA_ARCH_LIST='9.0a' pip install -e '.[cuda]'\n"
                "Requires CUTLASS >= 3.5.1 headers."
            ) from e
    return _ext


def _to_gpu_fp16(X: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(X, dtype=np.float16)).cuda()


class CUDAStateSM90:
    """Cached GPU tensors for SM90. Same interface as SM80/SM120 states."""

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


def precompute_sm90(
    X:     np.ndarray,
    beta:  float,
    alpha: float = 0.5,
) -> CUDAStateSM90:
    """
    Two-pass precomputation on SM90 GPU.
    TMA is used inside the kernel for GMEM→SMEM loads.
    """
    ext = _load_ext()
    N, d = X.shape
    print(f"[precompute_sm90] N={N}  d={d}  beta={beta}  alpha={alpha}")

    X_fp16 = _to_gpu_fp16(X)
    sq_X   = ext.compute_sq_norms(X_fp16)

    t0 = time.perf_counter()
    w1      = torch.ones(N, dtype=torch.float32, device='cuda')
    D       = ext.prepass(X_fp16, w1, sq_X, beta)
    w       = D.pow(-alpha)
    D_alpha = ext.prepass(X_fp16, w, sq_X, beta) * w
    rscale  = D_alpha.pow(-0.5) * w
    print(f"  precompute done (SM90 wgmma+TMA) in {time.perf_counter()-t0:.3f}s")

    return CUDAStateSM90(X_fp16, sq_X, rscale, D, D_alpha, beta, alpha)


def matvec_sm90(state: CUDAStateSM90, v: np.ndarray) -> np.ndarray:
    """Apply M to Lanczos vector v using SM90 wgmma kernel."""
    ext = _load_ext()
    v_gpu = torch.from_numpy(v.astype(np.float32)).cuda()
    with torch.no_grad():
        out = ext.matvec(
            state.X_fp16, v_gpu, state.rscale, state.sq_X, state.beta)
    return out.cpu().numpy().astype(np.float64)
