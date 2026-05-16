"""
flashdiffusion/kernel_cuda_sm120.py
=====================================
Python wrapper for the SM120 (RTX 5090 / consumer Blackwell) kernel.

Differences from SM80 wrapper:
  - module name: flash_diffusion_sm120
  - larger tiles (128×128 vs 64×64) — same interface
  - cp.async pipeline baked into kernel — no API change needed
  - compile with -arch=sm_120
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Optional

_ext = None   # injected by notebook or loaded via _load_ext()


def _load_ext():
    global _ext
    if _ext is None:
        try:
            import flash_diffusion_sm120 as ext
            _ext = ext
        except ImportError as e:
            raise ImportError(
                "flash_diffusion_sm120 not compiled. Build with:\n"
                "  FLASHDIFFUSION_BUILD_CUDA=1 "
                "  TORCH_CUDA_ARCH_LIST='12.0' "
                "  pip install -e '.[cuda]'"
            ) from e
    return _ext


def _to_gpu_fp16(X: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(X, dtype=np.float16)).cuda()


def _to_gpu_fp32(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).cuda()


class CUDAStateSM120:
    """Cached GPU tensors for SM120. Same interface as CUDAState (SM80)."""

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


def precompute_sm120(
    X:     np.ndarray,
    beta:  float,
    alpha: float = 0.5,
) -> CUDAStateSM120:
    """Two-pass precomputation on SM120 GPU."""
    import time
    ext = _load_ext()
    N, d = X.shape
    print(f"[precompute_sm120] N={N}  d={d}  beta={beta}  alpha={alpha}")

    X_fp16 = _to_gpu_fp16(X)
    sq_X   = ext.compute_sq_norms(X_fp16)

    t0 = time.perf_counter()
    w1 = torch.ones(N, dtype=torch.float32, device='cuda')
    D  = ext.prepass(X_fp16, w1, sq_X, beta)
    w  = D.pow(-alpha)
    D_alpha = ext.prepass(X_fp16, w, sq_X, beta) * w
    rscale  = D_alpha.pow(-0.5) * w
    print(f"  precompute done (SM120) in {time.perf_counter()-t0:.3f}s")

    return CUDAStateSM120(X_fp16, sq_X, rscale, D, D_alpha, beta, alpha)


def matvec_sm120(state: CUDAStateSM120, v: np.ndarray) -> np.ndarray:
    """Apply M to Lanczos vector v. GPU→GPU, pull back to CPU for ARPACK."""
    ext = _load_ext()
    v_gpu = torch.from_numpy(v.astype(np.float32)).cuda()
    with torch.no_grad():
        out = ext.matvec(
            state.X_fp16, v_gpu, state.rscale, state.sq_X, state.beta)
    return out.cpu().numpy().astype(np.float64)
