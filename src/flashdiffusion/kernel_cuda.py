"""
flashdiffusion/kernel_cuda.py
==============================
Python wrapper around flash_diffusion_cuda (the compiled SM80 extension).

This module is imported by kernel.py when CUDA is available.
It mirrors the numpy backend interface exactly — only the compute happens
on GPU. All tensors are managed here; the caller (kernel.py) stays backend-agnostic.

No gradients anywhere: we are computing eigenvectors of a fixed symmetric
operator (the harmonic basis). There is no training loop through this kernel.
If you later want to train a model that *uses* diffusion coordinates, that
goes through flexattention on the transformed coordinates, not through here.

Precision contract
------------------
  X storage   : fp16  (halved bandwidth vs fp32, safe because ||xi-xj||^2
                        is computed as sqi + sqj - 2*dot with fp32 accum)
  dot product : fp32  (mma.sync fp16->fp32, avoids catastrophic cancellation)
  exp, accum  : fp32
  rscale, D   : fp32
  Lanczos vec : fp32  on GPU,  float64 orthogonalisation on CPU (unchanged)
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Optional

# Lazy import — only loaded if CUDA is available and extension is compiled
_ext = None

def _load_ext():
    global _ext
    if _ext is None:
        try:
            import flash_diffusion_cuda as ext
            _ext = ext
        except ImportError as e:
            raise ImportError(
                "flash_diffusion_cuda extension not compiled. "
                "Run: pip install -e '.[cuda]' or "
                "cd csrc && nvcc -arch=sm_80 ... "
            ) from e
    return _ext


def _to_gpu_fp16(X: np.ndarray) -> torch.Tensor:
    """Move coordinate array to GPU as fp16."""
    return torch.from_numpy(np.asarray(X, dtype=np.float16)).cuda()


def _to_gpu_fp32(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).cuda()


class CUDAState:
    """
    Cached GPU tensors for a fixed (X, beta, alpha).
    Analogous to the 'norm' dict in the numpy backend, but lives on GPU.

    Built once by precompute_cuda(), reused across all Lanczos iterations.
    The K-side tensors (X_fp16, sq_X, rscale) are the "KV cache" —
    they never change while only the Lanczos vector v changes.
    """

    def __init__(
        self,
        X_fp16:  torch.Tensor,   # (N, d) fp16 on GPU
        sq_X:    torch.Tensor,   # (N,)   fp32 on GPU
        rscale:  torch.Tensor,   # (N,)   fp32 on GPU
        D:       torch.Tensor,   # (N,)   fp32 on GPU  — raw row sums
        D_alpha: torch.Tensor,   # (N,)   fp32 on GPU
        beta:    float,
        alpha:   float,
    ):
        self.X_fp16  = X_fp16
        self.sq_X    = sq_X
        self.rscale  = rscale
        self.D       = D
        self.D_alpha = D_alpha
        self.beta    = beta
        self.alpha   = alpha
        self.N       = X_fp16.shape[0]
        self.d       = X_fp16.shape[1]


def precompute_cuda(
    X:     np.ndarray,
    beta:  float,
    alpha: float = 0.5,
) -> CUDAState:
    """
    Two-pass precomputation on GPU.  Mirrors precompute() in kernel.py.

    Pass 1: D      = K @ ones   (raw row sums)
    Pass 2: D_alpha = (K @ w) * w    where w = D^{-alpha}
    rscale = D_alpha^{-1/2} * w

    Returns a CUDAState with all GPU tensors cached.
    """
    ext = _load_ext()
    N, d = X.shape

    print(f"[precompute_cuda] N={N}  d={d}  beta={beta}  alpha={alpha}")

    # Move X to GPU as fp16 — stays there for the entire eigensolver run
    X_fp16 = _to_gpu_fp16(X)

    # Precompute sq norms (fp32, stays on GPU)
    sq_X = ext.compute_sq_norms(X_fp16)

    # Pass 1: D_i = Σ_j K(i,j)   (w = 1)
    w1 = torch.ones(N, dtype=torch.float32, device='cuda')
    D  = ext.prepass(X_fp16, w1, sq_X, beta)

    # w = D^{-alpha}
    w  = D.pow(-alpha)

    # Pass 2: D_alpha_i = (Σ_j K(i,j) * w_j) * w_i
    D_alpha = ext.prepass(X_fp16, w, sq_X, beta) * w

    # Combined diagonal
    rscale = D_alpha.pow(-0.5) * w

    return CUDAState(X_fp16, sq_X, rscale, D, D_alpha, beta, alpha)


def matvec_cuda(state: CUDAState, v: np.ndarray) -> np.ndarray:
    """
    Apply M to a single Lanczos vector v (N,) float64.

    v arrives as numpy float64 (from ARPACK).
    We cast to float32, push to GPU, run the kernel, pull back as float64.

    The CPU<->GPU transfer for v is O(N) floats — negligible vs O(N²) compute.
    The K-side (X_fp16, sq_X, rscale) never moves — it's the KV cache.
    """
    ext = _load_ext()
    v_gpu = _to_gpu_fp32(v.astype(np.float32))

    with torch.no_grad():
        out_gpu = ext.matvec(
            state.X_fp16,
            v_gpu,
            state.rscale,
            state.sq_X,
            state.beta,
        )

    # Pull back to CPU as float64 for ARPACK orthogonalisation
    return out_gpu.cpu().numpy().astype(np.float64)
