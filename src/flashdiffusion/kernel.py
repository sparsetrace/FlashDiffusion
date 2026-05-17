"""
flashdiffusion/kernel.py
========================
The FlashDiffusion primitive.

FlashDiffusion(Q, K, V, beta, alpha, h, rscale_q, rscale_k, tile)

Computes, tile by tile, without materialising the O(N²) kernel matrix:

    score(i,j)   = beta * ||q_i - k_j||²              (Gaussian, default)
                 | score_mod(q_i, k_j)                 (custom, future)

    K(i,j)       = exp(-score(i,j))

    out_i        = rscale_q_i · Σⱼ K(i,j) · rscale_k_j · (h_j · v_j)

where rscale_{q,k} encode the Coifman-Lafon normalisation and h is an
optional Doob potential (default h=1, no transform).

Modes
-----
DMAP matvec    Q=K=X,  rscale_q=rscale_k=rscale (precomputed by DiffusionMap)
AMAP matvec    Q≠K,    rscale_q/k separate (directed/NESS operator, DAC §3)
Doob mode      h ≠ 1:  h-transform of either operator (DAC §4, Theorem 5.1)
Prepass        V=1 (or V=w), h=1: computes row sums D or D_alpha

Complexity
----------
Arithmetic : O(N_q · N_k · d)   — same as dense, unavoidable
Memory     : O((N_q + N_k) · tile)  — never O(N²)

Current backend: NumPy (CPU reference).
CUDA target:     CuTe SM90, fused prepass+matvec, tensor-core GEMM for
                 the xi·xj term, elementwise exp epilogue. No online
                 softmax needed — rscale is precomputed.
"""

from __future__ import annotations
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Backend detection + dispatch
# ---------------------------------------------------------------------------

def _cuda_sm() -> int | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability()
        return major * 10 + minor
    except ImportError:
        return None

# SM -> which kernel family to use
# SM120 (RTX 50xx) uses mma.sync (same family as SM80), not wgmma
_SM_BACKEND = {
    80:  "sm80", 86: "sm80", 89: "sm80",
    90:  "sm90",
    100: "sm100",
    103: "sm100",
    120: "sm120", 121: "sm120",
}

def _get_cuda_backend():
    """Return (precompute_fn, matvec_fn) or None if unavailable."""
    sm = _cuda_sm()
    if sm is None:
        return None
    backend_name = _SM_BACKEND.get(sm, "sm80")

    if backend_name == "sm80":
        try:
            from .kernel_cuda import precompute_cuda, matvec_cuda
            return precompute_cuda, matvec_cuda
        except (ImportError, OSError):
            return None

    elif backend_name == "sm90":
        try:
            from .kernel_cuda_sm90 import precompute_sm90, matvec_sm90
            return precompute_sm90, matvec_sm90
        except (ImportError, OSError):
            try:
                from .kernel_cuda import precompute_cuda, matvec_cuda
                print("[FlashDiffusion] SM90 not found, falling back to SM80")
                return precompute_cuda, matvec_cuda
            except (ImportError, OSError):
                return None

    elif backend_name == "sm100":
        try:
            from .kernel_cuda_sm100 import precompute_sm100, matvec_sm100
            return precompute_sm100, matvec_sm100
        except (ImportError, OSError):
            try:
                from .kernel_cuda import precompute_cuda, matvec_cuda
                print("[FlashDiffusion] SM100 not found, falling back to SM80")
                return precompute_cuda, matvec_cuda
            except (ImportError, OSError):
                return None

    elif backend_name == "sm120":
        try:
            from .kernel_cuda_sm120 import precompute_sm120, matvec_sm120
            return precompute_sm120, matvec_sm120
        except (ImportError, OSError):
            try:
                from .kernel_cuda import precompute_cuda, matvec_cuda
                print("[FlashDiffusion] SM120 not found, falling back to SM80")
                return precompute_cuda, matvec_cuda
            except (ImportError, OSError):
                return None

    return None


# ---------------------------------------------------------------------------
# Distance tile  (shared by numpy prepass and matvec)
# ---------------------------------------------------------------------------

def _dist2_tile(
    Qi: np.ndarray,   # (ti, d)  float32
    Kj: np.ndarray,   # (tj, d)  float32
    sqi: np.ndarray,  # (ti,)
    sqk: np.ndarray,  # (tj,)
) -> np.ndarray:
    """Squared Euclidean distances for one (Q-tile, K-tile) pair. float64 output."""
    d2 = sqi[:, None] + sqk[None, :] - 2.0 * (Qi @ Kj.T)
    np.maximum(d2, 0.0, out=d2)
    return d2.astype(np.float64)


# ---------------------------------------------------------------------------
# Prepass: row-sum reduction  (used twice in precompute_normalization)
# ---------------------------------------------------------------------------

def prepass_rowsum(
    Q: np.ndarray,            # (N_q, d)
    K: np.ndarray,            # (N_k, d)
    beta: float,
    weights_k: Optional[np.ndarray] = None,   # (N_k,)  default = 1
    tile: int = 512,
) -> np.ndarray:
    """
    out_i = Σⱼ exp(-beta · ||q_i - k_j||²) · weights_k_j

    Returns (N_q,) float64.
    This is the primitive used for both normalisation passes:
      pass 1:  weights_k = np.ones(N)   -> D_i  (raw row sums)
      pass 2:  weights_k = w = D^{-α}  -> D_alpha_i (before * w_i)
    """
    N_q, d = Q.shape
    N_k    = K.shape[0]
    if weights_k is None:
        weights_k = np.ones(N_k, dtype=np.float64)

    Q32 = Q.astype(np.float32)
    K32 = K.astype(np.float32)
    sqQ = (Q32 * Q32).sum(axis=1)
    sqK = (K32 * K32).sum(axis=1)

    out = np.zeros(N_q, dtype=np.float64)

    for i0 in range(0, N_q, tile):
        i1  = min(i0 + tile, N_q)
        Qi  = Q32[i0:i1]
        sqi = sqQ[i0:i1]

        for j0 in range(0, N_k, tile):
            j1  = min(j0 + tile, N_k)
            Kj  = K32[j0:j1]
            sqk = sqK[j0:j1]

            d2      = _dist2_tile(Qi, Kj, sqi, sqk)
            K_tile  = np.exp(-beta * d2)              # (ti, tj)
            out[i0:i1] += K_tile @ weights_k[j0:j1]

    return out


# ---------------------------------------------------------------------------
# Matvec: the core FlashDiffusion kernel
# ---------------------------------------------------------------------------

def FlashDiffusion(
    Q:           np.ndarray,                   # (N_q, d)
    K:           np.ndarray,                   # (N_k, d)
    V:           np.ndarray,                   # (N_k, r)  r columns
    beta:        float,
    rscale_q:    Optional[np.ndarray] = None,  # (N_q,)  default = 1
    rscale_k:    Optional[np.ndarray] = None,  # (N_k,)  default = 1
    h:           Optional[np.ndarray] = None,  # (N_k,)  Doob potential
    tile:        int  = 512,
    coord_dtype       = np.float32,            # X tile dtype (f32/f16)
    accum_dtype       = np.float64,            # accumulator dtype
) -> np.ndarray:
    """
    FlashDiffusion numpy backend (CPU reference / fallback).

    out_i = rscale_q_i · Σⱼ exp(-beta·||qᵢ-kⱼ||²) · rscale_k_j · h_j · v_j

    Precision contract (numpy path)
    --------------------------------
    coord_dtype : X tile storage — fp32 default, fp16 for memory savings
    accum_dtype : accumulator — fp64 for CPU reference correctness
    GEMM (dot)  : always fp32 regardless of coord_dtype (catastrophic
                  cancellation safety for dist² = sqi + sqj - 2*dot)

    CUDA path (see kernel_cuda.py + csrc/flash_diffusion_sm80.cu)
    ---------------------------------------------------------------
    coord_dtype -> fp16 smem  (hardcoded in CUDA kernel)
    GEMM        -> mma.sync fp16->fp32 (SM80) / wgmma fp16->fp32 (SM90)
    accum_dtype -> fp32 register tile
    No online softmax: rscale precomputed, epilogue = scalar multiply.
    No gradients: we compute the harmonic basis (eigenvectors), not a
    differentiable layer. Use flexattention for the trained forward pass.
    """
    N_q, d = Q.shape
    N_k, r = V.shape

    # GEMM always in fp32 for dist² numerical safety
    Q32 = Q.astype(np.float32)
    K32 = K.astype(np.float32)
    sqQ = (Q32 * Q32).sum(axis=1)
    sqK = (K32 * K32).sum(axis=1)

    # pre-scale RHS: u_j = rscale_k_j · h_j · v_j
    u = V.astype(accum_dtype)
    if rscale_k is not None:
        u = rscale_k[:, None] * u
    if h is not None:
        u = h[:, None] * u

    out = np.zeros((N_q, r), dtype=accum_dtype)

    for i0 in range(0, N_q, tile):
        i1  = min(i0 + tile, N_q)
        Qi  = Q32[i0:i1]
        sqi = sqQ[i0:i1]

        for j0 in range(0, N_k, tile):
            j1  = min(j0 + tile, N_k)
            Kj  = K32[j0:j1]
            sqk = sqK[j0:j1]

            d2     = _dist2_tile(Qi, Kj, sqi, sqk)       # fp64
            K_tile = np.exp(-beta * d2)                   # fp64 epilogue
            out[i0:i1] += K_tile @ u[j0:j1]

    if rscale_q is not None:
        out *= rscale_q[:, None]

    return out
