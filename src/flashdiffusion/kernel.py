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
# Distance tile  (shared by prepass and matvec)
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
    Q:         np.ndarray,                   # (N_q, d)
    K:         np.ndarray,                   # (N_k, d)
    V:         np.ndarray,                   # (N_k, r)  r columns
    beta:      float,
    rscale_q:  Optional[np.ndarray] = None,  # (N_q,)  default = 1
    rscale_k:  Optional[np.ndarray] = None,  # (N_k,)  default = 1
    h:         Optional[np.ndarray] = None,  # (N_k,)  Doob potential, default = 1
    tile:      int = 512,
) -> np.ndarray:
    """
    FlashDiffusion(Q, K, V, beta, rscale_q, rscale_k, h, tile)

    out_i = rscale_q_i · Σⱼ exp(-beta·||qᵢ-kⱼ||²) · rscale_k_j · h_j · v_j

    Parameters
    ----------
    Q, K   : coordinate matrices (may be identical for DMAP)
    V      : (N_k, r) right-hand side vectors
    beta   : kernel bandwidth
    rscale_q, rscale_k : Coifman-Lafon normalisation diagonals
                         set both to precomputed rscale for DMAP
    h      : Doob h-transform potential (optional)
    tile   : tile size controlling peak SMEM / GMEM traffic

    Returns
    -------
    out : (N_q, r) float64

    Notes
    -----
    - No online softmax: rscale is precomputed, epilogue is a scalar multiply.
    - Tile loop is embarrassingly parallel — maps directly to CuTe thread blocks.
    - For the CUDA kernel: dist2 tile is a GEMM + diagonal bias; exp is the
      epilogue; accumulation is plain float32 register tile (no max tracking).
    """
    N_q, d = Q.shape
    N_k, r = V.shape

    Q32 = Q.astype(np.float32)
    K32 = K.astype(np.float32)
    sqQ = (Q32 * Q32).sum(axis=1)
    sqK = (K32 * K32).sum(axis=1)

    # pre-scale the RHS:  u_j = rscale_k_j · h_j · v_j
    u = V.astype(np.float64)
    if rscale_k is not None:
        u = rscale_k[:, None] * u
    if h is not None:
        u = h[:, None] * u

    out = np.zeros((N_q, r), dtype=np.float64)

    for i0 in range(0, N_q, tile):
        i1  = min(i0 + tile, N_q)
        Qi  = Q32[i0:i1]
        sqi = sqQ[i0:i1]

        for j0 in range(0, N_k, tile):
            j1  = min(j0 + tile, N_k)
            Kj  = K32[j0:j1]
            sqk = sqK[j0:j1]

            d2     = _dist2_tile(Qi, Kj, sqi, sqk)
            K_tile = np.exp(-beta * d2)               # (ti, tj) — epilogue
            out[i0:i1] += K_tile @ u[j0:j1]           # accumulate

    # post-scale by rscale_q
    if rscale_q is not None:
        out *= rscale_q[:, None]

    return out
