"""
flashdiffusion/utils.py
=======================
Bandwidth selection, Nyström out-of-sample extension, dtype helpers.
"""

from __future__ import annotations
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Bandwidth selection
# ---------------------------------------------------------------------------

def median_bandwidth(X: np.ndarray, subsample: int = 2000) -> float:
    """
    Median heuristic for beta: beta = 1 / (2 * median(||xi-xj||²)).

    Subsamples if N > subsample to keep cost O(subsample²).
    """
    N = X.shape[0]
    if N > subsample:
        idx = np.random.default_rng(0).choice(N, subsample, replace=False)
        X = X[idx]
    sq   = (X * X).sum(axis=1)
    dist2 = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0)
    med   = np.median(dist2[dist2 > 0])
    return 1.0 / (2.0 * med) if med > 0 else 1.0


def knn_bandwidth(X: np.ndarray, k: int = 7) -> np.ndarray:
    """
    Self-tuning (Zelnik-Manor & Perona) per-point bandwidth.

    sigma_i = ||xi - x_{i,k}||   (distance to k-th nearest neighbour)

    For use with the anisotropic kernel:
        K(i,j) = exp(-||xi-xj||² / (sigma_i * sigma_j))

    Returns sigma (N,) — to use, replace beta * dist2 with dist2 / outer(sigma, sigma).
    Note: this changes the kernel primitive; requires a modified prepass.
    Currently returned as metadata for the caller to handle.
    """
    N = X.shape[0]
    sq    = (X * X).sum(1)
    dist2 = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0)
    # k-th nearest neighbour distance (excluding self at dist=0)
    sorted_d = np.sort(dist2, axis=1)
    sigma = np.sqrt(sorted_d[:, k])              # (N,)
    sigma = np.maximum(sigma, 1e-10)
    return sigma


# ---------------------------------------------------------------------------
# Nyström out-of-sample extension
# ---------------------------------------------------------------------------

def nystrom_extend(
    dm,                              # fitted DiffusionMap
    X_new: np.ndarray,               # (M, d)  new points
    t:     int = 1,
) -> np.ndarray:
    """
    Nyström extension: embed new points using the training eigenvectors.

    For a new point x*, the diffusion map coordinate is:
        ψₖ(x*) = (1/λₖ) · Σᵢ p(x*, xᵢ) · φₖ(xᵢ)

    where p(x*, xᵢ) = K_alpha(x*, xᵢ) / D_alpha(x*)
    and φₖ are the (row-normalised) eigenvectors of P⁺.

    This requires ONE tiled kernel pass for each new point batch,
    reusing the precomputed rscale_ from the training set.

    Returns
    -------
    coords : (M, n_components-1) diffusion coordinates for X_new
    """
    from .kernel import FlashDiffusion, prepass_rowsum

    if dm.eigenvalues_ is None:
        raise RuntimeError("Call transform() on training data first.")

    beta, alpha, tile = dm.beta, dm.alpha, dm.tile
    X_train  = dm.X_
    N_train  = X_train.shape[0]
    rscale_k = dm.rscale_

    # normalise new points against training set
    # D_new_i = Σⱼ K(x*_i, x_j)  · w_j   (using training w)
    w_train = dm.D_ ** (-alpha)
    D_new   = prepass_rowsum(X_new, X_train, beta,
                              weights_k=w_train, tile=tile)
    w_new   = D_new ** (-alpha)

    # D_alpha_new_i = (Σⱼ K(x*_i, xⱼ) · w_j) · w_new_i  (approx, Nyström)
    D_alpha_new = D_new * w_new          # approximate row sum in alpha-kernel
    inv_Dalpha_new = w_new / D_alpha_new # = 1/D_new  (since D_alpha_new = D_new*w)

    # rscale for new query points
    rscale_q_new = D_alpha_new ** (-0.5) * w_new

    # get training eigenvectors in P⁺ basis
    phi = dm.eigenvectors_ / dm.rscale_[:, None]   # (N_train, k)
    phi /= np.linalg.norm(phi, axis=0, keepdims=True)

    # Nyström: apply K_alpha from new -> training
    # coords_new_k = (1/λₖ) · Σⱼ K_alpha(x*_i, xⱼ) · φₖ(xⱼ)
    coords_new = FlashDiffusion(
        Q        = X_new,
        K        = X_train,
        V        = phi[:, 1:],           # drop trivial first eigvec
        beta     = beta,
        rscale_q = rscale_q_new,
        rscale_k = rscale_k,
        tile     = tile,
    )
    # scale by 1/λₖ and λₖ^t -> net scale λₖ^{t-1}
    lam = dm.eigenvalues_[1:]
    coords_new *= (lam ** (t - 1))[None, :]

    return coords_new                    # (M, k-1)


# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------

def to_f32(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float32)

def to_f64(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float64)
