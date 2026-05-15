"""
flashdiffusion/cdc.py
=====================
Carré du Champ (CDC) operator and diffusion metric tensor.

The Carré du Champ operator of the diffusion semigroup is:

    Γ(f, g)(x) = ½ [L(fg)(x) - f(x)Lg(x) - g(x)Lf(x)]

where L = I - P⁺ is the (infinitesimal) generator of the diffusion.
For discrete diffusion maps this becomes:

    Γ(φₖ, φₗ)(i) = ½ Σⱼ P⁺(i,j) [φₖ(j) - φₖ(i)] [φₗ(j) - φₗ(i)]
                  = ½ [Mₖₗ - λₖ φₖ(i) φₗ(i) - λₗ φₗ(i) φₖ(i)]    (*)

where Mₖₗ(i) = (P⁺ φₖ φₗ)(i) — the diffusion applied to the pointwise product.

(*) uses the fact that P⁺ φₖ = λₖ φₖ.

The Diffusion Metric Tensor at point i is:

    G^i_{kl} = Γ(φₖ, φₗ)(i)

and the global Riemannian metric on the embedding space is:

    M = Σᵢ Gⁱ (Gⁱ)ᵀ   (summed over data points)

This gives a data-driven geometry that is intrinsic to the diffusion —
it captures how much uncertainty/variance each direction in embedding
space has relative to the local transition structure.

References
----------
- Coifman & Lafon (2006), §3
- Singer & Coifman (2008) — diffusion distance and CDC
- Nadler, Lafon, Coifman, Kevrekidis (2006) — diffusion maps for data analysis
"""

from __future__ import annotations
import numpy as np
from typing import Optional

from .kernel import FlashDiffusion


def carre_du_champ(
    dm,                         # fitted DiffusionMap instance
    phi_k: np.ndarray,          # (N,)  eigenvector k
    phi_l: np.ndarray,          # (N,)  eigenvector l (may equal phi_k)
    lambda_k: float,
    lambda_l: float,
) -> np.ndarray:
    """
    Compute Γ(φₖ, φₗ)(i) for all i.

    Uses the identity (*):
        Γ(φₖ, φₗ)(i) = ½ [(P⁺ φₖφₗ)(i)  -  λₖ φₖ(i)φₗ(i)  -  λₗ φₗ(i)φₖ(i)]

    Only ONE additional FlashDiffusion matvec is needed (for P⁺ φₖφₗ),
    beyond the eigenvectors already computed by DiffusionMap.transform().

    Parameters
    ----------
    dm       : fitted DiffusionMap (after transform(), so rscale_ is set)
    phi_k/l  : eigenvectors of the *row-normalised* operator P+
               (i.e. vecs / rscale, as returned by dmap_coords)
    lambda_k/l : corresponding eigenvalues

    Returns
    -------
    gamma : (N,) pointwise Carré du Champ values
    """
    # pointwise product of the two eigenfunctions
    phi_kl = phi_k * phi_l                   # (N,)

    # apply P⁺ to phi_kl via FlashDiffusion
    # P⁺ f = (1/D_alpha) * K_alpha * f
    #      = rscale * K * (rscale * f)    (using our normalisation)
    # but phi_k/l are already in the P⁺-eigenvector basis, so we apply
    # the DMAP matvec directly
    P_phi_kl = dm.matvec(phi_kl)             # (N,)  = M applied to (phi_kl/rscale) * rscale
    # correction: dm.matvec applies the SYMMETRIC M, not P⁺.
    # We need P⁺ = D_alpha^{-1} K_alpha = rscale/d_half * M / rscale * d_half
    # But phi_k/l are P⁺ eigenvectors: P⁺ φ = λ φ.
    # For Γ we need (P⁺ φₖφₗ)(i), i.e. row-normalised application.
    # The row-normalised operator applied to f is:
    #   (P⁺ f)_i = rscale_i * Σⱼ K(i,j) * rscale_j * (f_j / rscale_j) * rscale_j
    # Wait — let's be explicit. With our notation:
    #   P⁺_{ij} = K_alpha(i,j) / D_alpha_i
    # and rscale_i = D_alpha_i^{-1/2} * w_i, so:
    #   (P⁺ f)_i = Σⱼ [K(i,j)*w_i*w_j / D_alpha_i] * f_j
    #            = [rscale_i/w_i]² * Σⱼ K(i,j) * (rscale_j/w_j)^{-1} * rscale_j * f_j ... complex
    # Simplest: use matvec (symmetric M) and note M = D_alpha^{-1/2} P⁺_sym D_alpha^{1/2}
    # For the CDC formula (*) with phi_k in P⁺ basis, apply P⁺ directly:
    P_phi_kl = _apply_Pplus(dm, phi_kl)

    gamma = 0.5 * (P_phi_kl
                   - lambda_k * phi_k * phi_l
                   - lambda_l * phi_l * phi_k)
    return gamma


def _apply_Pplus(dm, f: np.ndarray) -> np.ndarray:
    """
    Apply the row-normalised operator P⁺ to a function f.

    P⁺ f_i = Σⱼ K_alpha(i,j) / D_alpha_i · f_j
           = (1/D_alpha_i) · w_i · Σⱼ K(i,j) · w_j · f_j

    This is a FlashDiffusion call with rscale_q = 1/D_alpha (not d_half*w).
    """
    N = dm.X_.shape[0]
    # rscale_k for right side: w (density correction only)
    # rscale_q for left side: 1/D_alpha (full row normalisation)
    w        = dm.D_  ** (-dm.alpha)
    inv_Dalpha = dm.D_alpha_ ** (-1.0)

    out = FlashDiffusion(
        Q        = dm.X_,
        K        = dm.X_,
        V        = f[:, None],
        beta     = dm.beta,
        rscale_q = inv_Dalpha * w,    # (D_alpha^{-1}) * w = P⁺ left normaliser
        rscale_k = w,                 # right: w_j factor
        tile     = dm.tile,
    )
    return out.ravel()


def metric_tensor(
    dm,
    eigenvalues:  np.ndarray,       # (k,)
    eigenvectors: np.ndarray,       # (N, k)  in P⁺ basis
    pairs: Optional[list] = None,   # list of (k, l) index pairs; default all
) -> np.ndarray:
    """
    Compute the diffusion metric tensor G (N x k x k) where:

        G[i, k, l] = Γ(φₖ, φₗ)(i)

    and the global Riemannian metric M = Σᵢ G[i] G[i]ᵀ  (k x k).

    For large k, pass `pairs` to compute only the entries you need.

    Returns
    -------
    G : (N, k, k)  pointwise metric tensor (symmetric)
    M : (k, k)     global Riemannian metric
    """
    k = eigenvalues.shape[0]
    N = eigenvectors.shape[0]
    G = np.zeros((N, k, k), dtype=np.float64)

    if pairs is None:
        pairs = [(i, j) for i in range(k) for j in range(i, k)]

    for (ki, li) in pairs:
        gamma = carre_du_champ(
            dm,
            eigenvectors[:, ki], eigenvectors[:, li],
            eigenvalues[ki],     eigenvalues[li],
        )
        G[:, ki, li] = gamma
        G[:, li, ki] = gamma

    # global metric: M_kl = Σᵢ G[i,k,:] · G[i,l,:]  — batched einsum
    M = np.einsum('ikm,ilm->kl', G, G)
    return G, M
