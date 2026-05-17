"""
flashdiffusion/dmap.py
======================
DiffusionMap class: Coifman-Lafon diffusion maps via FlashDiffusion kernel.

Usage
-----
    dm = DiffusionMap(beta=1.0, alpha=0.5, n_components=8)
    dm.fit(X)               # precompute rscale (two kernel passes, O(N·tile) mem)
    coords = dm.transform() # Lanczos eigensolver -> diffusion coordinates
    new    = dm.transform(X_new)  # Nyström out-of-sample extension

Relationship to DAC paper
--------------------------
With Q=K=X and symmetric W (W=Wᵀ), this is the canonical EQ Schrödinger
bridge (Proposition 5.2). The operator M = D_α^{-1/2} K_α D_α^{-1/2} is
the symmetrised DMAP forward operator. Its eigenvectors give the diffusion
map embedding.

The Coifman-Lafon α-normalisation is an exact Doob h-transform
(Corollary 4.2, DAC): h = D^{-α}, so the EQ class is preserved.
"""

from __future__ import annotations
import numpy as np
from typing import Optional
import time

from .kernel import FlashDiffusion, prepass_rowsum


class DiffusionMap:
    """
    Coifman-Lafon diffusion maps.

    Parameters
    ----------
    beta : float
        Kernel bandwidth: K(i,j) = exp(-beta · ||xi - xj||²).
        Use utils.median_bandwidth(X) for automatic selection.
    alpha : float
        Density normalisation exponent (0 = no correction, 0.5 = default,
        1.0 = Fokker-Planck normalisation).
    n_components : int
        Number of eigenvectors to compute (including trivial λ=1).
    tile : int
        Tile size for the kernel. Controls peak memory: O(N · tile · d · 4 bytes).
    """

    def __init__(
        self,
        beta:          float = 1.0,
        alpha:         float = 0.5,
        n_components:  int   = 8,
        tile:          int   = 512,
    ):
        self.beta         = beta
        self.alpha        = alpha
        self.n_components = n_components
        self.tile         = tile

        # set after fit()
        self.X_:          Optional[np.ndarray] = None
        self.rscale_:     Optional[np.ndarray] = None
        self.D_:          Optional[np.ndarray] = None
        self.D_alpha_:    Optional[np.ndarray] = None
        self._cuda_state: Optional[object]     = None  # CUDAState if GPU used

        # set after transform()
        self.eigenvalues_:  Optional[np.ndarray] = None
        self.eigenvectors_: Optional[np.ndarray] = None
        self.embedding_:    Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # fit: two prepass reductions, O(N·tile) memory
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "DiffusionMap":
        """
        Precompute Coifman-Lafon normalisation for dataset X.

        Uses CUDA backend (two GPU prepass kernels) if available,
        otherwise numpy CPU fallback. Either way the interface is identical.
        """
        N = X.shape[0]
        self.X_ = X.astype(np.float64)
        beta, alpha, tile = self.beta, self.alpha, self.tile

        print(f"[DiffusionMap.fit] N={N}  beta={beta}  alpha={alpha}")
        t0 = time.perf_counter()

        # Try CUDA first
        from .kernel import _get_cuda_backend
        cuda = _get_cuda_backend()
        if cuda is not None:
            precompute_fn, matvec_fn = cuda
            self._cuda_state = precompute_fn(X, beta, alpha)
            self._matvec_fn  = matvec_fn
            # Mirror numpy fields from GPU tensors for CDC / Doob / Nyström
            self.D_       = self._cuda_state.D.cpu().numpy().astype(np.float64)
            self.D_alpha_ = self._cuda_state.D_alpha.cpu().numpy().astype(np.float64)
            self.rscale_  = self._cuda_state.rscale.cpu().numpy().astype(np.float64)
            print(f"  precompute done (CUDA) in {time.perf_counter()-t0:.3f}s")
            return self

        # Numpy fallback
        self._cuda_state = None
        D = prepass_rowsum(X, X, beta,
                           weights_k=np.ones(N, np.float64), tile=tile)
        self.D_ = D
        w = D ** (-alpha)
        D_alpha = prepass_rowsum(X, X, beta, weights_k=w, tile=tile) * w
        self.D_alpha_ = D_alpha
        d_half = D_alpha ** (-0.5)
        self.rscale_ = d_half * w
        print(f"  precompute done (numpy) in {time.perf_counter()-t0:.3f}s")
        return self

    def matvec(self, v: np.ndarray) -> np.ndarray:
        """
        Apply M to v. Uses CUDA kernel if available, numpy otherwise.
        Called every Lanczos iteration — v is the only thing that changes.
        """
        self._check_fitted()

        if self._cuda_state is not None:
            from .kernel_cuda import matvec_cuda
            return matvec_cuda(self._cuda_state, v)

        # numpy path
        v2d = v.reshape(-1, 1)
        out = FlashDiffusion(
            Q=self.X_, K=self.X_, V=v2d, beta=self.beta,
            rscale_q=self.rscale_, rscale_k=self.rscale_,
            tile=self.tile,
        )
        return out.ravel()

    # ------------------------------------------------------------------
    # transform: Lanczos eigensolver -> diffusion coordinates
    # ------------------------------------------------------------------

    def transform(
        self,
        X_new:    Optional[np.ndarray] = None,
        t:        int   = 1,
        tol:      float = 0.0,
        maxiter:  Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute diffusion map coordinates.

        If X_new is None: eigensolver on training set (fit must be called first).
        If X_new is given: Nyström out-of-sample extension (see utils.nystrom).

        Parameters
        ----------
        X_new   : (M, d) new points, or None for training set
        t       : diffusion time (eigenvalues raised to power t)
        tol     : ARPACK tolerance (0 = machine precision)
        maxiter : ARPACK max iterations

        Returns
        -------
        coords : (N, n_components-1) — drops the trivial first eigenvector
        """
        self._check_fitted()

        if X_new is not None:
            from .utils import nystrom_extend
            return nystrom_extend(self, X_new, t=t)

        from .eigensolver import lanczos_eigsh
        vals, vecs = lanczos_eigsh(
            self.matvec,
            N       = self.X_.shape[0],
            k       = self.n_components,
            tol     = tol,
            maxiter = maxiter,
        )
        self.eigenvalues_  = vals
        self.eigenvectors_ = vecs

        # convert symmetric-M eigenvectors -> row-normalised P+ eigenvectors
        psi = vecs / self.rscale_[:, None]
        psi /= np.linalg.norm(psi, axis=0, keepdims=True)

        # diffusion map embedding: scale by λ^t, drop trivial first vector
        coords = psi[:, 1:] * (vals[1:] ** t)[None, :]
        self.embedding_ = coords
        return coords

    # ------------------------------------------------------------------
    # Doob h-transform: returns a new DiffusionMap-like matvec
    # ------------------------------------------------------------------

    def doob_transform(self, h: np.ndarray) -> "DoobTransformedMap":
        """
        Apply a Doob h-transform to the current operator.

        By Theorem 5.1 (DAC): if the current operator is EQ (symmetric K),
        the transformed operator is also EQ. The new stationary distribution
        is π_h ∝ π · h · (P+h).

        Returns a DoobTransformedMap with a matvec() method compatible
        with eigensolver.py.
        """
        self._check_fitted()
        return DoobTransformedMap(self, h)

    # ------------------------------------------------------------------

    def _check_fitted(self):
        if self.rscale_ is None:
            raise RuntimeError("Call fit(X) before transform() or matvec().")


# ---------------------------------------------------------------------------
# Doob-transformed operator (wraps DiffusionMap with an h-potential)
# ---------------------------------------------------------------------------

class DoobTransformedMap:
    """
    Doob h-transform of a DiffusionMap operator.

    (M_h v)_i = rscale_i · Σⱼ K(i,j) · rscale_j · h_j · v_j
               / (P+ h)_i                         (row-renormalisation)

    By Theorem 5.1 (DAC), this preserves the EQ class when K is symmetric.
    The new rscale absorbs h into the kernel normalisation.
    """

    def __init__(self, base: DiffusionMap, h: np.ndarray):
        self.base = base
        self.h    = np.asarray(h, dtype=np.float64)

        # precompute (P+ h)_i = rscale_i * Σⱼ K(i,j) * rscale_j * h_j
        # = one FlashDiffusion call with V=h
        Ph = FlashDiffusion(
            Q        = base.X_,
            K        = base.X_,
            V        = self.h[:, None],
            beta     = base.beta,
            rscale_q = base.rscale_,
            rscale_k = base.rscale_,
            tile     = base.tile,
        ).ravel()
        self.Ph_ = Ph                      # normaliser for the Doob transform

    def matvec(self, v: np.ndarray) -> np.ndarray:
        """Apply Doob-transformed operator to v."""
        h   = self.h
        out = FlashDiffusion(
            Q        = self.base.X_,
            K        = self.base.X_,
            V        = (h * v)[:, None],
            beta     = self.base.beta,
            rscale_q = self.base.rscale_,
            rscale_k = self.base.rscale_,
            tile     = self.base.tile,
        ).ravel()
        return out / self.Ph_              # row-renormalise
