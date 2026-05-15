"""
flashdiffusion/eigensolver.py
=============================
Mixed-precision Krylov eigensolver for the FlashDiffusion operator.

ARPACK (via scipy.sparse.linalg.eigsh) handles the Lanczos recurrence
internally. It calls our matvec() on each iteration and manages
orthogonalisation in float64. Convergence for diffusion maps is typically
fast because:
  - The spectrum is bounded in [0, 1] with a clear gap after k eigenvectors
  - ARPACK uses implicit restarting (IRLM) with a Krylov space much larger
    than k, achieving near-optimal convergence per matvec
  - The trivial λ=1 eigenvector is found immediately

Precision ramp
--------------
  matvec arithmetic : float32 GEMM (dist2) + float64 exp + float64 accumulate
  Krylov basis      : float64 (ARPACK internal)
  orthogonalisation : float64 QR (ARPACK internal, O(N k²) per restart)
  eigenvalue output : float64

This is the same split as FA4: tensor-core GEMM in low precision,
correction/accumulation in high precision.
"""

from __future__ import annotations
import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator
from typing import Callable, Optional, Tuple
import time


def lanczos_eigsh(
    matvec:  Callable[[np.ndarray], np.ndarray],
    N:       int,
    k:       int       = 8,
    tol:     float     = 0.0,
    maxiter: Optional[int] = None,
    ncv:     Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute top-k eigenpairs via ARPACK Lanczos (implicitly restarted).

    Parameters
    ----------
    matvec  : callable (N,) -> (N,), applies the symmetric operator M
    N       : problem dimension
    k       : number of eigenpairs
    tol     : ARPACK convergence tolerance (0 = machine eps)
    maxiter : max ARPACK iterations (default: 10 * N)
    ncv     : Krylov space size (default: min(max(2k+1, 20), N))
              Larger ncv -> faster convergence, more memory

    Returns
    -------
    eigenvalues  : (k,) float64, descending
    eigenvectors : (N, k) float64
    """
    if ncv is None:
        ncv = min(max(2 * k + 1, 20), N)

    op = LinearOperator(
        shape  = (N, N),
        matvec = lambda v: matvec(np.asarray(v, dtype=np.float64).ravel()),
        dtype  = np.float64,
    )

    t0 = time.perf_counter()
    vals, vecs = eigsh(
        op,
        k       = k,
        which   = 'LM',
        tol     = tol,
        maxiter = maxiter,
        ncv     = ncv,
    )
    dt = time.perf_counter() - t0

    idx  = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]

    print(f"[lanczos_eigsh] k={k}  N={N}  time={dt:.3f}s")
    print(f"  eigenvalues: {vals}")
    return vals, vecs


def lobpcg_eigsh(
    matvec:  Callable[[np.ndarray], np.ndarray],
    N:       int,
    k:       int   = 8,
    tol:     float = 1e-8,
    maxiter: int   = 300,
    seed:    int   = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute top-k eigenpairs via LOBPCG.

    LOBPCG carries the previous search direction (conjugate gradient step),
    giving superlinear convergence for clustered spectra. Preferred over
    Lanczos when k is large relative to the spectral gap.

    orthogonalisation: float64 QR inside the k x k projected problem.
    Each outer iteration: one matvec + O(N k²) ortho (cheap).
    """
    from scipy.sparse.linalg import lobpcg

    rng = np.random.default_rng(seed)
    # float64 initial subspace, orthonormalised
    X0 = np.linalg.qr(rng.standard_normal((N, k)))[0]

    op = LinearOperator(
        shape  = (N, N),
        matvec = lambda v: matvec(np.asarray(v, dtype=np.float64).ravel()),
        dtype  = np.float64,
    )

    t0 = time.perf_counter()
    vals, vecs = lobpcg(
        op,
        X0,
        tol     = tol,
        maxiter = maxiter,
        largest = True,
    )
    dt = time.perf_counter() - t0

    idx  = np.argsort(vals)[::-1]
    print(f"[lobpcg_eigsh] k={k}  N={N}  time={dt:.3f}s")
    print(f"  eigenvalues: {vals[idx]}")
    return vals[idx], vecs[:, idx]
