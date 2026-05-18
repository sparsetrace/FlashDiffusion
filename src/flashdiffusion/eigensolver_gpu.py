"""
flashdiffusion/eigensolver_gpu.py
==================================
GPU-resident Lanczos eigensolver for the diffusion map operator.

v1: Plain Lanczos with full re-orthogonalization (double MGS).
    No restart — simpler, proven correct, already 10× faster than ARPACK
    for our case because vectors stay on GPU with zero CPU transfer.

v2 (future): Thick-restart Lanczos for even fewer matvecs.

Algorithm
---------
1. Run n_iter Lanczos steps, all vectors on GPU
2. Full double-pass MGS re-orthogonalization per step
3. Solve small (n_iter × n_iter) tridiagonal on GPU via torch.linalg.eigh
4. Return top-k Ritz pairs

Why this beats ARPACK for fast GPU kernels
------------------------------------------
ARPACK runs on CPU and transfers v to GPU + back each iteration:
  overhead per iter ≈ 2.35s  (B200, N=200k)
  50 iters × 2.35s = 117s overhead
  
GPU Lanczos: zero transfer, all computation on GPU:
  50 iters × 0.0s transfer = 0s overhead
  total ≈ 50 × 0.23s matvec + 50 × O(N×j) ortho ≈ 12s

References
----------
Golub & Van Loan, "Matrix Computations" 4th ed., Ch. 10
Parlett, "The Symmetric Eigenvalue Problem", SIAM 1998
"""

from __future__ import annotations
import time
import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def gpu_lanczos(
    matvec_fn,
    N:        int,
    k:        int   = 8,
    n_iter:   int   = None,
    tol:      float = 1e-6,
    dtype           = None,
    device:   str   = "cuda",
    verbose:  bool  = True,
    q0=None
) -> tuple:
    """
    Plain GPU Lanczos with full re-orthogonalization for top-k eigenvalues.

    Parameters
    ----------
    matvec_fn : callable
        (N,) float32 GPU tensor → (N,) float32 GPU tensor.
        Zero CPU transfer — vectors stay on GPU throughout.
    N : int
        Problem size.
    k : int
        Number of eigenvalues/vectors wanted (default 8).
    n_iter : int
        Number of Lanczos steps. Default: min(3*k + 20, N).
        More iterations → better accuracy for clustered spectra.
        Memory cost: N × n_iter × 4 bytes
        (N=200k, n_iter=60: 200k × 60 × 4 = 48MB — trivial on any GPU)
    tol : float
        Not used in plain Lanczos (kept for API compatibility with ARPACK).
    dtype : torch.dtype
        Default float32. Use float64 for better orthogonality (2× memory).
    device : str
        CUDA device (default 'cuda').
    verbose : bool
        Print iteration info.

    Returns
    -------
    vals : np.ndarray, shape (k,)
        Top-k eigenvalues, descending order.
    vecs : np.ndarray, shape (N, k)
        Corresponding Ritz vectors (approximate eigenvectors).
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch required for gpu_lanczos")

    import torch

    if dtype is None:
        dtype = torch.float32

    if n_iter is None:
        #n_iter = min(6 * k + 40, N)   # 88 for k=8 — converges for typical DMAP spectra
        n_iter = min(6 * k, N)   # 48 for k=8 with warm start


    t_start = time.perf_counter()
    if verbose:
        print(f"[gpu_lanczos] N={N}  k={k}  n_iter={n_iter}  dtype={dtype}")

    # ── allocate Lanczos basis on GPU ────────────────────────────────────────
    # Q: (N, n_iter) — all basis vectors, needed for re-orthogonalization
    # Memory: N=200k, n_iter=60, fp32 → 200k × 60 × 4 = 48MB
    Q      = torch.zeros(N, n_iter, dtype=dtype, device=device)
    alphas = torch.zeros(n_iter,    dtype=dtype, device=device)
    betas  = torch.zeros(n_iter,    dtype=dtype, device=device)

    # random normalised start vector
    torch.manual_seed(42)
    # correct warm start (exact leading eigenvector of P):
    # pass D_alpha from state as optional q0
    if q0 is not None:
        q = q0.to(dtype=dtype)
    else:
        q = torch.ones(N, dtype=dtype, device=device)
    q /= q.norm()

    Q[:, 0] = q

    j_final = n_iter   # actual steps taken (may terminate early)

    for j in range(n_iter):
        # ── matvec — stays on GPU, zero transfer ─────────────────────────
        z = matvec_fn(Q[:, j])

        # ── three-term recurrence ─────────────────────────────────────────
        alphas[j] = torch.dot(Q[:, j], z)
        z = z - alphas[j] * Q[:, j]
        if j > 0:
            z = z - betas[j - 1] * Q[:, j - 1]

        # ── full re-orthogonalization (double MGS) ────────────────────────
        # Critical for clustered eigenvalues — single pass loses orthogonality
        # Cost: 2 × j × N flops per step — tiny vs matvec cost O(N²)
        if j > 0:
            for _ in range(2):
                # (j+1, N) × (N,) → (j+1,) coefficients
                coeffs = Q[:, :j + 1].T @ z
                z      = z - Q[:, :j + 1] @ coeffs

        # ── next basis vector ─────────────────────────────────────────────
        beta_j  = z.norm().item()
        betas[j] = beta_j

        if j < n_iter - 1:
            if beta_j < 1e-12:
                # lucky breakdown — Krylov space exhausted
                j_final = j + 1
                if verbose:
                    print(f"  lucky breakdown at j={j}, β={beta_j:.2e}")
                break
            Q[:, j + 1] = z / beta_j

    # ── solve tridiagonal eigenvalue problem ──────────────────────────────────
    # T is (j_final × j_final) — tiny, solve on GPU
    m = j_final
    T = torch.diag(alphas[:m])
    if m > 1:
        T += torch.diag(betas[:m - 1], 1)
        T += torch.diag(betas[:m - 1], -1)

    # use numpy for robustness — T is tiny (60×60), CPU cost negligible
    T_np = T.cpu().numpy().astype(np.float64)
    evals_np, evecs_np = np.linalg.eigh(T_np)   # ascending, numpy is robust

    # sort descending, take top-k
    idx      = np.argsort(evals_np)[::-1][:k]
    evals_k  = evals_np[idx]
    evecs_k  = evecs_np[:, idx]   # (m, k)

    # ── Ritz vectors: Q[:, :m] @ evecs_k — GPU matmul ────────────────────────
    # (N, m) × (m, k) → (N, k)
    evecs_k_gpu = torch.from_numpy(evecs_k.astype(np.float32)).to(device)
    ritz_vecs   = Q[:, :m] @ evecs_k_gpu   # (N, k)

    elapsed = time.perf_counter() - t_start
    if verbose:
        print(f"[gpu_lanczos] {m} matvecs  time={elapsed:.2f}s")
        print(f"  eigenvalues: {evals_k}")

    return evals_k.astype(np.float64), ritz_vecs.cpu().numpy()


def make_gpu_matvec(state, ext_module):
    """
    Wrap a CUDAState + ext module into a pure GPU→GPU matvec callable.
    No numpy, no CPU transfer — v stays on GPU throughout.

    Parameters
    ----------
    state : CUDAState / CUDAStateSM100 / CUDAStateSM120 / etc.
        Precomputed GPU state with X_fp16, sq_X, rscale, beta.
    ext_module : compiled CUDA extension
        Must expose .matvec(X_fp16, v, rscale, sq_X, beta) → GPU tensor.

    Returns
    -------
    callable : (N,) float32 GPU tensor → (N,) float32 GPU tensor
    """
    import torch

    def matvec_gpu(v: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return ext_module.matvec(
                state.X_fp16,
                v,
                state.rscale,
                state.sq_X,
                state.beta,
            )
    return matvec_gpu
