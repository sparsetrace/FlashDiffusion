from __future__ import annotations

import time
import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def gpu_lanczos_thick_restart(
    matvec_fn,
    N: int,
    k: int = 8,
    ncv: int | None = None,
    keep: int | None = None,
    max_restarts: int = 20,
    tol: float = 1e-6,
    dtype=None,
    device: str = "cuda",
    verbose: bool = True,
    q0=None,
    breakdown_tol: float = 1e-12,
    relative_residual: bool = True,
    return_info: bool = False,
    return_torch: bool = False,
):
    """
    Thick-restart GPU Lanczos / symmetric Krylov-Schur style eigensolver.

    Computes the largest algebraic Ritz pairs of a symmetric/self-adjoint
    linear operator using only GPU-resident matvecs.

    This is intended for diffusion-map operators that are symmetric or have
    already been conjugated/projected into a symmetric form.

    Parameters
    ----------
    matvec_fn : callable
        GPU tensor -> GPU tensor matvec.
        Input and output should be length-N torch tensors on `device`.

    N : int
        Problem size.

    k : int
        Number of eigenpairs requested.

    ncv : int or None
        Maximum Krylov subspace size per restart cycle.
        Similar to ARPACK's ncv.
        Must be > keep + 1.
        Default: min(N, max(4*k + 40, k + 20)).

    keep : int or None
        Number of Ritz vectors retained at restart.
        Must satisfy k <= keep <= ncv - 2.
        Default: min(ncv - 2, k + max(8, k // 2)).

    max_restarts : int
        Maximum restart cycles.

    tol : float
        Ritz residual tolerance.

    dtype : torch.dtype
        Default torch.float32.

    device : str
        CUDA device.

    verbose : bool
        Print progress.

    q0 : torch.Tensor or None
        Optional initial vector. Should not be an exact eigenvector unless
        you intentionally want a one-dimensional Krylov space.

    breakdown_tol : float
        Treat beta below this as lucky breakdown.

    relative_residual : bool
        If True, convergence criterion is

            residual <= tol * max(1, abs(theta))

        Otherwise,

            residual <= tol

    return_info : bool
        If True, return a diagnostics dictionary.

    return_torch : bool
        If True, return eigenvectors as GPU torch tensors.
        If False, return eigenvectors as NumPy arrays on CPU.

    Returns
    -------
    vals : np.ndarray
        Eigenvalues in descending order.

    vecs : np.ndarray or torch.Tensor
        Ritz vectors, shape (N, k).

    info : dict, optional
        Returned only if return_info=True.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch required for gpu_lanczos_thick_restart")

    import torch

    if dtype is None:
        dtype = torch.float32

    if k < 1:
        raise ValueError("k must be >= 1")

    if ncv is None:
        ncv = min(N, max(4 * k + 40, k + 20))
    ncv = int(min(ncv, N))

    if ncv <= k + 1:
        raise ValueError(f"ncv={ncv} is too small for k={k}; need ncv > k + 1")

    if keep is None:
        keep = min(ncv - 2, k + max(8, k // 2))
    keep = int(keep)

    if keep < k:
        raise ValueError(f"keep={keep} must be >= k={k}")
    if keep > ncv - 2:
        raise ValueError(f"keep={keep} must be <= ncv - 2 = {ncv - 2}")

    t_start = time.perf_counter()

    if verbose:
        print(
            f"[trlanczos] N={N}  k={k}  ncv={ncv}  keep={keep}  "
            f"max_restarts={max_restarts}  tol={tol:g}  dtype={dtype}"
        )

    def _make_initial_vector() -> torch.Tensor:
        if q0 is not None:
            q = q0.to(device=device, dtype=dtype).reshape(-1)
            if q.numel() != N:
                raise ValueError(f"q0 has length {q.numel()}, expected N={N}")
        else:
            q = torch.randn(N, device=device, dtype=dtype)

        qn = q.norm()
        if float(qn.detach().cpu()) == 0.0:
            raise ValueError("Initial vector has zero norm")
        return q / qn

    def _project_out(Qbasis: torch.Tensor, v: torch.Tensor, passes: int = 2) -> torch.Tensor:
        """
        Orthogonalize v against columns of Qbasis.
        """
        if Qbasis.numel() == 0:
            return v

        for _ in range(passes):
            coeffs = Qbasis.T @ v
            v = v - Qbasis @ coeffs
        return v

    def _solve_small(H_small: torch.Tensor):
        """
        Solve the small symmetric projected eigenproblem on CPU.
        H_small is tiny, so this is robust and cheap.
        Returns eigenvalues descending and eigenvectors in matching order.
        """
        H_np = H_small.detach().cpu().numpy().astype(np.float64)
        H_np = 0.5 * (H_np + H_np.T)  # safety
        evals, evecs = np.linalg.eigh(H_np)  # ascending

        idx = np.argsort(evals)[::-1]
        return evals[idx], evecs[:, idx]

    def _expand_from_initial_basis(
        Q_init: torch.Tensor,
        H_init: torch.Tensor | None,
        start_col: int,
        target_dim: int,
    ):
        """
        Expand an existing orthonormal basis up to target_dim columns.

        Q_init has shape (N, s), where s = start_col + 1.
        We do not matvec the retained Ritz vectors before start_col.
        Expansion starts from Q[:, start_col].

        Returns
        -------
        Q : torch.Tensor, shape (N, ncv + 1)
            Basis columns 0..m-1 plus residual direction at column m
            when available.

        H : torch.Tensor, shape (ncv, ncv)
            Projected symmetric matrix.

        m : int
            Number of basis vectors in the current Ritz problem.

        beta_last : float
            Norm of final residual vector.

        has_residual_vector : bool
            Whether Q[:, m] contains a valid next direction.
        """
        s = Q_init.shape[1]
        if s != start_col + 1:
            raise ValueError("Q_init.shape[1] must equal start_col + 1")

        Q = torch.zeros(N, target_dim + 1, device=device, dtype=dtype)
        H = torch.zeros(target_dim, target_dim, device=device, dtype=dtype)

        Q[:, :s] = Q_init

        if H_init is not None:
            hs = H_init.shape[0]
            H[:hs, :hs] = H_init.to(device=device, dtype=dtype)

        beta_last = 0.0
        has_residual_vector = False
        m = s

        for j in range(start_col, target_dim):
            z = matvec_fn(Q[:, j])
            z = z.to(device=device, dtype=dtype).reshape(-1)

            # Full orthogonalization against all current basis vectors.
            # We accumulate h so H[:, j] approximates Q.T A q_j.
            Qj = Q[:, : j + 1]

            h = Qj.T @ z
            z = z - Qj @ h

            # Second pass MGS, important for clustered spectra.
            h2 = Qj.T @ z
            z = z - Qj @ h2
            h = h + h2

            H[: j + 1, j] = h
            H[j, : j + 1] = h

            beta_tensor = z.norm()
            beta_last = float(beta_tensor.detach().cpu())
            m = j + 1

            if beta_last < breakdown_tol:
                has_residual_vector = False
                return Q, H, m, beta_last, has_residual_vector

            # Store next residual direction. Even when j == target_dim - 1,
            # this gives us Q[:, m] for the next thick restart.
            Q[:, j + 1] = z / beta_tensor
            has_residual_vector = True

        return Q, H, m, beta_last, has_residual_vector

    # Initial cycle starts from one vector.
    q = _make_initial_vector()
    Q_init = q[:, None]
    H_init = None
    start_col = 0

    best_evals = None
    best_evecs_small = None
    best_Q = None
    best_m = None
    best_beta = None
    best_residuals = None

    converged = False
    breakdown = False
    n_matvec = 0
    cycles_done = 0

    for cycle in range(max_restarts + 1):
        cycles_done = cycle + 1

        Q, H, m, beta_last, has_residual_vector = _expand_from_initial_basis(
            Q_init=Q_init,
            H_init=H_init,
            start_col=start_col,
            target_dim=ncv,
        )

        # Number of matvecs this cycle:
        # initial cycle expands cols 0..ncv-1 => ncv matvecs.
        # restarted cycle expands cols keep..ncv-1 => ncv-keep matvecs.
        n_matvec += ncv - start_col

        Hm = H[:m, :m]
        evals_all, evecs_all = _solve_small(Hm)

        n_take = min(k, len(evals_all))
        evals_k = evals_all[:n_take]
        evecs_k = evecs_all[:, :n_take]

        # For this thick-restart construction, after expansion:
        #
        #     A Q_m = Q_m H_m + beta_last q_next e_m^T
        #
        # Hence Ritz residual_i ~= |beta_last * y_i[-1]|.
        residuals = np.abs(beta_last * evecs_k[-1, :])

        if relative_residual:
            thresholds = tol * np.maximum(1.0, np.abs(evals_k))
        else:
            thresholds = np.full_like(residuals, tol, dtype=np.float64)

        n_conv = int(np.sum(residuals <= thresholds))

        best_evals = evals_all
        best_evecs_small = evecs_all
        best_Q = Q
        best_m = m
        best_beta = beta_last
        best_residuals = residuals

        if verbose:
            max_res = float(np.max(residuals)) if residuals.size else float("nan")
            print(
                f"  cycle={cycle:02d}  m={m:4d}  beta={beta_last:.3e}  "
                f"converged={n_conv}/{n_take}  max_res={max_res:.3e}"
            )
            print(f"    theta[:min(5,k)] = {evals_k[:min(5, len(evals_k))]}")

        if n_take >= k and np.all(residuals[:k] <= thresholds[:k]):
            converged = True
            if verbose:
                print(f"[trlanczos] converged after cycle={cycle}")
            break

        if beta_last < breakdown_tol:
            breakdown = True
            if verbose:
                print(f"[trlanczos] lucky breakdown after cycle={cycle}")
            break

        if cycle == max_restarts:
            break

        if not has_residual_vector:
            breakdown = True
            if verbose:
                print("[trlanczos] no residual vector available for restart")
            break

        # ------------------------------------------------------------------
        # Thick restart:
        # Keep the leading `keep` Ritz vectors and the residual direction.
        # ------------------------------------------------------------------
        keep_eff = min(keep, m)
        theta_keep = evals_all[:keep_eff]
        Y_keep = evecs_all[:, :keep_eff]

        Y_keep_gpu = torch.from_numpy(Y_keep).to(device=device, dtype=dtype)

        # U = Q_m @ Y_keep, retained Ritz vectors.
        U = Q[:, :m] @ Y_keep_gpu

        # In exact arithmetic U is orthonormal. In fp32 it should still be
        # close because Q was fully reorthogonalized and Y is orthonormal.
        #
        # Avoid QR here because changing the retained basis would require
        # transforming the small retained H block and residual couplings.
        # Instead, just normalize columns defensively.
        U_norms = torch.linalg.norm(U, dim=0)
        U = U / U_norms.clamp_min(torch.finfo(dtype).eps)

        # Residual direction q_next = Q[:, m].
        q_next = Q[:, m].clone()

        # Re-orthogonalize residual direction against retained Ritz vectors.
        q_next = _project_out(U, q_next, passes=2)
        qn = q_next.norm()

        if float(qn.detach().cpu()) < breakdown_tol:
            # Fall back to a fresh random direction if the residual collapsed.
            q_next = torch.randn(N, device=device, dtype=dtype)
            q_next = _project_out(U, q_next, passes=2)
            qn = q_next.norm()

        q_next = q_next / qn

        # Build initial basis for the next cycle:
        # columns 0..keep_eff-1 are retained Ritz vectors,
        # column keep_eff is the residual direction.
        Q_init = torch.empty(N, keep_eff + 1, device=device, dtype=dtype)
        Q_init[:, :keep_eff] = U
        Q_init[:, keep_eff] = q_next

        # Initial projected matrix.
        #
        # For retained Ritz vectors:
        #   A U_i = theta_i U_i + beta * y_i[-1] q_next
        #
        # The coupling vector is g_i = beta_last * y_i[-1].
        # The column for q_next will be recomputed on the next expansion,
        # but setting the arrow entries here preserves the thick-restart
        # relation before expansion.
        H_init = torch.zeros(keep_eff + 1, keep_eff + 1, device=device, dtype=dtype)
        H_init[:keep_eff, :keep_eff] = torch.diag(
            torch.from_numpy(theta_keep).to(device=device, dtype=dtype)
        )

        g_keep = beta_last * Y_keep[-1, :]
        g_keep_gpu = torch.from_numpy(g_keep).to(device=device, dtype=dtype)

        H_init[:keep_eff, keep_eff] = g_keep_gpu
        H_init[keep_eff, :keep_eff] = g_keep_gpu

        # Next cycle starts expanding from the residual vector.
        start_col = keep_eff

    if best_evals is None or best_evecs_small is None or best_Q is None:
        raise RuntimeError("Lanczos failed before producing a Ritz problem")

    n_return = min(k, len(best_evals), best_m)

    vals = best_evals[:n_return]
    Y = best_evecs_small[:, :n_return]

    Y_gpu = torch.from_numpy(Y).to(device=device, dtype=dtype)
    ritz_vecs = best_Q[:, :best_m] @ Y_gpu

    # Final defensive normalization.
    ritz_vecs = ritz_vecs / torch.linalg.norm(ritz_vecs, dim=0).clamp_min(
        torch.finfo(dtype).eps
    )

    elapsed = time.perf_counter() - t_start

    if verbose:
        print(
            f"[trlanczos] finished  converged={converged}  breakdown={breakdown}  "
            f"cycles={cycles_done}  matvecs≈{n_matvec}  time={elapsed:.2f}s"
        )
        print(f"  eigenvalues: {vals}")
        if best_residuals is not None:
            print(f"  residuals:   {best_residuals[:n_return]}")

    vals_out = vals.astype(np.float64)

    if return_torch:
        vecs_out = ritz_vecs
    else:
        vecs_out = ritz_vecs.detach().cpu().numpy()

    if return_info:
        info = {
            "converged": bool(converged),
            "breakdown": bool(breakdown),
            "cycles": int(cycles_done),
            "n_matvec": int(n_matvec),
            "ncv": int(ncv),
            "keep": int(keep),
            "k": int(k),
            "tol": float(tol),
            "relative_residual": bool(relative_residual),
            "residuals": None
            if best_residuals is None
            else best_residuals[:n_return].astype(np.float64),
            "elapsed": float(elapsed),
            "m_final": int(best_m),
            "beta_final": float(best_beta),
        }
        return vals_out, vecs_out, info

    return vals_out, vecs_out
