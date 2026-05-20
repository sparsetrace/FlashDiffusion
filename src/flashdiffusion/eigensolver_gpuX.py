from __future__ import annotations

import time
import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def gpu_irlm_dense_qr(
    matvec_fn,
    N: int,
    k: int = 8,
    ncv: int | None = None,
    keep: int | None = None,
    max_restarts: int = 20,
    tol: float = 1e-6,
    dtype=None,
    small_dtype=None,
    device: str = "cuda",
    verbose: bool = True,
    q0=None,
    breakdown_tol: float = 1e-12,
    relative_residual: bool = True,
    reorthogonalize: bool = True,
    reorth_passes: int = 2,
    restart_residual_mode: str = "explicit",
    return_info: bool = False,
    return_torch: bool = False,
):
    """
    GPU-resident implicitly restarted Lanczos method using dense shifted QR
    on the small projected tridiagonal matrix.

    Computes the largest algebraic eigenpairs of a symmetric/self-adjoint
    matrix-free operator.

    Parameters
    ----------
    matvec_fn : callable
        GPU tensor -> GPU tensor matvec.
        Input and output should be length-N torch tensors on `device`.

    N : int
        Problem size.

    k : int
        Number of requested eigenpairs.

    ncv : int or None
        Krylov subspace size. Larger improves convergence but stores more
        vectors. Default: min(N, max(4*k + 40, k + 20)).

    keep : int or None
        Number of Ritz/Lanczos directions kept after each implicit restart.
        Must satisfy k <= keep <= ncv - 2.
        Default: min(ncv - 2, k + max(8, k // 2)).

    max_restarts : int
        Maximum number of implicit restart cycles.

    tol : float
        Residual tolerance.

    dtype : torch.dtype or None
        GPU vector dtype. Default: torch.float32.

    small_dtype : torch.dtype or None
        Dtype for small ncv x ncv projected problems. Default: torch.float64.
        Stays on GPU; float64 here is cheap and dramatically improves
        numerical quality of the tridiagonal eigensolve.

    device : str
        CUDA device string.

    verbose : bool
        Print convergence progress.

    q0 : torch.Tensor or None
        Optional initial vector. A small random perturbation is added
        automatically to prevent a one-dimensional Krylov space when q0
        is close to an exact eigenvector (e.g. a warm start from sqrt(D_alpha)).

    breakdown_tol : float
        Treat beta below this as lucky breakdown.

    relative_residual : bool
        If True, convergence criterion is
            residual_i <= tol * max(1, |theta_i|)
        Otherwise plain absolute tolerance.

    reorthogonalize : bool
        Full re-orthogonalization against all current basis vectors.
        Should be True for clustered spectra (diffusion maps).

    reorth_passes : int
        Number of MGS passes per orthogonalization step. 2 is standard.

    restart_residual_mode : str
        "explicit" (default, safer): one extra matvec per restart to
        recompute the residual exactly after the implicit QR rotation.
        "implicit": use the formal IRLM residual formula — faster but
        sensitive to off-tridiagonal fill from dense QR in float32.

    return_info : bool
        If True, return a diagnostics dictionary as third return value.

    return_torch : bool
        If True, return eigenvectors as GPU torch tensors.
        If False (default), return NumPy arrays on CPU.

    Returns
    -------
    vals : np.ndarray or torch.Tensor, shape (k,)
        Eigenvalues in descending order.

    vecs : np.ndarray or torch.Tensor, shape (N, k)
        Ritz vectors corresponding to vals.

    info : dict, optional
        Diagnostics. Only returned when return_info=True.

    Notes
    -----
    Bug fixes vs the original draft:
      1. Removed incorrect alpha update inside the reorthogonalization loop.
         dot(q_j, z_1) = 0 by construction so coeffs[-1] is zero in exact
         arithmetic; accumulating it added numerical noise without benefit.
      2. In implicit restart mode, the residual is now computed BEFORE
         _canonicalize_tridiag_and_basis flips basis signs, so the coupling
         coefficient c1 = T_shifted[keep_eff, keep_eff-1] is consistent with
         the basis used to form r. Sign flips are applied afterwards.
      3. q0 warm starts now receive a small random perturbation to prevent
         a one-dimensional Krylov space when q0 ≈ exact eigenvector.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch required for gpu_irlm_dense_qr")

    import torch

    if dtype is None:
        dtype = torch.float32
    if small_dtype is None:
        small_dtype = torch.float64

    device = torch.device(device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

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

    if restart_residual_mode not in {"explicit", "implicit"}:
        raise ValueError("restart_residual_mode must be 'explicit' or 'implicit'")

    grad_state = torch.is_grad_enabled()
    torch.set_grad_enabled(False)

    t_start = time.perf_counter()

    if verbose:
        print(
            f"[irlm] N={N}  k={k}  ncv={ncv}  keep={keep}  "
            f"max_restarts={max_restarts}  tol={tol:g}  "
            f"dtype={dtype}  small_dtype={small_dtype}  device={device}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _make_initial_vector() -> torch.Tensor:
        """
        Build a unit-norm starting vector.

        FIX 3: Add small random perturbation when q0 is provided to prevent
        a one-dimensional Krylov space if q0 is close to an exact eigenvector
        (e.g. sqrt(D_alpha) as a warm start for the leading diffusion mode).
        The perturbation scale is 1e-3 * ||q0|| / sqrt(N), negligible in norm
        but sufficient to seed all directions of the Krylov subspace.
        """
        if q0 is not None:
            q = torch.as_tensor(q0, device=device, dtype=dtype).reshape(-1)
            if q.numel() != N:
                raise ValueError(f"q0 has length {q.numel()}, expected N={N}")
            # Small perturbation — does not affect the dominant eigenvector
            # but prevents degenerate Krylov spaces.
            noise_scale = 1e-3 * float(torch.linalg.norm(q).detach().cpu()) / (N ** 0.5)
            q = q + noise_scale * torch.randn_like(q)
        else:
            q = torch.randn(N, device=device, dtype=dtype)

        qn = torch.linalg.norm(q)
        if float(qn.detach().cpu()) == 0.0:
            raise ValueError("Initial vector has zero norm")
        return q / qn

    def _orthogonalize(
        Qbasis: torch.Tensor,
        v: torch.Tensor,
        passes: int = 2,
    ) -> torch.Tensor:
        """Full re-orthogonalization of v against columns of Qbasis."""
        if Qbasis.numel() == 0:
            return v
        for _ in range(passes):
            coeffs = Qbasis.T @ v
            v = v - Qbasis @ coeffs
        return v

    def _build_T(
        alpha: torch.Tensor,
        beta: torch.Tensor,
        m: int,
    ) -> torch.Tensor:
        """Build symmetric tridiagonal m×m matrix from alpha/beta."""
        T = torch.diag(alpha[:m].to(device=device, dtype=small_dtype))
        if m > 1:
            b = beta[: m - 1].to(device=device, dtype=small_dtype)
            T = T + torch.diag(b, diagonal=1) + torch.diag(b, diagonal=-1)
        return 0.5 * (T + T.T)

    def _tridiag_part(T: torch.Tensor) -> torch.Tensor:
        """Extract the tridiagonal part of a matrix."""
        m = T.shape[0]
        out = torch.diag(torch.diagonal(T))
        if m > 1:
            up = torch.diagonal(T, offset=1)
            lo = torch.diagonal(T, offset=-1)
            off = 0.5 * (up + lo)
            out = out + torch.diag(off, diagonal=1) + torch.diag(off, diagonal=-1)
        return 0.5 * (out + out.T)

    def _off_tridiag_rel(T: torch.Tensor) -> float:
        """Relative off-tridiagonal fill — diagnostic for QR quality."""
        tri = _tridiag_part(T)
        off = T - tri
        denom = float(torch.linalg.norm(T).detach().cpu())
        if denom < 1e-300:
            return 0.0
        return float(torch.linalg.norm(off).detach().cpu()) / denom

    def _small_eigh_desc(T: torch.Tensor):
        """Eigendecomposition of small symmetric matrix, descending order."""
        T = 0.5 * (T + T.T)
        evals, evecs = torch.linalg.eigh(T)
        idx = torch.argsort(evals, descending=True)
        return evals[idx], evecs[:, idx]

    def _implicit_shifted_qr_dense(
        T: torch.Tensor,
        shifts: torch.Tensor,
    ):
        """
        Dense shifted QR steps applied to the small projected matrix.

        Each shift mu gives one QR step:
            T - mu*I = Q*R   =>   T_new = R*Q + mu*I

        The accumulated orthogonal transform Z satisfies T_new = Z^T T Z.

        Sign stabilization ensures Q has a canonical orientation,
        reducing jitter in repeated restarts.

        Returns
        -------
        T_work : torch.Tensor
            QR-transformed matrix (approximately block upper triangular).
        Z_total : torch.Tensor
            Accumulated orthogonal rotation, shape (m, m).
        """
        m = T.shape[0]
        I = torch.eye(m, device=device, dtype=small_dtype)
        T_work = T.clone().to(device=device, dtype=small_dtype)
        Z_total = torch.eye(m, device=device, dtype=small_dtype)

        for mu in shifts:
            Qs, Rs = torch.linalg.qr(T_work - mu * I, mode="reduced")

            # Stabilize sign convention: force positive diagonal on R.
            d = torch.diagonal(Rs)
            signs = torch.where(
                d < 0,
                torch.tensor(-1.0, device=device, dtype=small_dtype),
                torch.tensor( 1.0, device=device, dtype=small_dtype),
            )
            Qs = Qs * signs[None, :]
            Rs = signs[:, None] * Rs

            T_work = Rs @ Qs + mu * I
            T_work = 0.5 * (T_work + T_work.T)
            Z_total = Z_total @ Qs

        return T_work, Z_total

    def _canonicalize_tridiag_and_basis(
        Qkeep: torch.Tensor,
        Tkeep: torch.Tensor,
    ):
        """
        Convert retained small matrix to symmetric tridiagonal form and flip
        basis signs so all subdiagonal beta values are nonnegative.

        Returns updated (Qkeep, alpha_new, beta_new, Tkeep_tri).
        """
        Ttri = _tridiag_part(Tkeep).clone()
        r = Ttri.shape[0]

        for i in range(r - 1):
            val = float(Ttri[i + 1, i].detach().cpu())
            if val < 0:
                Qkeep[:, i + 1].mul_(-1)
                Ttri[i + 1, :].mul_(-1)
                Ttri[:, i + 1].mul_(-1)

        alpha_new = torch.diagonal(Ttri).clone().to(device=device, dtype=small_dtype)
        if r > 1:
            beta_new = torch.diagonal(Ttri, offset=-1).clone().abs().to(
                device=device, dtype=small_dtype
            )
        else:
            beta_new = torch.empty(0, device=device, dtype=small_dtype)

        return Qkeep, alpha_new, beta_new, Ttri

    # ──────────────────────────────────────────────────────────────────────
    # Persistent GPU state
    # Note: alpha, beta, Q are mutated in-place by _expand_from.
    # On exception they may be in a partially-written state.
    # ──────────────────────────────────────────────────────────────────────

    Q     = torch.empty((N, ncv + 1), device=device, dtype=dtype)
    alpha = torch.zeros(ncv,          device=device, dtype=small_dtype)
    beta  = torch.zeros(ncv,          device=device, dtype=small_dtype)

    Q[:, 0] = _make_initial_vector()

    n_matvec    = 0
    cycles_done = 0
    converged   = False
    breakdown   = False

    best_evals     = None
    best_evecs     = None
    best_residuals = None
    best_m         = None
    best_beta_last = None

    off_tridiag_history  = []
    restart_beta_history = []
    restart_leak_history = []

    start_col          = 0
    m_final            = 1
    beta_last          = torch.tensor(0.0, device=device, dtype=small_dtype)
    has_residual_vector = False

    def _expand_from(start_col_local: int):
        """
        Continue Lanczos recurrence from column start_col_local up to ncv.

        Q[:, :start_col_local+1] must already be an orthonormal partial basis.
        If start_col_local > 0, beta[start_col_local-1] must hold the coupling
        from the previous step into Q[:, start_col_local].

        Mutates alpha, beta, Q in-place.

        FIX 1: The alpha correction inside the reorthogonalization loop has
        been removed. In the original draft:
            aj += coeffs[-1]
        was intended to absorb numerical drift, but dot(q_j, z_1) = 0 by
        construction (z_1 = z_0 - aj*q_j has no q_j component), so coeffs[-1]
        is zero in exact arithmetic and only adds noise in floating point.
        alpha[j] is set correctly from the initial dot product before the
        reorthogonalization loop runs.
        """
        nonlocal n_matvec

        beta_last_local = torch.tensor(0.0, device=device, dtype=small_dtype)
        has_residual    = False
        m_local         = start_col_local + 1

        for j in range(start_col_local, ncv):
            qj = Q[:, j]
            z  = matvec_fn(qj)
            z  = z.to(device=device, dtype=dtype).reshape(-1)
            n_matvec += 1
            # Only new columns require matvecs; retained Ritz vectors are
            # reused from the previous cycle — that is the whole point of IRLM.

            if z.numel() != N:
                raise ValueError(
                    f"matvec returned length {z.numel()}, expected {N}"
                )

            # Three-term recurrence: subtract previous basis vector.
            if j > 0:
                z = z - beta[j - 1].to(dtype=dtype) * Q[:, j - 1]

            # Diagonal coefficient.
            aj = torch.dot(qj, z).to(dtype=small_dtype)
            z  = z - aj.to(dtype=dtype) * qj

            # Full re-orthogonalization (double MGS).
            # FIX 1: Do NOT update aj inside this loop.
            # The residual z after subtracting aj*qj already has zero q_j
            # component; the reorthogonalization loop removes numerical drift
            # against ALL basis vectors but does not change the alpha coefficient.
            if reorthogonalize:
                Qj = Q[:, : j + 1]
                for _ in range(reorth_passes):
                    coeffs = Qj.T @ z
                    z = z - Qj @ coeffs
                    # Note: we do NOT do `aj += coeffs[-1]` here.
                    # dot(q_j, z_1) = 0 by construction, so coeffs[-1] ≈ 0.
                    # Adding it only introduces floating-point noise.

            bj = torch.linalg.norm(z).to(dtype=small_dtype)

            alpha[j]        = aj
            beta_last_local = bj
            m_local         = j + 1

            if float(bj.detach().cpu()) < breakdown_tol:
                has_residual = False
                return m_local, beta_last_local, has_residual

            beta[j]       = bj
            Q[:, j + 1]   = z / bj.to(dtype=dtype)
            has_residual  = True

        return ncv, beta_last_local, has_residual

    # ──────────────────────────────────────────────────────────────────────
    # Main IRLM loop
    # ──────────────────────────────────────────────────────────────────────

    try:
        for cycle in range(max_restarts + 1):
            cycles_done = cycle + 1

            m, beta_last, has_residual_vector = _expand_from(start_col)
            m_final = m

            Tm = _build_T(alpha, beta, m)
            evals_all, evecs_all = _small_eigh_desc(Tm)

            n_take  = min(k, m)
            evals_k = evals_all[:n_take]
            evecs_k = evecs_all[:, :n_take]

            residuals = torch.abs(beta_last * evecs_k[-1, :])

            if relative_residual:
                thresholds = tol * torch.maximum(
                    torch.ones_like(evals_k),
                    torch.abs(evals_k),
                )
            else:
                thresholds = torch.full_like(residuals, float(tol))

            n_conv = int(torch.sum(residuals <= thresholds).detach().cpu())

            best_evals     = evals_all
            best_evecs     = evecs_all
            best_residuals = residuals
            best_m         = m
            best_beta_last = beta_last

            if verbose:
                max_res       = float(torch.max(residuals).detach().cpu()) if residuals.numel() else float("nan")
                theta_preview = evals_k[: min(5, len(evals_k))].detach().cpu().numpy()
                print(
                    f"  cycle={cycle:02d}  m={m:4d}  "
                    f"beta={float(beta_last.detach().cpu()):.3e}  "
                    f"converged={n_conv}/{n_take}  max_res={max_res:.3e}"
                )
                print(f"    theta[:min(5,k)] = {theta_preview}")

            if n_take >= k and bool(
                torch.all(residuals[:k] <= thresholds[:k]).detach().cpu()
            ):
                converged = True
                if verbose:
                    print(f"[irlm] converged after cycle={cycle}")
                break

            if float(beta_last.detach().cpu()) < breakdown_tol:
                breakdown = True
                if verbose:
                    print(f"[irlm] lucky breakdown after cycle={cycle}")
                break

            if cycle == max_restarts:
                break

            if not has_residual_vector:
                breakdown = True
                if verbose:
                    print("[irlm] no residual vector available for restart")
                break

            if m <= keep + 1:
                if verbose:
                    print("[irlm] projected dimension too small for restart")
                break

            # ──────────────────────────────────────────────────────────────
            # IRLM restart: apply unwanted Ritz values as implicit shifts.
            # ──────────────────────────────────────────────────────────────

            keep_eff = min(keep, m - 2)
            shifts   = evals_all[keep_eff:m].detach()

            if shifts.numel() == 0:
                if verbose:
                    print("[irlm] no shifts available")
                break

            T_shifted, Z = _implicit_shifted_qr_dense(Tm, shifts)

            # Diagnostic: relative off-tridiagonal fill in the retained block.
            off_rel = _off_tridiag_rel(T_shifted[:keep_eff, :keep_eff])
            off_tridiag_history.append(off_rel)

            # Rotate retained basis vectors only (saves VRAM).
            Z_keep = Z[:, :keep_eff].to(dtype=dtype)
            Qkeep  = Q[:, :m] @ Z_keep
            Tkeep  = T_shifted[:keep_eff, :keep_eff]

            # ──────────────────────────────────────────────────────────────
            # Restart residual.
            #
            # FIX 2: In implicit mode, compute the residual BEFORE calling
            # _canonicalize_tridiag_and_basis, because canonicalization may
            # flip signs on Qkeep columns. The coefficient c1 =
            # T_shifted[keep_eff, keep_eff-1] is extracted from T_shifted
            # before any sign flips; if we flipped Qkeep[:,keep_eff-1] first,
            # c1 would have the wrong sign.
            # In explicit mode, the residual is recomputed via a fresh matvec
            # after canonicalization, so ordering does not matter there.
            # ──────────────────────────────────────────────────────────────

            if restart_residual_mode == "implicit":
                # Formal IRLM residual (no extra matvec):
                #   r = T_shifted[keep_eff, keep_eff-1] * q_discard
                #       + beta_last * Z[-1, keep_eff-1] * q_next
                # This uses T_shifted and Qkeep BEFORE any sign canonicalization.
                Z_discard  = Z[:, keep_eff].to(dtype=dtype)
                q_discard  = Q[:, :m] @ Z_discard
                old_q_next = Q[:, m].clone()

                c1 = T_shifted[keep_eff, keep_eff - 1].to(dtype=dtype)
                c2 = (beta_last * Z[-1, keep_eff - 1]).to(dtype=dtype)

                r = c1 * q_discard + c2 * old_q_next

                # Leak diagnostic: in exact arithmetic, entries of the
                # residual row before keep_eff-1 should be zero.
                row_res = beta_last * Z[-1, :]
                leak    = float(
                    torch.linalg.norm(row_res[: max(0, keep_eff - 1)]).detach().cpu()
                )
                restart_leak_history.append(leak)

                # Now canonicalize basis and tridiagonal.
                Qkeep, alpha_new, beta_new, Tkeep_tri = \
                    _canonicalize_tridiag_and_basis(Qkeep, Tkeep)

                # Orthogonalize residual against canonicalized retained basis.
                r = _orthogonalize(Qkeep, r, passes=reorth_passes)
                beta_restart = torch.linalg.norm(r).to(dtype=small_dtype)

            else:
                # Explicit restart: canonicalize first, then one fresh matvec.
                Qkeep, alpha_new, beta_new, Tkeep_tri = \
                    _canonicalize_tridiag_and_basis(Qkeep, Tkeep)

                qlast = Qkeep[:, keep_eff - 1]
                r     = matvec_fn(qlast)
                r     = r.to(device=device, dtype=dtype).reshape(-1)
                n_matvec += 1

                # Subtract tridiagonal neighbors.
                r = r - alpha_new[keep_eff - 1].to(dtype=dtype) * qlast
                if keep_eff > 1:
                    r = r - beta_new[keep_eff - 2].to(dtype=dtype) * Qkeep[:, keep_eff - 2]

                r = _orthogonalize(Qkeep, r, passes=reorth_passes)
                beta_restart = torch.linalg.norm(r).to(dtype=small_dtype)

            restart_beta_history.append(float(beta_restart.detach().cpu()))

            if float(beta_restart.detach().cpu()) < breakdown_tol:
                breakdown = True
                if verbose:
                    print("[irlm] restart residual collapsed; treating as breakdown")
                break

            q_restart = r / beta_restart.to(dtype=dtype)

            # ──────────────────────────────────────────────────────────────
            # Install restarted basis into persistent storage.
            # ──────────────────────────────────────────────────────────────

            Q[:, :keep_eff]  = Qkeep
            Q[:, keep_eff]   = q_restart

            alpha.zero_()
            beta.zero_()

            alpha[:keep_eff] = alpha_new
            if keep_eff > 1:
                beta[: keep_eff - 1] = beta_new
            beta[keep_eff - 1] = beta_restart

            start_col = keep_eff

            if verbose:
                print(
                    f"    restart: shifts={int(shifts.numel())}  "
                    f"off_tridiag={off_rel:.3e}  "
                    f"beta_restart={float(beta_restart.detach().cpu()):.3e}"
                )

        # ──────────────────────────────────────────────────────────────────
        # Final Ritz extraction
        # ──────────────────────────────────────────────────────────────────

        if best_evals is None or best_evecs is None or best_m is None:
            raise RuntimeError("IRLM failed before producing any Ritz pairs")

        Tfinal         = _build_T(alpha, beta, m_final)
        evals_final, evecs_final = _small_eigh_desc(Tfinal)

        n_return   = min(k, m_final)
        vals       = evals_final[:n_return]
        Y          = evecs_final[:, :n_return].to(dtype=dtype)

        ritz_vecs  = Q[:, :m_final] @ Y
        ritz_vecs  = ritz_vecs / torch.linalg.norm(
            ritz_vecs, dim=0
        ).clamp_min(torch.finfo(dtype).eps)

        final_residuals = torch.abs(beta_last * evecs_final[-1, :n_return])

        elapsed = time.perf_counter() - t_start

        if verbose:
            print(
                f"[irlm] finished  converged={converged}  breakdown={breakdown}  "
                f"cycles={cycles_done}  matvecs={n_matvec}  time={elapsed:.2f}s"
            )
            print(f"  eigenvalues: {vals.detach().cpu().numpy()}")
            print(f"  residuals:   {final_residuals.detach().cpu().numpy()}")

        if return_torch:
            vals_out = vals
            vecs_out = ritz_vecs
        else:
            vals_out = vals.detach().cpu().numpy().astype(np.float64)
            vecs_out = ritz_vecs.detach().cpu().numpy()

        if return_info:
            info = {
                "converged":             bool(converged),
                "breakdown":             bool(breakdown),
                "cycles":                int(cycles_done),
                "n_matvec":              int(n_matvec),
                "ncv":                   int(ncv),
                "keep":                  int(keep),
                "k":                     int(k),
                "tol":                   float(tol),
                "relative_residual":     bool(relative_residual),
                "restart_residual_mode": restart_residual_mode,
                "elapsed":               float(elapsed),
                "m_final":               int(m_final),
                "beta_final":            float(beta_last.detach().cpu()),
                "residuals":             final_residuals.detach().cpu().numpy().astype(np.float64),
                "off_tridiag_restart":   off_tridiag_history,
                "restart_beta":          restart_beta_history,
                "restart_leak":          restart_leak_history,
                "dtype":                 str(dtype),
                "small_dtype":           str(small_dtype),
                "device":                str(device),
            }
            return vals_out, vecs_out, info

        return vals_out, vecs_out

    finally:
        torch.set_grad_enabled(grad_state)
