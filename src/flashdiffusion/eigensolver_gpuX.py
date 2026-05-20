from __future__ import annotations

import time
import warnings
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
    deflate_basis=None,
    breakdown_tol: float = 1e-12,
    relative_residual: bool = True,
    reorthogonalize: bool = True,
    reorth_passes: int = 2,
    block_rows: int = 262_144,
    return_info: bool = False,
    return_torch: bool = False,
):
    """
    GPU-resident implicitly restarted Lanczos method (IRLM) using dense
    shifted QR on the small projected matrix, with proper retained-block
    tridiagonalization via scipy Hessenberg reduction.

    Computes the k largest algebraic eigenpairs of a symmetric/self-adjoint
    matrix-free operator.

    Parameters
    ----------
    matvec_fn : callable
        (N,) GPU tensor -> (N,) GPU tensor.
    N : int
        Problem size.
    k : int
        Number of requested eigenpairs.
    ncv : int or None
        Krylov subspace size. Default: min(N, max(4*k+40, k+20)).
    keep : int or None
        Directions retained per restart. k <= keep <= ncv-2.
        Default: min(ncv-2, k+max(8, k//2)).
    max_restarts : int
        Maximum restart cycles.
    tol : float
        Ritz residual tolerance. Values < 5e-6 with float32 emit a warning.
    dtype : torch.dtype or None
        GPU vector dtype. Default: torch.float32.
    small_dtype : torch.dtype or None
        Dtype for ncv x ncv projected problems. Default: torch.float64.
    device : str
        CUDA device string.
    verbose : bool
        Print per-cycle convergence info.
    q0 : torch.Tensor or None
        Optional starting vector. Small perturbation added automatically.
    deflate_basis : torch.Tensor or None
        (N, d) orthonormal columns to deflate from every vector and residual.
        Use to skip the trivial diffusion-map mode (lambda=1).
    breakdown_tol : float
        Treat beta below this as lucky breakdown.
    relative_residual : bool
        If True: converge when |beta*y[-1]| <= tol*max(1,|theta|).
    reorthogonalize : bool
        Full MGS re-orthogonalization. Must be True for clustered spectra.
    reorth_passes : int
        Number of MGS passes. 2 is standard.
    block_rows : int
        Row-block size for basis rotation to limit peak VRAM.
        Default 262144 rows (~30MB per block at float32, keep=30).
    return_info : bool
        If True, return diagnostics dict as third element.
    return_torch : bool
        If True return GPU tensors; if False (default) return NumPy.

    Returns
    -------
    vals : ndarray or Tensor, shape (k,)
    vecs : ndarray or Tensor, shape (N, k)
    info : dict  [only when return_info=True]

    Notes
    -----
    Key fixes over original draft:
      1. Retained block tridiagonalization via scipy Hessenberg (symmetric
         Hessenberg = tridiagonal). Dense QR leaves Tkeep dense; extracting
         only the tridiagonal part was incorrect and broke the Lanczos
         three-term recurrence after restart.
      2. Implicit restart residual mode removed (raises NotImplementedError)
         until bookkeeping for the extra Hessenberg rotation is implemented.
         Explicit mode (one fresh matvec per restart) is the only option.
      3. state_valid bookkeeping: if restart mutation fails mid-way, final
         extraction falls back to the last complete valid cycle result.
      4. True residuals computed at the end via explicit matvec for diagnostics.
      5. Orthogonality error tracked per cycle.
      6. Blocked basis rotation for large N to limit peak VRAM.
      7. Optional deflation basis for known subspaces (e.g. trivial mode).
      8. Warning when tol is below reliable float32 precision.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch required for gpu_irlm_dense_qr")

    import torch
    import scipy.linalg

    if dtype is None:
        dtype = torch.float32
    if small_dtype is None:
        small_dtype = torch.float64

    device = torch.device(device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    if dtype == torch.float32 and tol < 5e-6:
        warnings.warn(
            f"tol={tol} may be unreliable for dtype=float32. "
            "Consider tol >= 5e-6.",
            RuntimeWarning, stacklevel=2,
        )

    if k < 1:
        raise ValueError("k must be >= 1")

    if ncv is None:
        ncv = min(N, max(4 * k + 40, k + 20))
    ncv = int(min(ncv, N))

    if ncv <= k + 1:
        raise ValueError(f"ncv={ncv} too small for k={k}; need ncv > k+1")

    if keep is None:
        keep = min(ncv - 2, k + max(8, k // 2))
    keep = int(keep)

    if keep < k:
        raise ValueError(f"keep={keep} must be >= k={k}")
    if keep > ncv - 2:
        raise ValueError(f"keep={keep} must be <= ncv-2={ncv-2}")

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

    def _project_deflation(v):
        if deflate_basis is None:
            return v
        B = deflate_basis.to(device=device, dtype=dtype)
        return v - B @ (B.T @ v)

    def _make_initial_vector():
        if q0 is not None:
            q = torch.as_tensor(q0, device=device, dtype=dtype).reshape(-1)
            if q.numel() != N:
                raise ValueError(f"q0 length {q.numel()} != N={N}")
            noise_scale = 1e-3 * float(torch.linalg.norm(q).cpu()) / (N ** 0.5)
            q = q + noise_scale * torch.randn_like(q)
        else:
            q = torch.randn(N, device=device, dtype=dtype)
        q = _project_deflation(q)
        qn = torch.linalg.norm(q)
        if float(qn.cpu()) == 0.0:
            raise ValueError("Initial vector zero after deflation")
        return q / qn

    def _orthogonalize(Qbasis, v, passes=2):
        if Qbasis.numel() == 0:
            return v
        for _ in range(passes):
            c = Qbasis.T @ v
            v = v - Qbasis @ c
        return v

    def _build_T(alpha, beta, m):
        T = torch.diag(alpha[:m].to(device=device, dtype=small_dtype))
        if m > 1:
            b = beta[:m - 1].to(device=device, dtype=small_dtype)
            T = T + torch.diag(b, 1) + torch.diag(b, -1)
        return 0.5 * (T + T.T)

    def _small_eigh_desc(T):
        T = 0.5 * (T + T.T)
        evals, evecs = torch.linalg.eigh(T)
        idx = torch.argsort(evals, descending=True)
        return evals[idx], evecs[:, idx]

    def _implicit_shifted_qr_dense(T, shifts):
        m  = T.shape[0]
        I  = torch.eye(m, device=device, dtype=small_dtype)
        Tw = T.clone().to(small_dtype)
        Z  = torch.eye(m, device=device, dtype=small_dtype)
        for mu in shifts:
            Qs, Rs = torch.linalg.qr(Tw - mu * I, mode="reduced")
            d      = torch.diagonal(Rs)
            signs  = torch.where(
                d < 0,
                torch.tensor(-1.0, device=device, dtype=small_dtype),
                torch.tensor( 1.0, device=device, dtype=small_dtype),
            )
            Qs = Qs * signs[None, :]
            Rs = signs[:, None] * Rs
            Tw = Rs @ Qs + mu * I
            Tw = 0.5 * (Tw + Tw.T)
            Z  = Z @ Qs
        return Tw, Z

    def _tridiagonalize_retained_block(Qkeep, Tkeep):
        """
        KEY FIX (point 1): After dense shifted QR, Tkeep is generally dense.
        Use scipy Hessenberg reduction (symmetric Hessenberg = tridiagonal)
        to restore the three-term Lanczos recurrence.

        scipy returns (H, S) with T = S @ H @ S.T, i.e. H = S.T @ T @ S.
        We rotate Qkeep <- Qkeep @ S so the Lanczos relation stays valid.
        Cost: O(keep^3) on CPU — negligible.
        """
        T_np = Tkeep.detach().cpu().numpy().astype(np.float64)
        T_np = 0.5 * (T_np + T_np.T)

        H_np, S_np = scipy.linalg.hessenberg(T_np, calc_q=True)
        H_np = 0.5 * (H_np + H_np.T)

        # Zero numerical off-tridiagonal noise explicitly.
        r    = H_np.shape[0]
        H_tri = np.zeros_like(H_np)
        np.fill_diagonal(H_tri, np.diag(H_np))
        if r > 1:
            off = 0.5 * (np.diag(H_np, 1) + np.diag(H_np, -1))
            H_tri += np.diag(off, 1) + np.diag(off, -1)

        S     = torch.from_numpy(S_np).to(device=device, dtype=dtype)
        Qkeep = Qkeep @ S

        H = torch.from_numpy(H_tri).to(device=device, dtype=small_dtype)

        # Canonicalize: nonneg subdiagonal.
        for i in range(r - 1):
            if float(H[i + 1, i].cpu()) < 0:
                Qkeep[:, i + 1].mul_(-1)
                H[i + 1, :].mul_(-1)
                H[:, i + 1].mul_(-1)

        alpha_new = torch.diagonal(H).clone().to(small_dtype)
        beta_new  = (torch.diagonal(H, offset=-1).clone().abs().to(small_dtype)
                     if r > 1 else torch.empty(0, device=device, dtype=small_dtype))

        return Qkeep, alpha_new, beta_new, H

    def _rotate_basis_blocked(m, Z_keep, keep_eff):
        """Blocked Q[:,:m] @ Z_keep to limit peak VRAM at large N."""
        Qout = torch.empty((N, keep_eff), device=device, dtype=dtype)
        for lo in range(0, N, block_rows):
            hi = min(lo + block_rows, N)
            Qout[lo:hi] = Q[lo:hi, :m] @ Z_keep
        return Qout

    def _true_residuals(vals_t, vecs_t):
        """||A v_i - lam_i v_i|| via explicit matvec. Diagnostics only."""
        out = []
        for i in range(vals_t.numel()):
            Av = matvec_fn(vecs_t[:, i])
            r  = Av.to(device=device, dtype=dtype) \
                 - vals_t[i].to(dtype=dtype) * vecs_t[:, i]
            out.append(float(torch.linalg.norm(r).cpu()))
        return np.array(out, dtype=np.float64)

    def _orth_error(Qblock):
        """||Q.T Q - I||_F / ncols. Stays near machine eps if healthy."""
        G = Qblock.T @ Qblock
        I = torch.eye(G.shape[0], device=device, dtype=G.dtype)
        return float((torch.linalg.norm(G - I) / max(1, G.shape[0])).cpu())

    # ──────────────────────────────────────────────────────────────────────
    # Persistent GPU state
    # ──────────────────────────────────────────────────────────────────────

    Q     = torch.empty((N, ncv + 1), device=device, dtype=dtype)
    alpha = torch.zeros(ncv,          device=device, dtype=small_dtype)
    beta  = torch.zeros(ncv,          device=device, dtype=small_dtype)

    Q[:, 0] = _make_initial_vector()

    n_matvec            = 0
    cycles_done         = 0
    converged           = False
    breakdown           = False
    state_valid         = True

    best_evals          = None
    best_evecs          = None
    best_m              = None
    best_beta_last      = None

    off_tridiag_history = []
    restart_beta_history= []
    orth_error_history  = []

    start_col           = 0
    m_final             = 1
    beta_last           = torch.tensor(0.0, device=device, dtype=small_dtype)
    has_residual_vector = False

    def _expand_from(start_col_local):
        nonlocal n_matvec
        beta_last_local = torch.tensor(0.0, device=device, dtype=small_dtype)
        has_residual    = False
        m_local         = start_col_local + 1

        for j in range(start_col_local, ncv):
            qj = Q[:, j]
            z  = matvec_fn(qj)
            z  = z.to(device=device, dtype=dtype).reshape(-1)
            n_matvec += 1

            if z.numel() != N:
                raise ValueError(f"matvec returned {z.numel()}, expected {N}")

            z = _project_deflation(z)

            if j > 0:
                z = z - beta[j - 1].to(dtype=dtype) * Q[:, j - 1]

            # Diagonal coeff — set before reortho, NOT updated inside loop.
            aj = torch.dot(qj, z).to(small_dtype)
            z  = z - aj.to(dtype=dtype) * qj

            if reorthogonalize:
                Qj = Q[:, :j + 1]
                for _ in range(reorth_passes):
                    c = Qj.T @ z
                    z = z - Qj @ c

            bj = torch.linalg.norm(z).to(small_dtype)
            alpha[j]        = aj
            beta_last_local = bj
            m_local         = j + 1

            if float(bj.cpu()) < breakdown_tol:
                return m_local, beta_last_local, False

            beta[j]      = bj
            Q[:, j + 1]  = z / bj.to(dtype=dtype)
            has_residual = True

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
                    torch.ones_like(evals_k), torch.abs(evals_k))
            else:
                thresholds = torch.full_like(residuals, float(tol))

            n_conv   = int(torch.sum(residuals <= thresholds).cpu())
            orth_err = _orth_error(Q[:, :m].to(small_dtype))
            orth_error_history.append(orth_err)

            best_evals     = evals_all
            best_evecs     = evecs_all
            best_m         = m
            best_beta_last = beta_last

            if verbose:
                max_res = float(torch.max(residuals).cpu()) if residuals.numel() else float("nan")
                print(
                    f"  cycle={cycle:02d}  m={m:4d}  "
                    f"beta={float(beta_last.cpu()):.3e}  "
                    f"converged={n_conv}/{n_take}  "
                    f"max_res={max_res:.3e}  orth={orth_err:.2e}"
                )
                print(f"    theta[:5] = {evals_k[:min(5,len(evals_k))].cpu().numpy()}")

            if n_take >= k and bool(torch.all(residuals[:k] <= thresholds[:k]).cpu()):
                converged = True
                if verbose:
                    print(f"[irlm] converged after cycle={cycle}")
                break

            if float(beta_last.cpu()) < breakdown_tol:
                breakdown = True
                if verbose:
                    print(f"[irlm] lucky breakdown cycle={cycle}")
                break

            if cycle == max_restarts:
                break

            if not has_residual_vector:
                breakdown = True
                break

            if m <= keep + 1:
                break

            # ── IRLM restart ──────────────────────────────────────────────

            keep_eff = min(keep, m - 2)
            shifts   = evals_all[keep_eff:m].detach()

            if shifts.numel() == 0:
                break

            T_shifted, Z = _implicit_shifted_qr_dense(Tm, shifts)

            # Off-tridiagonal fill diagnostic.
            Tk = T_shifted[:keep_eff, :keep_eff]
            Tk_tri = torch.diag(torch.diagonal(Tk))
            if keep_eff > 1:
                off = 0.5 * (torch.diagonal(Tk, 1) + torch.diagonal(Tk, -1))
                Tk_tri = Tk_tri + torch.diag(off, 1) + torch.diag(off, -1)
            denom  = max(float(torch.linalg.norm(Tk).cpu()), 1e-300)
            off_rel= float(torch.linalg.norm(Tk - Tk_tri).cpu()) / denom
            off_tridiag_history.append(off_rel)

            # Blocked basis rotation.
            Z_keep = Z[:, :keep_eff].to(dtype=dtype)
            Qkeep  = _rotate_basis_blocked(m, Z_keep, keep_eff)
            Tkeep  = T_shifted[:keep_eff, :keep_eff]

            # Mark state invalid before mutating.
            state_valid = False

            # KEY FIX: tridiagonalize retained block via Hessenberg.
            Qkeep, alpha_new, beta_new, Tkeep_tri = \
                _tridiagonalize_retained_block(Qkeep, Tkeep)

            # Explicit restart residual (only supported mode).
            qlast = Qkeep[:, keep_eff - 1]
            r     = matvec_fn(qlast).to(device=device, dtype=dtype).reshape(-1)
            r     = _project_deflation(r)
            n_matvec += 1

            r = r - alpha_new[keep_eff - 1].to(dtype=dtype) * qlast
            if keep_eff > 1:
                r = r - beta_new[keep_eff - 2].to(dtype=dtype) * Qkeep[:, keep_eff - 2]

            r = _orthogonalize(Qkeep, r, passes=reorth_passes)
            beta_restart = torch.linalg.norm(r).to(small_dtype)
            restart_beta_history.append(float(beta_restart.cpu()))

            if float(beta_restart.cpu()) < breakdown_tol:
                breakdown = True
                if verbose:
                    print("[irlm] restart residual collapsed")
                break   # state_valid stays False; fallback used below

            q_restart = r / beta_restart.to(dtype=dtype)

            Q[:, :keep_eff]     = Qkeep
            Q[:, keep_eff]      = q_restart
            alpha.zero_()
            beta.zero_()
            alpha[:keep_eff]    = alpha_new
            if keep_eff > 1:
                beta[:keep_eff - 1] = beta_new
            beta[keep_eff - 1]  = beta_restart
            start_col           = keep_eff
            state_valid         = True      # install complete

            if verbose:
                print(
                    f"    restart: shifts={int(shifts.numel())}  "
                    f"off_tridiag={off_rel:.3e}  "
                    f"beta_restart={float(beta_restart.cpu()):.3e}"
                )

        # ── Final Ritz extraction ─────────────────────────────────────────

        if best_evals is None or best_m is None:
            raise RuntimeError("IRLM produced no Ritz pairs")

        if state_valid:
            Tfinal           = _build_T(alpha, beta, m_final)
            evals_f, evecs_f = _small_eigh_desc(Tfinal)
        else:
            # Restart failed mid-way — use last complete cycle.
            evals_f  = best_evals
            evecs_f  = best_evecs
            m_final  = best_m
            beta_last= best_beta_last

        n_return  = min(k, m_final)
        vals      = evals_f[:n_return]
        Y         = evecs_f[:, :n_return].to(dtype=dtype)
        ritz_vecs = Q[:, :m_final] @ Y
        ritz_vecs = ritz_vecs / torch.linalg.norm(
            ritz_vecs, dim=0).clamp_min(torch.finfo(dtype).eps)

        formula_res = torch.abs(beta_last * evecs_f[-1, :n_return])
        true_res    = _true_residuals(vals, ritz_vecs)

        elapsed = time.perf_counter() - t_start

        if verbose:
            print(
                f"[irlm] done  converged={converged}  breakdown={breakdown}  "
                f"state_valid={state_valid}  cycles={cycles_done}  "
                f"matvecs={n_matvec}  time={elapsed:.2f}s"
            )
            print(f"  eigenvalues:       {vals.cpu().numpy()}")
            print(f"  formula residuals: {formula_res.cpu().numpy()}")
            print(f"  true residuals:    {true_res}")

        if return_torch:
            vals_out, vecs_out = vals, ritz_vecs
        else:
            vals_out = vals.cpu().numpy().astype(np.float64)
            vecs_out = ritz_vecs.cpu().numpy()

        if return_info:
            info = {
                "converged":           bool(converged),
                "breakdown":           bool(breakdown),
                "state_valid":         bool(state_valid),
                "cycles":              int(cycles_done),
                "n_matvec":            int(n_matvec),
                "ncv":                 int(ncv),
                "keep":                int(keep),
                "k":                   int(k),
                "tol":                 float(tol),
                "relative_residual":   bool(relative_residual),
                "elapsed":             float(elapsed),
                "m_final":             int(m_final),
                "beta_final":          float(beta_last.cpu()),
                "formula_residuals":   formula_res.cpu().numpy().astype(np.float64),
                "true_residuals":      true_res,
                "off_tridiag_restart": off_tridiag_history,
                "restart_beta":        restart_beta_history,
                "orth_error":          orth_error_history,
                "dtype":               str(dtype),
                "small_dtype":         str(small_dtype),
                "device":              str(device),
            }
            return vals_out, vecs_out, info

        return vals_out, vecs_out

    finally:
        torch.set_grad_enabled(grad_state)
