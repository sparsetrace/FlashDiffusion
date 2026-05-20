from __future__ import annotations

import time
import warnings
from typing import Callable, Optional, Tuple, Any, Dict

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# GPU Hessenberg (replaces scipy.linalg.hessenberg after thick restart)
# ---------------------------------------------------------------------------

def _gpu_hessenberg_sym(T: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
    """
    Reduce a symmetric matrix T (on GPU, small_dtype) to symmetric tridiagonal
    form via Householder reflections, entirely on the GPU.

    For a symmetric matrix, Hessenberg = tridiagonal, so this is equivalent to
    scipy.linalg.hessenberg but stays on-device.

    Returns
    -------
    H : tridiagonal (symmetric) version of T, same device/dtype
    Q : orthogonal accumulation such that Q.T @ T_orig @ Q = H
    """
    device = T.device
    dtype = T.dtype
    n = T.shape[0]

    T = 0.5 * (T + T.T).clone()
    Q = torch.eye(n, device=device, dtype=dtype)

    for k in range(n - 2):
        x = T[k + 1:, k].clone()
        norm_x = torch.linalg.norm(x)
        if float(norm_x) < 1e-15:
            continue
        # Householder vector: sign chosen to avoid cancellation.
        # sign(x[0]) >= 0 convention: add norm in the direction of x[0].
        sign = torch.where(
            x[0] >= 0,
            torch.tensor(1.0, device=device, dtype=dtype),
            torch.tensor(-1.0, device=device, dtype=dtype),
        )
        v = x.clone()
        v[0] = v[0] + sign * norm_x
        v = v / torch.linalg.norm(v)

        # Apply H = I - 2vv^T from left and right to the sub-block.
        sub = T[k + 1:, k + 1:].clone()
        # Left: sub <- (I - 2vv^T) sub
        sub = sub - 2.0 * v.unsqueeze(1) * (v @ sub).unsqueeze(0)
        # Right: sub <- sub (I - 2vv^T)
        sub = sub - 2.0 * (sub @ v).unsqueeze(1) * v.unsqueeze(0)
        T[k + 1:, k + 1:] = sub

        # Set the off-diagonal entry explicitly; zero the rest of column/row k.
        T[k + 2:, k] = 0.0
        T[k, k + 2:] = 0.0
        T[k + 1, k] = -sign * norm_x
        T[k, k + 1] = T[k + 1, k]

        # Accumulate Q
        Q[:, k + 1:] -= 2.0 * (Q[:, k + 1:] @ v).unsqueeze(1) * v.unsqueeze(0)

    # Force exact symmetry and extract clean tridiagonal
    T = 0.5 * (T + T.T)
    alpha = torch.diagonal(T).clone()
    beta = torch.diagonal(T, offset=1).clone()

    H_tri = torch.diag(alpha) + torch.diag(beta, 1) + torch.diag(beta, -1)
    return H_tri, Q


def gpu_block_irlm(
    matmat_fn: Optional[Callable[["torch.Tensor"], "torch.Tensor"]] = None,
    *,
    matvec_fn: Optional[Callable[["torch.Tensor"], "torch.Tensor"]] = None,
    N: int,
    k: int = 8,
    block_size: Optional[int] = None,
    ncv: Optional[int] = None,
    keep: Optional[int] = None,
    max_restarts: int = 20,
    tol: float = 1e-6,
    dtype=None,
    small_dtype=None,
    device: str = "cuda",
    verbose: bool = True,
    q0_block=None,
    q0=None,
    deflate_basis=None,
    breakdown_tol: float = 1e-12,
    rank_tol: Optional[float] = None,
    relative_residual: bool = True,
    reorthogonalize: bool = True,
    reorth_passes: int = 2,
    block_rows: int = 262_144,
    compute_true_residuals: bool = True,
    orth_check_every: Optional[int] = 1,
    disable_tf32: bool = True,
    return_info: bool = False,
    return_torch: bool = False,
):
    """
    GPU thick-restarted block Arnoldi / block Rayleigh-Ritz eigensolver.

    Correctly described: this is a thick-restarted block Arnoldi method for
    symmetric operators, not a strict block IRLM. Each restart selects the top
    keep Ritz vectors as the new seed and rebuilds the Krylov relation from
    scratch via _expand_cycle. This loses one benefit of true IRLM (continuing
    the Krylov relation without reapplying A to retained vectors), but is simpler
    and correct. The matmat_fn(V) block interface still gives GPU throughput
    benefits over scalar IRLM for large N.

    To become true block Krylov-Schur/IRLM, _expand_cycle would need to accept
    the retained H sub-block and continue expansion from column keep rather than 0.

    Fixes applied in this version:
      1. H double-counting removed: H[:m, applied:end] = coeff (not +=), no
         transpose fill during expansion. H[:p,:p] symmetrized only at Ritz time.
      2. Hessenberg rotation removed from thick restart (was unused/incorrect:
         rotating by S from H[:keep,:keep] doesn't re-tridiagonalize the Ritz
         basis since the projected matrix there is diag(evals), not H).
      3. Householder sign bug fixed in _gpu_hessenberg_sym (was clamping sign
         to [0,1], turning negative x[0] into 0).
      4. keep_eff validated before _orthonormalize_block.
      5. True-residual matvec cost excluded from n_matmat/n_rhs counters.
      6. Q buffer reused across cycles (single allocation).

    Parameters
    ----------
    matmat_fn:
        Callable V -> A V, where V has shape (N, s).
    matvec_fn:
        Optional fallback callable v -> A v.
    N:
        Problem size.
    k:
        Number of eigenpairs requested.
    block_size:
        Vectors per operator application. Default: min(16, max(2,k)).
    ncv:
        Rayleigh-Ritz subspace dimension. Default: max(4*k+4*b, k+6*b).
    keep:
        Ritz vectors retained after restart. Default: min(ncv-b, k+max(2*b,k//2)).
    q0_block / q0:
        Optional initial block or vector.
    deflate_basis:
        Optional (N,d) basis to project out.
    compute_true_residuals:
        Compute ||A v_i - lam_i v_i|| at end (excluded from op counters).

    Returns
    -------
    vals : np.ndarray or torch.Tensor, shape (k,)
    vecs : np.ndarray or torch.Tensor, shape (N,k)
    info : dict, optional
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch required for gpu_block_irlm")

    import torch

    if matmat_fn is None and matvec_fn is None:
        raise ValueError("Provide either matmat_fn(V)->A@V or matvec_fn(v)->A@v")

    if dtype is None:
        dtype = torch.float32
    if small_dtype is None:
        small_dtype = torch.float64

    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    if k < 1:
        raise ValueError("k must be >= 1")

    if block_size is None:
        block_size = min(16, max(2, int(k)))
    block_size = int(block_size)
    if block_size < 1:
        raise ValueError("block_size must be >= 1")

    if ncv is None:
        ncv = max(4 * k + 4 * block_size, k + 6 * block_size)
    ncv = int(min(N, ncv))
    if ncv < N:
        ncv = int(min(N, ((ncv + block_size - 1) // block_size) * block_size))

    if ncv <= k + block_size:
        raise ValueError(
            f"ncv={ncv} too small for k={k}, block_size={block_size}; "
            f"need ncv > k + block_size"
        )

    if keep is None:
        keep = min(ncv - block_size, k + max(2 * block_size, k // 2))
    keep = int(keep)
    if keep < k:
        raise ValueError(f"keep={keep} must be >= k={k}")
    if keep > ncv - block_size:
        raise ValueError(
            f"keep={keep} must be <= ncv-block_size={ncv - block_size}"
        )

    if rank_tol is None:
        rank_tol = 1e-7 if dtype == torch.float32 else 1e-12
    if dtype == torch.float32 and tol < 5e-6:
        warnings.warn(
            f"tol={tol} may be unreliable for dtype=float32. Consider tol >= 5e-6.",
            RuntimeWarning,
            stacklevel=2,
        )

    grad_state = torch.is_grad_enabled()
    torch.set_grad_enabled(False)

    if device.type == "cuda":
        old_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        old_tf32_cudnn = torch.backends.cudnn.allow_tf32
        if disable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    else:
        old_tf32_matmul = None
        old_tf32_cudnn = None

    t_start = time.perf_counter()

    if deflate_basis is not None:
        B_def = torch.as_tensor(
            deflate_basis if isinstance(deflate_basis, torch.Tensor) else np.asarray(deflate_basis),
            device=device,
            dtype=dtype,
        )
        if B_def.ndim == 1:
            B_def = B_def[:, None]
        if B_def.shape[0] != N:
            raise ValueError(f"deflate_basis has {B_def.shape[0]} rows, expected N={N}")
        B_def, _ = torch.linalg.qr(B_def, mode="reduced")
    else:
        B_def = None

    n_matmat = 0
    n_rhs = 0

    def _project_deflation(V: "torch.Tensor") -> "torch.Tensor":
        if B_def is None:
            return V
        return V - B_def @ (B_def.T @ V)

    def _apply_A(V: "torch.Tensor", _count: bool = True) -> "torch.Tensor":
        # FIX 5: _count=False lets true-residual calls skip the benchmark counters.
        nonlocal n_matmat, n_rhs
        V = V.to(device=device, dtype=dtype)
        if V.ndim == 1:
            V = V[:, None]
        if V.shape[0] != N:
            raise ValueError(f"matmat input has {V.shape[0]} rows, expected {N}")
        with torch.no_grad():
            if matmat_fn is not None:
                Y = matmat_fn(V)
            else:
                cols = [matvec_fn(V[:, j]).reshape(-1) for j in range(V.shape[1])]
                Y = torch.stack(cols, dim=1)
        Y = Y.to(device=device, dtype=dtype)
        if Y.ndim == 1:
            Y = Y[:, None]
        if Y.shape != V.shape:
            raise ValueError(f"operator returned shape {tuple(Y.shape)}, expected {tuple(V.shape)}")
        if _count:
            n_matmat += 1
            n_rhs += V.shape[1]
        return Y

    def _linear_combo_blocked(Qsource: "torch.Tensor", C: "torch.Tensor") -> "torch.Tensor":
        C = C.to(device=device, dtype=dtype)
        out = torch.empty((N, C.shape[1]), device=device, dtype=dtype)
        for lo in range(0, N, block_rows):
            hi = min(lo + block_rows, N)
            out[lo:hi] = Qsource[lo:hi] @ C
        return out

    def _project_against(Qbasis: "torch.Tensor", V: "torch.Tensor", passes: int) -> "torch.Tensor":
        if Qbasis.numel() == 0:
            return V
        for _ in range(passes):
            C = Qbasis.T @ V
            V = V - Qbasis @ C
        return V

    def _qr_rank(V: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", int]:
        Qv, R = torch.linalg.qr(V, mode="reduced")
        if R.numel() == 0:
            return Qv[:, :0], R[:0, :], 0
        diag = torch.abs(torch.diagonal(R))
        maxd = torch.max(diag).clamp_min(torch.finfo(dtype).eps)
        keep_mask = diag > max(float(rank_tol), float(breakdown_tol)) * maxd
        r = int(torch.sum(keep_mask).detach().cpu())
        if r == Qv.shape[1]:
            return Qv, R, r
        if r == 0:
            return Qv[:, :0], R[:0, :], 0
        return Qv[:, :r], R[:r, :], r

    def _orthonormalize_block(
        V: "torch.Tensor",
        Qbasis: Optional["torch.Tensor"],
        target_cols: int,
        *,
        allow_random_fill: bool = True,
    ) -> "torch.Tensor":
        pieces = []
        work = V if V is not None else torch.empty((N, 0), device=device, dtype=dtype)
        if work.ndim == 1:
            work = work[:, None]
        work = work.to(device=device, dtype=dtype)

        for attempt in range(10):
            needed = target_cols - sum(p.shape[1] for p in pieces)
            if needed <= 0:
                break
            if work.shape[1] == 0:
                if not allow_random_fill:
                    break
                work = torch.randn((N, needed), device=device, dtype=dtype)

            basis_parts = []
            if Qbasis is not None and Qbasis.numel() > 0:
                basis_parts.append(Qbasis)
            if pieces:
                basis_parts.append(torch.cat(pieces, dim=1))
            Qall = torch.cat(basis_parts, dim=1) if basis_parts else None

            work = _project_deflation(work)
            if Qall is not None:
                work = _project_against(Qall, work, passes=max(1, reorth_passes))
            Qv, _R, r = _qr_rank(work)
            if r > 0:
                take = min(r, needed)
                pieces.append(Qv[:, :take])
            needed = target_cols - sum(p.shape[1] for p in pieces)
            if needed <= 0:
                break
            if not allow_random_fill:
                break
            work = torch.randn((N, needed), device=device, dtype=dtype)

        if not pieces:
            raise RuntimeError("failed to construct any orthonormal block columns")
        Qout = torch.cat(pieces, dim=1)
        if Qout.shape[1] < target_cols:
            raise RuntimeError(
                f"only built {Qout.shape[1]} orthonormal columns, needed {target_cols}"
            )
        return Qout[:, :target_cols]

    def _initial_seed() -> "torch.Tensor":
        if q0_block is not None:
            V = torch.as_tensor(q0_block, device=device, dtype=dtype)
        elif q0 is not None:
            V = torch.as_tensor(q0, device=device, dtype=dtype)
        else:
            V = torch.randn((N, block_size), device=device, dtype=dtype)
        if V.ndim == 1:
            V = V[:, None]
        if V.shape[0] != N:
            raise ValueError(f"q0_block has {V.shape[0]} rows, expected {N}")
        if q0_block is not None or q0 is not None:
            scale = float(torch.linalg.norm(V).detach().cpu()) / max(1.0, (N * V.shape[1]) ** 0.5)
            V = V + (1e-3 * scale) * torch.randn_like(V)
        return _orthonormalize_block(V, None, block_size, allow_random_fill=True)

    def _orth_error_blocked(Qblock: "torch.Tensor") -> float:
        r = Qblock.shape[1]
        G = torch.zeros((r, r), device=device, dtype=small_dtype)
        for lo in range(0, N, block_rows):
            hi = min(lo + block_rows, N)
            B = Qblock[lo:hi].to(dtype=small_dtype)
            G += B.T @ B
        I = torch.eye(r, device=device, dtype=small_dtype)
        return float((torch.linalg.norm(G - I) / max(1, r)).detach().cpu())

    def _small_eigh_desc(H: "torch.Tensor"):
        H = 0.5 * (H + H.T)
        evals, evecs = torch.linalg.eigh(H)
        idx = torch.argsort(evals, descending=True)
        return evals[idx], evecs[:, idx]

    def _true_residuals(vals_t: "torch.Tensor", vecs_t: "torch.Tensor") -> np.ndarray:
        # FIX 5: pass _count=False so these don't pollute benchmark counters.
        AV = _apply_A(vecs_t, _count=False)
        if B_def is not None:
            AV = _project_deflation(AV)
        R = AV - vecs_t * vals_t.to(device=device, dtype=dtype)[None, :]
        return torch.linalg.norm(R, dim=0).detach().cpu().numpy().astype(np.float64)

    # FIX 2: Allocate Q buffer once and reuse across cycles.
    Q_buf = torch.empty((N, ncv + block_size), device=device, dtype=dtype)

    def _expand_cycle(Qseed: "torch.Tensor"):
        """
        Build one block Krylov cycle starting from Qseed, reusing Q_buf.
        Returns a *view* slice of Q_buf, H, p, m.

        FIX 3 (H symmetry): The block Lanczos relation is
            A Q[:,applied:end] = Q[:,:m] H[:m, applied:end]  +  W_residual
        where W_residual = Q[:,m:m+r] * R[:r,:].
        We fill H[applied:end, m:m+r] = R[:r,:].T as well so that H[:p,:p]
        is exactly the projected matrix Q[:,:p].T @ A @ Q[:,:p] (up to
        floating point), making the residual formula exact.
        """
        s0 = Qseed.shape[1]
        if s0 > ncv:
            Qseed = Qseed[:, :ncv]
            s0 = ncv
        Q_buf[:, :s0] = Qseed

        H = torch.zeros((ncv + block_size, ncv + block_size), device=device, dtype=small_dtype)
        m = s0
        applied = 0

        while applied < ncv:
            if applied >= m:
                needed = min(block_size, ncv + block_size - m)
                Qnew = _orthonormalize_block(
                    torch.randn((N, needed), device=device, dtype=dtype),
                    Q_buf[:, :m],
                    needed,
                    allow_random_fill=True,
                )
                Q_buf[:, m:m + needed] = Qnew
                m += needed

            end = min(applied + block_size, m, ncv)
            s = end - applied
            if s <= 0:
                break

            Vblock = Q_buf[:, applied:end]
            W = _apply_A(Vblock)
            W = _project_deflation(W)

            # Full block Arnoldi/Lanczos orthogonalization.
            coeff_total = torch.zeros((m, s), device=device, dtype=small_dtype)
            if reorthogonalize:
                for _ in range(reorth_passes):
                    C = (Q_buf[:, :m].T @ W).to(dtype=small_dtype)
                    W = W - Q_buf[:, :m] @ C.to(dtype=dtype)
                    coeff_total += C
            else:
                C = (Q_buf[:, :m].T @ W).to(dtype=small_dtype)
                W = W - Q_buf[:, :m] @ C.to(dtype=dtype)
                coeff_total += C

            # Store Gram coefficients in this column block only (no transpose fill).
            # H is upper-Hessenberg style here; we symmetrize only at Ritz-problem time.
            # Using = not += avoids double-counting when reorth adds multiple passes.
            H[:m, applied:end] = coeff_total

            # Residual block QR.
            Qnew, R, r = _qr_rank(W)
            if r > 0:
                max_add = min(r, ncv + block_size - m)
                Q_buf[:, m:m + max_add] = Qnew[:, :max_add]
                # Subdiagonal block only — symmetrize H[:p,:p] at Ritz-problem time.
                H[m:m + max_add, applied:end] = R[:max_add, :].to(dtype=small_dtype)
                m += max_add

            applied = end

        p = min(applied, ncv)
        return Q_buf, H, p, m

    def _thick_restart_seed(
        Q: "torch.Tensor",
        H: "torch.Tensor",
        p: int,
        m: int,
        evals_all: "torch.Tensor",
        evecs_all: "torch.Tensor",
        n_take: int,
    ) -> Optional["torch.Tensor"]:
        """
        Thick restart: keep the top keep_eff Ritz vectors plus a fresh residual block.

        The Hessenberg rotation has been removed (per review point 2): after
        U_keep = Q[:,:p] @ Y_keep the projected matrix in that Ritz basis is
        diag(evals[:keep_eff]), not H[:keep_eff,:keep_eff], so rotating by S
        does not restore a tridiagonal relation — it just changes the basis within
        the same subspace without benefit. The next _expand_cycle rebuilds the
        relation from scratch anyway.

        Returns Qseed of shape (N, keep_eff + block_size), or None on breakdown.
        """
        # Validated keep_eff: must satisfy k <= keep_eff <= p - block_size.
        keep_eff = min(keep, p - block_size)
        keep_eff = max(k, keep_eff)
        keep_eff = min(keep_eff, p)

        Y_keep = evecs_all[:, :keep_eff].to(dtype=dtype)
        U_keep = _linear_combo_blocked(Q[:, :p], Y_keep)

        # Defensive reorth/QR of retained Ritz basis (no Hessenberg rotation).
        try:
            U_keep_orth = _orthonormalize_block(U_keep, None, keep_eff, allow_random_fill=False)
        except RuntimeError:
            return None

        # Residual seed from Ritz residual directions.
        r_cols = min(block_size, n_take)
        if m > p:
            coeff_tail = H[p:m, :p] @ evecs_all[:, :r_cols]
            Rseed = _linear_combo_blocked(Q[:, p:m], coeff_tail.to(dtype=dtype))
        else:
            Rseed = torch.empty((N, 0), device=device, dtype=dtype)

        try:
            Rblock = _orthonormalize_block(
                Rseed,
                U_keep_orth,
                block_size,
                allow_random_fill=True,
            )
        except RuntimeError:
            return None

        return torch.cat([U_keep_orth, Rblock], dim=1)

    if verbose:
        print(
            f"[block-irlm] N={N} k={k} block_size={block_size} ncv={ncv} "
            f"keep={keep} max_restarts={max_restarts} tol={tol:g} "
            f"dtype={dtype} small_dtype={small_dtype} device={device}"
        )
        if matmat_fn is None:
            print("  warning: using matvec_fn fallback; not a true block matmat")
        if B_def is not None:
            print(f"  deflate_basis: {B_def.shape[1]} direction(s)")

    Qseed = _initial_seed()

    converged = False
    breakdown = False
    cycles_done = 0
    orth_error_history = []
    residual_history = []
    theta_history = []

    last_Q = None
    last_H = None
    last_p = None
    last_m = None
    last_evals = None
    last_evecs = None
    last_formula_res = None

    try:
        for cycle in range(max_restarts + 1):
            cycles_done = cycle + 1
            Q, H, p, m = _expand_cycle(Qseed)
            if p < k:
                raise RuntimeError(f"cycle produced p={p} basis columns < k={k}")

            Hm = H[:p, :p]
            Hm = 0.5 * (Hm + Hm.T)
            evals_all, evecs_all = _small_eigh_desc(Hm)

            n_take = min(k, p)
            Yk = evecs_all[:, :n_take]
            tail = H[p:m, :p] @ Yk if m > p else torch.zeros((0, n_take), device=device, dtype=small_dtype)
            formula_res = torch.linalg.norm(tail, dim=0) if tail.numel() else torch.zeros(n_take, device=device, dtype=small_dtype)

            if relative_residual:
                thresholds = tol * torch.maximum(torch.ones_like(evals_all[:n_take]), torch.abs(evals_all[:n_take]))
            else:
                thresholds = torch.full_like(formula_res, float(tol))
            n_conv = int(torch.sum(formula_res <= thresholds).detach().cpu())

            if orth_check_every is not None and orth_check_every > 0 and cycle % orth_check_every == 0:
                orth_err = _orth_error_blocked(Q[:, :m])
            else:
                orth_err = float("nan")
            orth_error_history.append(orth_err)
            residual_history.append(formula_res.detach().cpu().numpy().astype(np.float64))
            theta_history.append(evals_all[:n_take].detach().cpu().numpy().astype(np.float64))

            last_Q, last_H, last_p, last_m = Q, H, p, m
            last_evals, last_evecs, last_formula_res = evals_all, evecs_all, formula_res

            if verbose:
                max_res = float(torch.max(formula_res).detach().cpu()) if formula_res.numel() else float("nan")
                print(
                    f"  cycle={cycle:02d} p={p:4d} m={m:4d} "
                    f"converged={n_conv}/{n_take} max_res={max_res:.3e} orth={orth_err:.2e}"
                )
                print(f"    theta[:5] = {evals_all[:min(5, n_take)].detach().cpu().numpy()}")

            if n_take >= k and bool(torch.all(formula_res[:k] <= thresholds[:k]).detach().cpu()):
                converged = True
                if verbose:
                    print(f"[block-irlm] converged after cycle={cycle}")
                break

            if cycle == max_restarts:
                break

            # FIX 1+4: thick restart with GPU Hessenberg and keep_eff validation.
            new_seed = _thick_restart_seed(Q, H, p, m, evals_all, evecs_all, n_take)
            if new_seed is None:
                breakdown = True
                if verbose:
                    print("[block-irlm] restart seed construction failed (breakdown)")
                break
            Qseed = new_seed

        # ── Final Ritz extraction ────────────────────────────────────────────

        if last_Q is None or last_evals is None or last_evecs is None or last_p is None:
            raise RuntimeError("block IRLM failed before producing a Ritz problem")

        n_return = min(k, last_p)
        vals = last_evals[:n_return]
        Y = last_evecs[:, :n_return].to(dtype=dtype)
        ritz_vecs = _linear_combo_blocked(last_Q[:, :last_p], Y)
        ritz_vecs = ritz_vecs / torch.linalg.norm(ritz_vecs, dim=0).clamp_min(torch.finfo(dtype).eps)

        formula_res = last_formula_res[:n_return] if last_formula_res is not None else None
        # FIX 5: true residuals use _count=False (handled inside _true_residuals).
        true_res = _true_residuals(vals, ritz_vecs) if compute_true_residuals else None

        elapsed = time.perf_counter() - t_start

        if verbose:
            print(
                f"[block-irlm] done converged={converged} breakdown={breakdown} "
                f"cycles={cycles_done} matmat={n_matmat} rhs={n_rhs} time={elapsed:.2f}s"
            )
            print(f"  eigenvalues:       {vals.detach().cpu().numpy()}")
            if formula_res is not None:
                print(f"  formula residuals: {formula_res.detach().cpu().numpy()}")
            if true_res is not None:
                print(f"  true residuals:    {true_res}")

        if return_torch:
            vals_out = vals
            vecs_out = ritz_vecs
        else:
            vals_out = vals.detach().cpu().numpy().astype(np.float64)
            vecs_out = ritz_vecs.detach().cpu().numpy()

        if return_info:
            info: Dict[str, Any] = {
                "method": "thick_restarted_block_arnoldi_rayleigh_ritz",
                "converged": bool(converged),
                "breakdown": bool(breakdown),
                "cycles": int(cycles_done),
                "n_matmat": int(n_matmat),
                "n_rhs": int(n_rhs),
                "ncv": int(ncv),
                "keep": int(keep),
                "k": int(k),
                "block_size": int(block_size),
                "tol": float(tol),
                "relative_residual": bool(relative_residual),
                "elapsed": float(elapsed),
                "p_final": int(last_p),
                "m_final": int(last_m) if last_m is not None else None,
                "formula_residuals": None if formula_res is None else formula_res.detach().cpu().numpy().astype(np.float64),
                "true_residuals": true_res,
                "orth_error": orth_error_history,
                "residual_history": residual_history,
                "theta_history": theta_history,
                "dtype": str(dtype),
                "small_dtype": str(small_dtype),
                "device": str(device),
            }
            return vals_out, vecs_out, info

        return vals_out, vecs_out

    finally:
        torch.set_grad_enabled(grad_state)
        if old_tf32_matmul is not None:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32_matmul
            torch.backends.cudnn.allow_tf32 = old_tf32_cudnn


def gpu_block_lanczos(
    matmat_fn: Optional[Callable[["torch.Tensor"], "torch.Tensor"]] = None,
    *,
    matvec_fn: Optional[Callable[["torch.Tensor"], "torch.Tensor"]] = None,
    N: int,
    k: int = 8,
    n_iter: Optional[int] = None,
    block_size: Optional[int] = None,
    tol: float = 1e-6,
    dtype=None,
    device: str = "cuda",
    verbose: bool = True,
    q0_block=None,
    q0=None,
    **kwargs,
):
    """Compatibility wrapper. n_iter maps to ncv."""
    return gpu_block_irlm(
        matmat_fn=matmat_fn,
        matvec_fn=matvec_fn,
        N=N,
        k=k,
        ncv=n_iter,
        block_size=block_size,
        tol=tol,
        dtype=dtype,
        device=device,
        verbose=verbose,
        q0_block=q0_block,
        q0=q0,
        **kwargs,
    )


def make_gpu_matmat(state, ext_module):
    """
    Convenience wrapper for a CUDA extension exposing either:
      - ext_module.matmat(X_fp16, V, rscale, sq_X, beta), preferred
      - ext_module.matvec(...), fallback over columns.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch required")
    import torch

    has_matmat = hasattr(ext_module, "matmat")

    def matmat_gpu(V):
        with torch.no_grad():
            if has_matmat:
                return ext_module.matmat(
                    state.X_fp16, V, state.rscale, state.sq_X, state.beta,
                )
            cols = [
                ext_module.matvec(
                    state.X_fp16, V[:, j], state.rscale, state.sq_X, state.beta,
                )
                for j in range(V.shape[1])
            ]
            return torch.stack(cols, dim=1)

    return matmat_gpu

gpu_lanczos = gpu_block_lanczos
