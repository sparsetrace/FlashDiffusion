def gpu_lanczos(
    matvec_fn,
    N:        int,
    k:        int   = 8,
    n_iter:   int   = None,
    tol:      float = 1e-6,
    dtype           = None,
    device:   str   = "cuda",
    verbose:  bool  = True,
    q0=None,
    check_every: int = 5,
    min_iter: int | None = None,
    breakdown_tol: float = 1e-12,
    relative_residual: bool = True,
    return_info: bool = False,
) -> tuple:
    """
    GPU Lanczos with full re-orthogonalization and Ritz residual convergence checks.

    This is still plain non-restarted Lanczos, but now `tol` is active:
    the solver stops early once the top-k Ritz residuals satisfy

        residual_i = |beta_m * y_i[-1]|

    where y_i is the eigenvector of the small tridiagonal T_m.

    Parameters
    ----------
    matvec_fn : callable
        (N,) GPU tensor -> (N,) GPU tensor.
    N : int
        Problem size.
    k : int
        Number of largest eigenpairs wanted.
    n_iter : int
        Maximum Lanczos steps. If None, uses min(6*k + 20, N).
    tol : float
        Convergence tolerance for Ritz residuals.
    dtype : torch.dtype
        Default float32.
    device : str
        CUDA device.
    verbose : bool
        Print progress.
    q0 : torch.Tensor or None
        Optional initial vector. Warning: if q0 is exactly an eigenvector and k > 1,
        Lanczos may break down after one step.
    check_every : int
        Check Ritz residuals every this many iterations.
    min_iter : int or None
        Do not check convergence before this many steps.
        Default: min(max(2*k + 5, k + 2), n_iter).
    breakdown_tol : float
        Stop if beta falls below this value.
    relative_residual : bool
        If True, use residual <= tol * max(1, |theta|).
        If False, use residual <= tol.
    return_info : bool
        If True, return (evals, evecs, info).

    Returns
    -------
    vals : np.ndarray, shape <= (k,)
        Ritz eigenvalues in descending order.
    vecs : np.ndarray, shape (N, len(vals))
        Ritz vectors.
    info : dict, optional
        Returned only when return_info=True.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch required for gpu_lanczos")

    import torch

    if dtype is None:
        dtype = torch.float32

    if n_iter is None:
        n_iter = min(6 * k + 20, N)

    n_iter = int(min(n_iter, N))

    if min_iter is None:
        min_iter = min(max(2 * k + 5, k + 2), n_iter)
    else:
        min_iter = int(min(min_iter, n_iter))

    check_every = max(1, int(check_every))

    t_start = time.perf_counter()

    if verbose:
        print(
            f"[gpu_lanczos] N={N}  k={k}  max_iter={n_iter}  "
            f"min_iter={min_iter}  tol={tol:g}  dtype={dtype}"
        )

    # Lanczos basis and recurrence coefficients.
    Q      = torch.zeros(N, n_iter, dtype=dtype, device=device)
    alphas = torch.zeros(n_iter,    dtype=dtype, device=device)
    betas  = torch.zeros(n_iter,    dtype=dtype, device=device)

    # Initial vector.
    torch.manual_seed(42)

    if q0 is not None:
        q = q0.to(device=device, dtype=dtype).reshape(-1)
        if q.numel() != N:
            raise ValueError(f"q0 has length {q.numel()}, expected N={N}")
    else:
        # Random start is safer than all-ones if the constant vector is exactly
        # the leading eigenvector. All-ones can cause lucky breakdown at m=1.
        q = torch.randn(N, dtype=dtype, device=device)

    q_norm = q.norm()
    if q_norm.item() == 0:
        raise ValueError("Initial vector q0 has zero norm.")

    q = q / q_norm
    Q[:, 0] = q

    converged = False
    breakdown = False
    j_final = n_iter

    final_evals = None
    final_evecs_small = None
    final_residuals = None

    def solve_small_tridiagonal(m: int):
        """
        Solve T_m eigenproblem on CPU using NumPy.
        T_m is tiny, so CPU transfer is negligible and robust.
        Returns eigenvalues descending and eigenvectors in matching order.
        """
        T = torch.diag(alphas[:m])
        if m > 1:
            T = T + torch.diag(betas[:m - 1], 1)
            T = T + torch.diag(betas[:m - 1], -1)

        T_np = T.detach().cpu().numpy().astype(np.float64)
        evals_np, evecs_np = np.linalg.eigh(T_np)  # ascending

        idx = np.argsort(evals_np)[::-1]
        return evals_np[idx], evecs_np[:, idx]

    for j in range(n_iter):
        # Matvec.
        z = matvec_fn(Q[:, j])
        z = z.to(device=device, dtype=dtype).reshape(-1)

        # Three-term recurrence.
        alphas[j] = torch.dot(Q[:, j], z)
        z = z - alphas[j] * Q[:, j]

        if j > 0:
            z = z - betas[j - 1] * Q[:, j - 1]

        # Full double-pass MGS re-orthogonalization.
        # Use Q[:, :j+1] because current vector Q[:, j] is already filled.
        for _ in range(2):
            coeffs = Q[:, :j + 1].T @ z
            z = z - Q[:, :j + 1] @ coeffs

        beta_j_tensor = z.norm()
        beta_j = float(beta_j_tensor.detach().cpu())
        betas[j] = beta_j

        m = j + 1

        should_check = (
            m >= min_iter
            and (
                m == n_iter
                or m % check_every == 0
                or beta_j < breakdown_tol
            )
        )

        if should_check:
            evals_all, evecs_all = solve_small_tridiagonal(m)
            n_take = min(k, len(evals_all))

            evals_k = evals_all[:n_take]
            evecs_k = evecs_all[:, :n_take]

            # Lanczos Ritz residual:
            # ||A(Qy) - theta(Qy)|| ~= |beta_m * y[-1]|
            residuals = np.abs(beta_j * evecs_k[-1, :])

            if relative_residual:
                thresholds = tol * np.maximum(1.0, np.abs(evals_k))
            else:
                thresholds = np.full_like(residuals, tol, dtype=np.float64)

            n_conv = int(np.sum(residuals <= thresholds))

            if verbose:
                max_res = float(np.max(residuals)) if len(residuals) else float("nan")
                print(
                    f"  iter={m:4d}  beta={beta_j:.3e}  "
                    f"converged={n_conv}/{n_take}  max_res={max_res:.3e}"
                )

            final_evals = evals_all
            final_evecs_small = evecs_all
            final_residuals = residuals

            if n_take >= k and np.all(residuals[:k] <= thresholds[:k]):
                converged = True
                j_final = m
                if verbose:
                    print(f"[gpu_lanczos] converged at iter={m}")
                break

        # Lucky breakdown / invariant subspace exhausted.
        if beta_j < breakdown_tol:
            breakdown = True
            j_final = m
            if verbose:
                print(f"[gpu_lanczos] breakdown at iter={m}, beta={beta_j:.3e}")
            break

        # Store next basis vector.
        if j < n_iter - 1:
            Q[:, j + 1] = z / beta_j_tensor

    # Final solve if we never checked, or if the latest check was not final.
    m = j_final if (converged or breakdown) else n_iter

    if final_evals is None or final_evecs_small is None or final_evecs_small.shape[0] != m:
        final_evals, final_evecs_small = solve_small_tridiagonal(m)

        # Compute residuals for final reported Ritz pairs.
        n_take_tmp = min(k, len(final_evals))
        beta_m = float(betas[m - 1].detach().cpu()) if m > 0 else 0.0
        final_residuals = np.abs(beta_m * final_evecs_small[-1, :n_take_tmp])

    # Sort descending already done.
    n_return = min(k, len(final_evals), m)
    evals_k = final_evals[:n_return]
    evecs_k = final_evecs_small[:, :n_return]

    # Ritz vectors: Q[:, :m] @ evecs_k.
    evecs_k_gpu = torch.from_numpy(evecs_k.astype(np.float32)).to(device=device)
    ritz_vecs = Q[:, :m] @ evecs_k_gpu

    elapsed = time.perf_counter() - t_start

    if verbose:
        print(f"[gpu_lanczos] finished: m={m}  converged={converged}  time={elapsed:.2f}s")
        print(f"  eigenvalues: {evals_k}")
        if final_residuals is not None:
            print(f"  residuals:   {final_residuals[:n_return]}")

    vals_out = evals_k.astype(np.float64)
    vecs_out = ritz_vecs.detach().cpu().numpy()

    if return_info:
        info = {
            "converged": bool(converged),
            "breakdown": bool(breakdown),
            "n_iter_used": int(m),
            "n_iter_max": int(n_iter),
            "tol": float(tol),
            "relative_residual": bool(relative_residual),
            "residuals": None if final_residuals is None else final_residuals[:n_return].astype(np.float64),
            "elapsed": float(elapsed),
        }
        return vals_out, vecs_out, info

    return vals_out, vecs_out
