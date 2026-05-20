"""
ldm_matmat.py
=============
Linearized Diffusion Map (LDM) and 2nd-order LDM matmat operators
for use with gpu_block_irlm.

These are the O(ND) and O(ND^2) warm-start operators for full DMAP.
Both expose the matmat_fn(V) -> A@V interface expected by gpu_block_irlm,
so block_size can be anything and GPU throughput is fully used.

Hierarchy
---------
  Stage 1:  LDM       O(NDs)   -- linear in N and D
  Stage 2:  2nd-order O(ND^2)  -- one-time O(ND^2) setup, then O(NDs)/matmat
  Stage 3:  full DMAP O(N^2 D) -- your existing CUDA exp kernel

Usage
-----
    from ldm_matmat import LDMOperator, LDM2Operator, ldm_warm_start

    # build operators (precomputes potentials once)
    ldm  = LDMOperator(X, beta, alpha=1.0, device="cuda")
    ldm2 = LDM2Operator(X, beta, alpha=1.0, device="cuda")

    # run block IRLM for warm start
    vals0, U0 = gpu_block_irlm(matmat_fn=ldm.matmat,  N=N, k=k, ...)
    vals1, U1 = gpu_block_irlm(matmat_fn=ldm2.matmat, N=N, k=k, q0_block=U0, ...)

    # full DMAP warm-started
    vals,  V  = gpu_block_irlm(matmat_fn=dmap_matmat,  N=N, k=k, q0_block=U1, ...)
"""

from __future__ import annotations
from typing import Optional
import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Shared potential precomputation
# ─────────────────────────────────────────────────────────────────────────────

def _precompute_potentials(
    X: "torch.Tensor",
    beta: float,
    eps: float,
    alpha: float,
    device: "torch.device",
    dtype: "torch.dtype",
    verbose: bool,
) -> dict:
    """
    Precompute all four potentials mu, r, q, m from centered data.
    All O(ND) or O(N). Returns dict with tensors on device.

    The combined potential v_hat absorbs:
      v  = exp(-beta * r / eps)          [from kernel factorization]
      q  = CL normalization (alpha)      [O(N) closed form after centering]
      m  = Markov symmetrization         [O(ND) one matrix-vector product]
    """
    import torch

    N, D = X.shape
    t0 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    t1 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    if t0: t0.record()

    # ── 1. Center ────────────────────────────────────────────────────────────
    mu   = X.mean(dim=0)                   # (D,)
    Rb   = X - mu.unsqueeze(0)             # (N, D)  centered, Rb^T 1 = 0

    # ── 2. Squared norms r = diag(Rb Rb^T) ───────────────────────────────────
    r    = (Rb * Rb).sum(dim=1)            # (N,)  O(ND)
    S_r  = r.sum()                         # scalar = tr(Rb Rb^T)

    # ── 3. Kernel potential v = exp(-beta r / eps) ────────────────────────────
    v    = torch.exp(-beta * r / eps)      # (N,)

    # ── 4. CL normalization q  (O(N) closed form, centering killed cross-term)─
    #   k_i = N(1 - beta*r_i/eps) - beta*S_r/eps
    #   q_i = k_i^{-1}   (alpha=1 shown; general alpha below)
    k_lin = N * (1.0 - beta * r / eps) - (beta / eps) * S_r   # (N,)
    # For general alpha: q = k_lin^{-alpha}
    if abs(alpha - 1.0) < 1e-9:
        q = k_lin.pow(-1.0)
    else:
        q = k_lin.pow(-alpha)

    # ── 5. Markov symmetrization m  (O(ND) one matmul) ───────────────────────
    #   S_q  = 1^T q
    #   S_rq = r^T q
    #   c    = Rb^T q  in R^D
    #   k_hat_i = q_i * [S_q*(1 - beta*r_i/eps) - beta*S_rq/eps + 2*beta/eps * Rb_i . c]
    #   m_i  = k_hat_i^{-1/2}
    S_q  = q.sum()                         # scalar
    S_rq = (r * q).sum()                   # scalar
    c    = Rb.T @ q                        # (D,)  O(ND)

    Rbc  = Rb @ c                          # (N,)  O(ND)  -- Rb_i . c
    k_hat = q * (
        S_q * (1.0 - beta * r / eps)
        - (beta / eps) * S_rq
        + (2.0 * beta / eps) * Rbc
    )                                       # (N,)
    m    = k_hat.pow(-0.5)                 # (N,)

    # ── 6. Combined potential ─────────────────────────────────────────────────
    # For LDM: v_hat = m * q  (exp(-beta r) absorbed into q/m in linear degree)
    # For DMAP: v_hat = m * q * exp(-beta r / eps)
    v_hat_ldm  = m * q                     # (N,)
    v_hat_dmap = m * q * v                 # (N,)

    if t1:
        t1.record()
        torch.cuda.synchronize()
        elapsed = t0.elapsed_time(t1) / 1000.0
    else:
        elapsed = 0.0

    if verbose:
        print(f"[LDM precompute] N={N} D={D} beta={beta:.4g} eps={eps:.4g} alpha={alpha:.4g}")
        print(f"  S_r={float(S_r):.4g}  S_q={float(S_q):.4g}  elapsed={elapsed:.3f}s")
        print(f"  r   : min={float(r.min()):.4g} max={float(r.max()):.4g}")
        print(f"  v   : min={float(v.min()):.4g} max={float(v.max()):.4g}")
        print(f"  k_lin: min={float(k_lin.min()):.4g} max={float(k_lin.max()):.4g}")
        print(f"  k_hat: min={float(k_hat.min()):.4g} max={float(k_hat.max()):.4g}")

    return dict(
        Rb=Rb, mu=mu, r=r, v=v, q=q, m=m,
        v_hat_ldm=v_hat_ldm,
        v_hat_dmap=v_hat_dmap,
        k_lin=k_lin, k_hat=k_hat,
        S_r=S_r, S_q=S_q, S_rq=S_rq, c=c,
        N=N, D=D, beta=beta, eps=eps, alpha=alpha,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LDM Operator  O(NDs) per matmat
# ─────────────────────────────────────────────────────────────────────────────

class LDMOperator:
    """
    K^LDM U = diag(v_hat) [1 1^T + (2*beta/eps) Rb Rb^T] diag(v_hat) U

    Cost per matmat: O(NDs)  -- two GEMMs with Rb (shape N x D).
    The rank-1 term 1 1^T should be deflated via deflate_basis in gpu_block_irlm.

    Parameters
    ----------
    X : array-like (N, D)
        Raw (uncentered) data. Will be centered internally.
    beta, eps, alpha : float
        Kernel bandwidth, lengthscale, CL exponent.
    device, dtype : torch device/dtype.
    """

    def __init__(
        self,
        X,
        beta: float,
        eps: Optional[float] = None,
        alpha: float = 1.0,
        device: str = "cuda",
        dtype=None,
        verbose: bool = True,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError("torch required")
        import torch

        self.device = torch.device(device)
        self.dtype  = dtype or torch.float32

        Xt = torch.as_tensor(
            X if isinstance(X, torch.Tensor) else np.asarray(X, dtype=np.float32),
            device=self.device, dtype=self.dtype,
        )
        N, D = Xt.shape
        # Default eps: use 4 * mean(r) so linearization is valid
        if eps is None:
            r_tmp = (Xt * Xt).sum(dim=1)
            mu_tmp = Xt.mean(dim=0)
            Rb_tmp = Xt - mu_tmp
            r_tmp  = (Rb_tmp * Rb_tmp).sum(dim=1)
            eps = float(4.0 * r_tmp.mean())
            if verbose:
                print(f"[LDMOperator] auto eps={eps:.4g} (4 * mean(r))")

        self.eps = eps
        self.pot = _precompute_potentials(
            Xt, beta, eps, alpha, self.device, self.dtype, verbose,
        )
        self.N = N
        self.D = D

        # Precompute the trivial-mode vector for deflation reference
        # (caller should pass this as deflate_basis to gpu_block_irlm)
        self.trivial_mode = (self.pot["v_hat_ldm"] /
                             self.pot["v_hat_ldm"].norm()).unsqueeze(1)  # (N,1)

    def matmat(self, V: "torch.Tensor") -> "torch.Tensor":
        """
        V : (N, s)  ->  K^LDM V : (N, s)

        Expanding:
          W      = diag(v_hat) V                   O(Ns)
          rank1  = diag(v_hat) 1 (1^T W)           O(Ns)   [trivial mode]
          gramv  = diag(v_hat) Rb (Rb^T W)         O(NDs)  [two GEMMs]
          output = rank1 + (2*beta/eps) * gramv
        """
        import torch
        pot   = self.pot
        vh    = pot["v_hat_ldm"]          # (N,)
        Rb    = pot["Rb"]                 # (N, D)
        beta  = pot["beta"]
        eps   = pot["eps"]

        W     = vh.unsqueeze(1) * V       # diag(v_hat) V,   (N, s)
        rank1 = vh.unsqueeze(1) * W.sum(dim=0, keepdim=True)   # (N, s)  broadcast
        gramv = vh.unsqueeze(1) * (Rb @ (Rb.T @ W))            # (N, s)
        return rank1 + (2.0 * beta / eps) * gramv

    @property
    def deflate_vector(self) -> "torch.Tensor":
        """
        Return the trivial diffusion mode (N,1) for use as deflate_basis.
        Pass this to gpu_block_irlm to kill the lambda=1 mode.
        """
        return self.trivial_mode


# ─────────────────────────────────────────────────────────────────────────────
# 2nd-Order LDM Operator  O(ND^2) setup + O(NDs) per matmat
# ─────────────────────────────────────────────────────────────────────────────

class LDM2Operator:
    """
    K^(2) U = diag(v_hat) [
        1 1^T
      + (2*beta/eps) Rb Rb^T
      + (2*beta^2/eps^2) (Rb Rb^T)^{o2}
    ] diag(v_hat) U

    The n=2 Hadamard-power term is computed via:
      A = Rb^T diag(u) Rb  in R^{D x D}   (precomputed once, O(ND^2))
      [( Rb Rb^T )^{o2} (diag(v_hat) V)]_i = (Rb_i . (A * Z_i)) summed

    where Z = Rb^T (diag(v_hat) V) in R^{D x s}, and A*Z is elementwise
    (Hadamard) between A (D x D) broadcast over s columns.

    Cost: O(ND^2) once at init, then O(NDs) per matmat call.
    For D=3 (Swiss roll) this is trivially cheap.
    For D=784 (MNIST) A is 784x784 ~ 5MB, still fine.
    """

    def __init__(
        self,
        X,
        beta: float,
        eps: Optional[float] = None,
        alpha: float = 1.0,
        device: str = "cuda",
        dtype=None,
        verbose: bool = True,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError("torch required")
        import torch

        self.device = torch.device(device)
        self.dtype  = dtype or torch.float32

        Xt = torch.as_tensor(
            X if isinstance(X, torch.Tensor) else np.asarray(X, dtype=np.float32),
            device=self.device, dtype=self.dtype,
        )
        N, D = Xt.shape

        if eps is None:
            r_tmp  = (Xt * Xt).sum(dim=1)
            mu_tmp = Xt.mean(dim=0)
            Rb_tmp = Xt - mu_tmp
            r_tmp  = (Rb_tmp * Rb_tmp).sum(dim=1)
            eps    = float(4.0 * r_tmp.mean())
            if verbose:
                print(f"[LDM2Operator] auto eps={eps:.4g} (4 * mean(r))")

        self.eps = eps
        self.pot = _precompute_potentials(
            Xt, beta, eps, alpha, self.device, self.dtype, verbose,
        )
        self.N = N
        self.D = D

        # ── Precompute A = Rb^T diag(u) Rb  in R^{D x D}  O(ND^2) ──────────
        # u here is the kernel potential v = exp(-beta r / eps)
        Rb   = self.pot["Rb"]             # (N, D)
        u    = self.pot["v"]              # (N,)  = exp(-beta r / eps)
        # A_{xy} = sum_i u_i Rb_{ix} Rb_{iy}
        Rb_u = Rb * u.unsqueeze(1)        # (N, D)  diag(u) Rb
        self.A = Rb_u.T @ Rb             # (D, D)  O(ND^2)

        if verbose:
            print(f"[LDM2Operator] A precomputed: shape={tuple(self.A.shape)} "
                  f"norm={float(self.A.norm()):.4g}")

        self.trivial_mode = (self.pot["v_hat_ldm"] /
                             self.pot["v_hat_ldm"].norm()).unsqueeze(1)

    def matmat(self, V: "torch.Tensor") -> "torch.Tensor":
        """
        V : (N, s)  ->  K^(2) V : (N, s)

        Steps:
          W      = diag(v_hat) V                              O(Ns)
          rank1  = diag(v_hat) 1 (1^T W)                     O(Ns)
          Z      = Rb^T W                                     O(NDs)  [GEMM1]
          n1term = diag(v_hat) Rb Z                           O(NDs)  [GEMM2]
          n2term = diag(v_hat) Rb (A * Z)                    O(D^2 s + NDs)
          output = rank1
                 + (2b/eps) n1term
                 + (2b^2/eps^2) n2term
        """
        import torch
        pot   = self.pot
        vh    = pot["v_hat_ldm"]          # (N,)
        Rb    = pot["Rb"]                 # (N, D)
        beta  = pot["beta"]
        eps   = pot["eps"]
        A     = self.A                    # (D, D)

        W     = vh.unsqueeze(1) * V                           # (N, s)
        rank1 = vh.unsqueeze(1) * W.sum(dim=0, keepdim=True) # (N, s)
        Z     = Rb.T @ W                                      # (D, s) GEMM1

        # n=1 term: diag(v_hat) Rb Z
        n1    = vh.unsqueeze(1) * (Rb @ Z)                   # (N, s) GEMM2

        # n=2 term: diag(v_hat) Rb (A odot Z)
        # A: (D, D), Z: (D, s) -> A*Z is (D,s) where col j of Z is scaled by A
        # Correct: for each sample i, contribution is Rb_i . (A * (Rb^T W)_i)
        # = Rb_i . (A @ w_col) -- but A is symmetric so A * Z means A @ Z here
        # The Hadamard-power identity: (Rb Rb^T)^{o2} u = Rb (A o (Rb^T u))
        # where o here is column-wise: each column z of Z gives A * z,
        # meaning the element-wise product of matrix A broadcast -- but since
        # we want sum_j (Rb_i.Rb_j)^2 u_j = Rb_i . (Rb^T (diag(Rb Rb^T u)))
        # = sum_x Rb_{ix} sum_y A_{xy} Z_{ys}  -- this IS A @ Z (matrix mult)
        AZ    = A @ Z                                         # (D, s) O(D^2 s)
        n2    = vh.unsqueeze(1) * (Rb @ AZ)                  # (N, s) GEMM3

        c1 = 2.0 * beta / eps
        c2 = 2.0 * beta**2 / eps**2

        return rank1 + c1 * n1 + c2 * n2

    @property
    def deflate_vector(self) -> "torch.Tensor":
        return self.trivial_mode


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: three-stage warm-start pipeline
# ─────────────────────────────────────────────────────────────────────────────

def ldm_warm_start(
    X,
    k: int,
    beta: float,
    full_matmat_fn,
    *,
    eps: Optional[float] = None,
    alpha: float = 1.0,
    device: str = "cuda",
    dtype=None,
    block_size: int = 16,
    tol_ldm: float = 1e-3,
    tol_ldm2: float = 1e-4,
    tol_full: float = 1e-6,
    max_restarts_ldm: int = 5,
    max_restarts_ldm2: int = 8,
    max_restarts_full: int = 20,
    skip_ldm2: bool = False,
    verbose: bool = True,
):
    """
    Three-stage block Lanczos warm start:

      Stage 1: LDM       O(NDs)   -> U0
      Stage 2: 2nd-order O(ND^2)  -> U1  (skip if D large or skip_ldm2=True)
      Stage 3: full DMAP O(N^2 D) -> final eigenpairs

    Parameters
    ----------
    X : array-like (N, D)
    k : int   number of eigenpairs
    beta : float   kernel bandwidth
    full_matmat_fn : callable V -> A@V  your CUDA exp kernel
    eps : float or None   lengthscale (auto if None)
    alpha : float   CL exponent
    block_size : int   Lanczos block size (use >= k, powers of 2 work well)
    tol_* : float   convergence tolerances per stage
    skip_ldm2 : bool   skip 2nd-order stage (e.g. for large D)

    Returns
    -------
    vals : np.ndarray (k,)
    vecs : np.ndarray (N, k)
    info : dict with per-stage diagnostics
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch required")
    import torch
    import time
    # import here to avoid circular if in same package
    from eigensolver_gpu import gpu_block_irlm

    import numpy as np

    device_ = torch.device(device)
    dtype_  = dtype or torch.float32

    Xt = torch.as_tensor(
        X if isinstance(X, torch.Tensor) else np.asarray(X, dtype=np.float32),
        device=device_, dtype=dtype_,
    )
    N  = Xt.shape[0]

    info = {}

    # ── Stage 1: LDM ─────────────────────────────────────────────────────────
    if verbose:
        print("\n[ldm_warm_start] Stage 1: LDM  O(NDs)")
    t0   = time.perf_counter()
    ldm  = LDMOperator(Xt, beta=beta, eps=eps, alpha=alpha,
                       device=device, dtype=dtype_, verbose=verbose)
    eps  = ldm.eps   # propagate auto-eps to stage 2

    vals0, U0, info0 = gpu_block_irlm(
        matmat_fn      = ldm.matmat,
        N              = N,
        k              = k,
        block_size     = block_size,
        tol            = tol_ldm,
        max_restarts   = max_restarts_ldm,
        deflate_basis  = ldm.deflate_vector,
        compute_true_residuals = False,
        return_info    = True,
        return_torch   = True,
        verbose        = verbose,
    )
    info["stage1"] = info0
    info["stage1"]["elapsed_total"] = time.perf_counter() - t0
    if verbose:
        print(f"  Stage 1 done: {info0['cycles']} cycles, "
              f"{info0['n_matmat']} matmats, "
              f"{info['stage1']['elapsed_total']:.2f}s")

    U_seed = U0

    # ── Stage 2: 2nd-order LDM ───────────────────────────────────────────────
    if not skip_ldm2:
        if verbose:
            print(f"\n[ldm_warm_start] Stage 2: 2nd-order LDM  O(ND^2), D={ldm.D}")
        t0    = time.perf_counter()
        ldm2  = LDM2Operator(Xt, beta=beta, eps=eps, alpha=alpha,
                             device=device, dtype=dtype_, verbose=verbose)

        vals1, U1, info1 = gpu_block_irlm(
            matmat_fn      = ldm2.matmat,
            N              = N,
            k              = k,
            block_size     = block_size,
            tol            = tol_ldm2,
            max_restarts   = max_restarts_ldm2,
            q0_block       = U_seed,
            deflate_basis  = ldm2.deflate_vector,
            compute_true_residuals = False,
            return_info    = True,
            return_torch   = True,
            verbose        = verbose,
        )
        info["stage2"] = info1
        info["stage2"]["elapsed_total"] = time.perf_counter() - t0
        if verbose:
            print(f"  Stage 2 done: {info1['cycles']} cycles, "
                  f"{info1['n_matmat']} matmats, "
                  f"{info['stage2']['elapsed_total']:.2f}s")
        U_seed = U1

    # ── Stage 3: full DMAP ───────────────────────────────────────────────────
    if verbose:
        print("\n[ldm_warm_start] Stage 3: full DMAP  O(N^2 D)")
    t0 = time.perf_counter()

    vals, vecs, info3 = gpu_block_irlm(
        matmat_fn      = full_matmat_fn,
        N              = N,
        k              = k,
        block_size     = block_size,
        tol            = tol_full,
        max_restarts   = max_restarts_full,
        q0_block       = U_seed,
        compute_true_residuals = True,
        return_info    = True,
        return_torch   = False,
        verbose        = verbose,
    )
    info["stage3"] = info3
    info["stage3"]["elapsed_total"] = time.perf_counter() - t0
    if verbose:
        print(f"  Stage 3 done: {info3['cycles']} cycles, "
              f"{info3['n_matmat']} matmats, "
              f"{info['stage3']['elapsed_total']:.2f}s")

    return vals, vecs, info
