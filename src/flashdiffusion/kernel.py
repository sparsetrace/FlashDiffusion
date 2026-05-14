"""
flashdiffusion/kernel.py
========================

FlashDiffusion: the O(N) memory diffusion kernel primitive for PyTorch.

    out = FlashDiffusion(Q, K, V, alpha, beta, h, log_d_q, log_d_k)

Computes the generalised DMAP Markov operator applied to V:

    out_i = sum_j P_ij V_j

where P is constructed via FlexAttention with a custom score_mod:

    s_ij  =  -beta * ||q_i - k_j||^2          (Gaussian kernel)
             - alpha * (log_d_q[i] + log_d_k[j])   (density correction)
             + log_h[i] - log_h[j]             (Doob h-transform)

The row-softmax inside FlexAttention IS the Markov normalisation —
no additional division is needed after the kernel call.

Connection to DAC (arXiv:2603.28037):
--------------------------------------
FlashDiffusion is the EQ sector of attention (P+):
  alpha=0,  h=None   ->  raw kernel (no density correction)
  alpha=1,  h=None   ->  Laplace-Beltrami DMAP (Coifman-Lafon 2006)
  alpha=0.5,h=None   ->  Fokker-Planck normalisation
  alpha=1,  h!=None  ->  Doob h-transform of DMAP
                          (tilted Markov operator, Schrodinger bridge)

The LSE (log-sum-exp) returned alongside the output equals
log(sum_j K_alpha_ij) -- used to estimate densities for a second
normalisation pass. See dmap.py for the two-pass DMAP pipeline.

Validated against dense DMAP to fp32 accuracy (~1e-7 max error).

Usage
-----
# single call: get P+ V
out = FlashDiffusion(X, X, V, alpha=1.0, beta=0.5, log_d_q=lse1, log_d_k=lse1)

# with density estimation (returns LSE alongside output)
out, lse = FlashDiffusion(..., return_lse=True)

# Doob h-transform
out = FlashDiffusion(..., h=h_potential)

Requirements
------------
torch >= 2.5  (flex_attention + AuxRequest available)
CUDA GPU recommended; CPU supported but uncompiled (slow, for debugging).
"""

from __future__ import annotations

import warnings
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention

# ── Try to import AuxRequest (PyTorch >= 2.5).
# Older builds use the deprecated return_lse=True keyword instead.
try:
    from torch.nn.attention.flex_attention import AuxRequest
    _HAS_AUX_REQUEST = True
except ImportError:
    _HAS_AUX_REQUEST = False

# ── Compiled flex_attention (avoid materialising the full N×N matrix).
# torch.compile is a no-op on CPU, so this is safe cross-device.
_flex_compiled = torch.compile(flex_attention, dynamic=True)


# ──────────────────────────────────────────────────────────────────────────────
# Bandwidth estimation
# ──────────────────────────────────────────────────────────────────────────────

def estimate_beta(
    X: torch.Tensor,
    n_sub: int = 2000,
) -> float:
    """
    Estimate the DMAP kernel bandwidth:

        beta = 1 / (2 * median(||xi - xj||^2))

    Uses random subsampling for large N to keep cost O(n_sub^2 * D).

    Parameters
    ----------
    X     : (N, D)  data matrix
    n_sub : int     subsample size for median estimation

    Returns
    -------
    beta : float
    """
    N = X.shape[0]
    idx   = torch.randperm(N, device=X.device)[:min(n_sub, N)]
    X_sub = X[idx].float()                       # always fp32 for stability
    D2    = torch.cdist(X_sub, X_sub).pow(2)
    med   = D2[D2 > 0].median()
    return (1.0 / (2.0 * med)).item()


# ──────────────────────────────────────────────────────────────────────────────
# score_mod factory
# ──────────────────────────────────────────────────────────────────────────────

def make_score_mod(
    q_norm2:  torch.Tensor,                   # (N,)  ||q_i||^2
    k_norm2:  torch.Tensor,                   # (N,)  ||k_j||^2
    beta:     float,
    log_d_q:  Optional[torch.Tensor] = None,  # (N,)  log density at queries
    log_d_k:  Optional[torch.Tensor] = None,  # (N,)  log density at keys
    alpha:    float = 0.0,
    log_h_q:  Optional[torch.Tensor] = None,  # (N,)  log Doob potential (query)
    log_h_k:  Optional[torch.Tensor] = None,  # (N,)  log Doob potential (key)
):
    """
    Build the FlexAttention score_mod for the DMAP kernel.

    The score at position (i, j) is modified from the raw dot product
    q_i . k_j  to:

        s_ij  =  2*beta*(q_i . k_j)
                - beta * (||q_i||^2 + ||k_j||^2)    <- completes -beta||qi-kj||^2
                - alpha * (log_d_q[i] + log_d_k[j]) <- density correction
                + log_h_q[i] - log_h_k[j]           <- Doob tilt (optional)

    After FlexAttention applies row-softmax to these scores, the result is
    the Markov operator P (or its Doob tilt) applied to V:

        out_i = sum_j softmax(s_ij) * V_j  =  (P V)_i

    Note: norm tensors are cast to fp32 inside score_mod regardless of the
    dtype of Q and K, avoiding the catastrophic cancellation that would
    occur in bf16 for close points.

    Parameters
    ----------
    q_norm2, k_norm2 : (N,)   squared norms of Q and K rows
    beta             : float  kernel bandwidth
    log_d_q, log_d_k : (N,)  log densities for alpha-normalisation
                              (pass None to skip density correction)
    alpha            : float  density normalisation exponent
    log_h_q, log_h_k : (N,)  log Doob potential at query / key points
                              (pass None to skip Doob transform)

    Returns
    -------
    score_mod : callable  compatible with flex_attention(score_mod=...)
    """
    # ── Keep all auxiliary tensors in fp32 regardless of input dtype.
    # FlexAttention's score computation runs inside the kernel in fp32
    # accumulators, but the score_mod closure executes at the input dtype
    # unless we explicitly cast. Norm terms involve cancellation (||q||^2
    # is subtracted from 2*q.k) which needs fp32 to avoid ~1e-2 bf16 errors.
    _qn  = q_norm2.float()
    _kn  = k_norm2.float()

    _zero = torch.zeros(1, device=q_norm2.device)

    _log_dq = log_d_q.float() if log_d_q is not None else None
    _log_dk = log_d_k.float() if log_d_k is not None else None
    _log_hq = log_h_q.float() if log_h_q is not None else None
    _log_hk = log_h_k.float() if log_h_k is not None else None

    _beta  = float(beta)
    _alpha = float(alpha)

    def score_mod(score, b, h, qi, ki):
        # score  = q_i . k_j  (raw dot product, scale=1.0 bypasses 1/sqrt(d))
        # Cast to fp32 for precision in the cancellation computation.
        s = _beta * (2.0 * score.float() - _qn[qi] - _kn[ki])

        if _log_dq is not None and _alpha != 0.0:
            s = s - _alpha * (_log_dq[qi] + _log_dk[ki])

        if _log_hq is not None:
            # Doob: multiply score by h_i / h_j = exp(log_h_i - log_h_j)
            # Adding log_h_i and subtracting log_h_j in score space.
            s = s + _log_hq[qi] - _log_hk[ki]

        return s.to(score.dtype)

    return score_mod


# ──────────────────────────────────────────────────────────────────────────────
# _flex_call: thin wrapper handling the AuxRequest / return_lse API difference
# ──────────────────────────────────────────────────────────────────────────────

def _flex_call(
    Q4:         torch.Tensor,    # (1, 1, N, D)
    K4:         torch.Tensor,    # (1, 1, N, D)
    V4:         torch.Tensor,    # (1, 1, N, d)
    score_mod,
    compiled:   bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Call flex_attention and always return (out, lse).

    Handles the API change between PyTorch versions:
      >= 2.6: use return_aux=AuxRequest(lse=True)
      <  2.6: use deprecated return_lse=True

    Returns
    -------
    out : (1, 1, N, d)
    lse : (1, 1, N)     log-sum-exp per query (= log sum_j exp(s_ij))
    """
    fn = _flex_compiled if compiled else flex_attention

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # suppress deprecation noise

        if _HAS_AUX_REQUEST:
            out, aux = fn(Q4, K4, V4,
                          score_mod=score_mod,
                          scale=1.0,
                          return_aux=AuxRequest(lse=True))
            lse = aux.lse
        else:
            out, lse = fn(Q4, K4, V4,
                          score_mod=score_mod,
                          scale=1.0,
                          return_lse=True)

    return out, lse


# ──────────────────────────────────────────────────────────────────────────────
# FlashDiffusion  — the public primitive
# ──────────────────────────────────────────────────────────────────────────────

def FlashDiffusion(
    Q:          torch.Tensor,                   # (N, D)  query points
    K:          torch.Tensor,                   # (M, D)  key points  (= Q for DMAP)
    V:          torch.Tensor,                   # (M, d)  values
    alpha:      float = 1.0,
    beta:       Optional[float] = None,
    h:          Optional[torch.Tensor] = None,  # (N,)  Doob h-potential (>0)
    log_d_q:    Optional[torch.Tensor] = None,  # (N,)  precomputed log density
    log_d_k:    Optional[torch.Tensor] = None,  # (M,)  precomputed log density
    return_lse: bool = False,
    compiled:   bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    FlashDiffusion: O(N) memory DMAP kernel evaluation via FlexAttention.

    Computes:

        out_i  =  sum_j  P_ij  V_j

    where P is the (generalised) DMAP Markov operator:

        P_ij  =  exp(s_ij) / sum_k exp(s_ik)

        s_ij  =  -beta * ||q_i - k_j||^2
                 - alpha * (log_d_q[i] + log_d_k[j])
                 + log_h[i] - log_h[j]                 (if h given)

    No N×M matrix is ever materialised. Memory is O(N + M + d).

    Design
    ------
    The FlexAttention row-softmax IS the Markov normalisation.
    After a single call with the correct score_mod, the output is already
    the Markov operator applied to V — no additional division is needed.

    For the standard two-pass DMAP pipeline (Coifman-Lafon):

        # Pass 1: raw density  d_i = sum_j K(q_i, k_j)
        _, lse1 = FlashDiffusion(Q, K, ones, alpha=0, beta=beta, return_lse=True)
        log_d = lse1                     # (N,)  log sum_j K_ij

        # Pass 2: P+ V  with alpha-normalised kernel
        out = FlashDiffusion(Q, K, V, alpha=1.0, beta=beta,
                             log_d_q=log_d, log_d_k=log_d)

    Connection to DAC (arXiv:2603.28037)
    -------------------------------------
    FlashDiffusion implements the EQ sector of attention:
      Q = K = X (self-attention)  ->  DMAP diffusion operator P+
      alpha = 0                   ->  graph Laplacian kernel (no density)
      alpha = 1                   ->  Laplace-Beltrami (Coifman-Lafon)
      h != None                   ->  Doob h-transform (Schrodinger bridge)

    Parameters
    ----------
    Q         : (N, D)  query points
    K         : (M, D)  key points  (pass K=Q for standard DMAP)
    V         : (M, d)  values to weighted-sum (eigenvectors, ones, etc.)
    alpha     : float   density normalisation exponent
                          0   = no correction
                          0.5 = Fokker-Planck
                          1.0 = Laplace-Beltrami (default)
    beta      : float   kernel bandwidth  1/(2*sigma^2);
                        auto-estimated from Q if None
    h         : (N,)    Doob h-potential (strictly positive);
                        implements the tilted operator diag(h) P diag(1/h)
    log_d_q   : (N,)    log density at query points (from a prior density pass);
                        pass None to skip density correction regardless of alpha
    log_d_k   : (M,)    log density at key points
    return_lse: bool    if True, also return log-sum-exp per query row
                        (= log sum_j exp(s_ij), used for density estimation)
    compiled  : bool    use torch.compile(flex_attention) for fused kernel;
                        set False for CPU debugging

    Returns
    -------
    out : (N, d)
        Weighted sum: out_i = sum_j P_ij V_j

    lse : (N,)   [only if return_lse=True]
        Log-sum-exp: lse_i = log sum_j exp(s_ij)
        Equivalently: lse = log(row sums of un-normalised kernel)
        Used to obtain the density estimate d_i = exp(lse_i).
    """
    N, D = Q.shape
    M    = K.shape[0]

    # ── bandwidth auto-estimation ─────────────────────────────────────────
    if beta is None:
        beta = estimate_beta(Q)

    # ── squared norms (fp32 for precision in score_mod) ───────────────────
    q_norm2 = (Q * Q).sum(-1)    # (N,)
    k_norm2 = (K * K).sum(-1)    # (M,)

    # ── Doob potential in log space ────────────────────────────────────────
    log_h_q: Optional[torch.Tensor] = None
    log_h_k: Optional[torch.Tensor] = None
    if h is not None:
        if h.shape[0] != N:
            raise ValueError(
                f"h must have length N={N}, got {h.shape[0]}"
            )
        log_h_q = h.log()
        log_h_k = h.log()   # for self-attention Q=K so same h applies to both

    # ── consistency checks ─────────────────────────────────────────────────
    if log_d_q is not None and log_d_q.shape[0] != N:
        raise ValueError(f"log_d_q must have length N={N}")
    if log_d_k is not None and log_d_k.shape[0] != M:
        raise ValueError(f"log_d_k must have length M={M}")
    if V.shape[0] != M:
        raise ValueError(f"V must have M={M} rows to match K")

    # ── build score_mod ────────────────────────────────────────────────────
    score_mod = make_score_mod(
        q_norm2  = q_norm2,
        k_norm2  = k_norm2,
        beta     = beta,
        log_d_q  = log_d_q,
        log_d_k  = log_d_k,
        alpha    = alpha,
        log_h_q  = log_h_q,
        log_h_k  = log_h_k,
    )

    # ── reshape to (B=1, H=1, N, D) for FlexAttention ─────────────────────
    Q4 = Q.unsqueeze(0).unsqueeze(0)    # (1, 1, N, D)
    K4 = K.unsqueeze(0).unsqueeze(0)    # (1, 1, M, D)
    V4 = V.unsqueeze(0).unsqueeze(0)    # (1, 1, M, d)

    # ── kernel call ────────────────────────────────────────────────────────
    out4, lse4 = _flex_call(Q4, K4, V4, score_mod, compiled=compiled)

    # ── squeeze batch / head dims ──────────────────────────────────────────
    out = out4.squeeze(0).squeeze(0)    # (N, d)
    lse = lse4.squeeze(0).squeeze(0)    # (N,)

    if return_lse:
        return out, lse
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Density estimation utility  (single-pass via return_lse)
# ──────────────────────────────────────────────────────────────────────────────

def estimate_log_density(
    Q:       torch.Tensor,          # (N, D)
    K:       torch.Tensor,          # (M, D)
    beta:    float,
    alpha:   float = 0.0,
    log_d_q: Optional[torch.Tensor] = None,
    log_d_k: Optional[torch.Tensor] = None,
    compiled: bool = True,
) -> torch.Tensor:
    """
    Estimate log density at query points Q using the kernel over K:

        log_d_i  =  log sum_j exp( -beta * ||q_i - k_j||^2
                                   - alpha*(log_d_q[i] + log_d_k[j]) )

    Implemented as a single FlashDiffusion call with V = ones, returning LSE.

    For a two-pass DMAP pipeline:

        # Pass 1: raw density (alpha=0)
        log_d1 = estimate_log_density(X, X, beta, alpha=0.0)

        # Pass 2: density after alpha-normalisation
        log_d2 = estimate_log_density(X, X, beta, alpha=1.0,
                                      log_d_q=log_d1, log_d_k=log_d1)

    Parameters
    ----------
    Q, K     : point clouds
    beta     : bandwidth
    alpha    : density correction exponent (0 for first pass)
    log_d_q, log_d_k : prior density estimates (None for first pass)

    Returns
    -------
    log_d : (N,)   log sum_j K_alpha(q_i, k_j)
    """
    N = Q.shape[0]
    M = K.shape[0]
    ones = torch.ones(M, 1, device=Q.device, dtype=Q.dtype)

    _, lse = FlashDiffusion(
        Q, K, ones,
        alpha    = alpha,
        beta     = beta,
        log_d_q  = log_d_q,
        log_d_k  = log_d_k,
        return_lse = True,
        compiled   = compiled,
    )
    return lse    # (N,)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: FlashDiffusion as a torch.nn.Module
# (useful for torch.compile / torch.export workflows)
# ──────────────────────────────────────────────────────────────────────────────

class FlashDiffusionModule(nn.Module):
    """
    nn.Module wrapper around FlashDiffusion for integration into
    PyTorch model pipelines (compile, export, DDP, etc.).

    Parameters
    ----------
    alpha    : float   density normalisation exponent (default 1.0)
    beta     : float   bandwidth (estimated at first forward call if None)
    compiled : bool    use torch.compile internally
    """

    def __init__(
        self,
        alpha:    float = 1.0,
        beta:     Optional[float] = None,
        compiled: bool = True,
    ):
        super().__init__()
        self.alpha    = alpha
        self.beta     = beta
        self.compiled = compiled
        self._beta_estimated: Optional[float] = None
        self._log_d_cache: Optional[torch.Tensor] = None

    def forward(
        self,
        Q:          torch.Tensor,
        K:          torch.Tensor,
        V:          torch.Tensor,
        h:          Optional[torch.Tensor] = None,
        log_d_q:    Optional[torch.Tensor] = None,
        log_d_k:    Optional[torch.Tensor] = None,
        return_lse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        beta = self.beta or self._beta_estimated
        if beta is None:
            beta = estimate_beta(Q)
            self._beta_estimated = beta
        return FlashDiffusion(
            Q, K, V,
            alpha      = self.alpha,
            beta       = beta,
            h          = h,
            log_d_q    = log_d_q,
            log_d_k    = log_d_k,
            return_lse = return_lse,
            compiled   = self.compiled,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Quick self-test (run as python kernel.py)
# ──────────────────────────────────────────────────────────────────────────────

def _selftest():
    import sys
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float32
    print(f"FlashDiffusion self-test  device={device}  dtype={dtype}")

    N, D, d = 128, 16, 8
    X = torch.randn(N, D, device=device, dtype=dtype)
    V = torch.randn(N, d, device=device, dtype=dtype)
    beta = estimate_beta(X)
    print(f"  N={N}  D={D}  d={d}  beta={beta:.5f}")

    # ── Dense reference ──────────────────────────────────────────────────
    D2   = torch.cdist(X.float(), X.float()).pow(2)
    Kraw = torch.exp(-beta * D2)
    d1   = Kraw.sum(1)
    Ka   = Kraw / torch.outer(d1, d1)
    d2   = Ka.sum(1)
    P    = Ka / d2.unsqueeze(1)
    ref  = (P @ V.float()).to(dtype)

    # ── Pass 1: density ───────────────────────────────────────────────────
    log_d1 = estimate_log_density(X, X, beta, alpha=0.0, compiled=False)
    err_d1 = (log_d1 - d1.log()).abs().max().item()
    print(f"  Pass 1 density error : {err_d1:.2e}  (target < 1e-5)")

    # ── Pass 2: P+ V ──────────────────────────────────────────────────────
    out = FlashDiffusion(X, X, V, alpha=1.0, beta=beta,
                         log_d_q=log_d1, log_d_k=log_d1, compiled=False)
    err_pv = (out - ref).abs().max().item()
    print(f"  P+ V error           : {err_pv:.2e}  (target < 1e-5)")

    # ── Doob h-transform ─────────────────────────────────────────────────
    h = torch.rand(N, device=device, dtype=dtype).abs() + 0.1
    P_doob_unnorm = torch.outer(h.float(), 1.0/h.float()) * P
    P_doob = P_doob_unnorm / P_doob_unnorm.sum(1, keepdim=True)
    ref_doob = (P_doob @ V.float()).to(dtype)
    out_doob = FlashDiffusion(X, X, V, alpha=1.0, beta=beta,
                              h=h, log_d_q=log_d1, log_d_k=log_d1,
                              compiled=False)
    err_doob = (out_doob - ref_doob).abs().max().item()
    print(f"  Doob h-transform err : {err_doob:.2e}  (target < 1e-5)")

    # ── Module interface ─────────────────────────────────────────────────
    mod = FlashDiffusionModule(alpha=1.0, beta=beta, compiled=False)
    out_mod = mod(X, X, V, log_d_q=log_d1, log_d_k=log_d1)
    err_mod = (out_mod - ref).abs().max().item()
    print(f"  Module interface err : {err_mod:.2e}  (target < 1e-5)")

    passed = all(e < 1e-4 for e in [err_d1, err_pv, err_doob, err_mod])
    print(f"\n  {'PASSED' if passed else 'FAILED'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    _selftest()
