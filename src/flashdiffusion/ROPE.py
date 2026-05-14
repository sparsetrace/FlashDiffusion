# rope_encoder.py
# ============================================================
# ROPE Encoder: RoPE-DMAP on Hankel windows
#
# - Learns diffusion coordinates ψ_T for each training Hankel window
# - Applies RoPE BEFORE the diffusion-map kernel is constructed
# - Supports Nyström out-of-sample embedding for new windows/series
#
# This is intentionally an "encoder only":
#   - no decoder
#   - no forecasting
#   - just RoPE-DMAP coordinates + Nyström extension
#
# Relation to NLSA:
#   - pos_scale = 0 recovers ordinary dense NLSA / DMAP on Hankel windows.
#   - rope_mode = "sample" applies RoPE to each sample inside each Hankel window.
#   - rope_mode = "window" applies RoPE once to the flattened Hankel vector.
#
# ============================================================

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import eigsh


# ============================================================
# Hankel-window helpers
# ============================================================

def make_hankel_windows(F_tX: np.ndarray, L: int, step: int = 1) -> np.ndarray:
    """
    Build Hankel windows from a time series.

    F_tX: (N,D)
    returns W_TcX: (K,L,D), where
        W[T,c] = F[T + c*step]
        K = N - (L-1)*step
    """
    F = np.asarray(F_tX, dtype=float)
    if F.ndim == 1:
        F = F[:, None]

    N, D = F.shape
    L = int(L)
    step = int(step)

    if L <= 0:
        raise ValueError("L must be positive.")
    if step <= 0:
        raise ValueError("step must be positive.")

    K = N - (L - 1) * step
    if K <= 0:
        raise ValueError(f"Need N={N} >= (L-1)*step+1 = {(L-1)*step+1}")

    return np.stack([F[c * step : c * step + K, :] for c in range(L)], axis=1)


def make_window_positions(
    sample_positions: np.ndarray,
    L: int,
    step: int = 1,
    mode: str = "sample",
    window_position: str = "end",
) -> np.ndarray:
    """
    Construct positions for Hankel windows.

    sample_positions: (N,)

    If mode == "sample":
        returns Pos_Tc: (K,L), the absolute/sample positions inside each window.

    If mode == "window":
        returns Pos_T: (K,), one position per window.
    """
    pos = np.asarray(sample_positions, dtype=float)
    if pos.ndim != 1:
        raise ValueError("sample_positions must be one-dimensional.")

    N = len(pos)
    K = N - (L - 1) * step
    if K <= 0:
        raise ValueError("Not enough positions for windows.")

    Pos_Tc = np.stack([pos[c * step : c * step + K] for c in range(L)], axis=1)

    if mode == "sample":
        return Pos_Tc

    if mode != "window":
        raise ValueError("mode must be 'sample' or 'window'.")

    if window_position == "start":
        return Pos_Tc[:, 0]
    if window_position == "center":
        return Pos_Tc[:, L // 2]
    if window_position == "end":
        return Pos_Tc[:, -1]

    raise ValueError("window_position must be 'start', 'center', or 'end'.")


# ============================================================
# ROPE Encoder
# ============================================================

class ROPE:
    """
    Dense RoPE-DMAP encoder on Hankel windows.

    Training series:
      F_tX : (N,D)
      windows: W_T = [F_T, F_{T+step}, ..., F_{T+(L-1)step}]
      T = 0..K-1, K = N-(L-1)step

    Pre-kernel RoPE:
      In sample mode:
        each sample F_{T+c} inside the window is rotated by its own position.

      In window mode:
        the flattened window is rotated once by the window position.

    Kernel:
      K(T,T') = exp(-beta * ||RoPE(W_T) - RoPE(W_T')||^2)

    Diffusion normalization:
      q(T) = sum_T' K(T,T')
      K_alpha = K / (q(T)^alpha q(T')^alpha)
      d(T) = sum_T' K_alpha(T,T')
      P_sym = d^{-1/2} K_alpha d^{-1/2}

    Embedding:
      P_sym phi_j = lambda_j phi_j
      psi_j(T) = d(T)^(-1/2) phi_j(T)

    Out-of-sample:
      Nyström extension using the same pre-kernel RoPE transformation.
    """

    def __init__(
        self,
        F_tX: np.ndarray,
        L: int,
        rank: int = 20,
        beta: float | None = None,
        alpha: float = 1.0,
        center: bool = True,
        scale: bool = False,
        drop_first: bool = True,
        step: int = 1,
        sample_positions: np.ndarray | None = None,
        rope_mode: str = "sample",
        window_position: str = "end",
        pos_scale: float = 1.0,
        base: float = 10000.0,
        freqs: np.ndarray | None = None,
        n_freqs: int | None = None,
        beta_sample_pairs: int = 20000,
        seed: int = 0,
    ):
        R = np.asarray(F_tX, dtype=float)
        if R.ndim == 1:
            R = R[:, None]

        self.center = bool(center)
        self.scale = bool(scale)

        self.mu_ = R.mean(axis=0, keepdims=True) if self.center else np.zeros((1, R.shape[1]))
        self.sigma_ = R.std(axis=0, keepdims=True) if self.scale else np.ones((1, R.shape[1]))
        self.sigma_ = np.where(self.sigma_ == 0, 1.0, self.sigma_)

        self.R_raw_ = R
        self.R_ = (R - self.mu_) / self.sigma_

        self.N_, self.D_ = self.R_.shape
        self.L = int(L)
        self.step = int(step)

        self.K_ = self.N_ - (self.L - 1) * self.step
        if self.K_ <= 0:
            raise ValueError(f"N={self.N_} too small for L={self.L}, step={self.step}")

        self.rank_req_ = int(rank)
        self.beta_in_ = beta
        self.alpha_ = float(alpha)
        self.drop_first_ = bool(drop_first)

        self.rope_mode_ = str(rope_mode)
        if self.rope_mode_ not in ("sample", "window"):
            raise ValueError("rope_mode must be 'sample' or 'window'.")

        self.window_position_ = str(window_position)
        self.pos_scale_ = float(pos_scale)
        self.base_ = float(base)
        self.freqs_in_ = None if freqs is None else np.asarray(freqs, dtype=float)
        self.n_freqs_ = None if n_freqs is None else int(n_freqs)

        if sample_positions is None:
            sample_positions = np.arange(self.N_, dtype=float)
        self.sample_positions_ = np.asarray(sample_positions, dtype=float)
        if self.sample_positions_.shape != (self.N_,):
            raise ValueError(f"sample_positions must have shape ({self.N_},).")

        self.beta_sample_pairs_ = int(beta_sample_pairs)
        self.rng_ = np.random.default_rng(int(seed))

        # learned artifacts
        self.W_TcX_ = None              # raw centered/scaled windows, shape (K,L,D)
        self.Pos_ = None                # (K,L) for sample mode or (K,) for window mode
        self.Y_ = None                  # RoPE-transformed flattened windows, shape (K,D_rope)
        self.r2_T_ = None               # squared norms of Y_
        self.G_ = None                  # dense Gram of Y_
        self.beta_ = None
        self.Kmat_ = None               # raw Gaussian kernel
        self.K_T_ = None                # q(T)
        self.d_T_ = None
        self.inv_sqrt_d_ = None
        self.lam_ = None
        self.phi_ = None
        self.psi_ = None
        self.freqs_ = None

        self.fit()

    # -------------------------
    # RoPE internals
    # -------------------------

    def _freqs(self, dim: int) -> np.ndarray:
        """
        Standard RoPE frequency schedule over adjacent 2D pairs.

        If freqs was provided, use it.
        If n_freqs is provided, only the first n_freqs pairs rotate; remaining pairs have frequency 0.
        Odd leftover dimension is not rotated.
        """
        n_pairs = dim // 2
        if n_pairs == 0:
            return np.zeros((0,), dtype=float)

        if self.freqs_in_ is not None:
            freqs = np.asarray(self.freqs_in_, dtype=float)
            if freqs.size > n_pairs:
                freqs = freqs[:n_pairs]
            if freqs.size < n_pairs:
                freqs = np.concatenate([freqs, np.zeros(n_pairs - freqs.size)])
            return freqs

        m = np.arange(n_pairs, dtype=float)
        freqs = self.base_ ** (-2.0 * m / max(2, 2 * n_pairs))

        if self.n_freqs_ is not None:
            n_active = max(0, min(int(self.n_freqs_), n_pairs))
            mask = np.arange(n_pairs) < n_active
            freqs = freqs * mask

        return freqs

    @staticmethod
    def _rotate_matrix_rows(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Rotate adjacent pairs of the last dimension of X.

        X: (..., D)
        theta: (..., n_pairs), broadcast-compatible with X[..., 0::2] pairs

        Odd final dimension is copied unchanged.
        """
        X = np.asarray(X, dtype=float)
        D = X.shape[-1]
        n_pairs = D // 2

        if n_pairs == 0:
            return X.copy()

        pair_part = X[..., : 2 * n_pairs].reshape(*X.shape[:-1], n_pairs, 2)
        a = pair_part[..., 0]
        b = pair_part[..., 1]

        c = np.cos(theta)
        s = np.sin(theta)

        Ypair = np.empty_like(pair_part)
        Ypair[..., 0] = a * c - b * s
        Ypair[..., 1] = a * s + b * c

        Yflat = Ypair.reshape(*X.shape[:-1], 2 * n_pairs)

        if D % 2 == 1:
            return np.concatenate([Yflat, X[..., -1:]], axis=-1)

        return Yflat

    def _rope_sample_windows(self, W_TcX: np.ndarray, Pos_Tc: np.ndarray) -> np.ndarray:
        """
        Apply RoPE to each sample in each Hankel window.

        W_TcX: (K,L,D)
        Pos_Tc: (K,L)
        returns Y: (K, L*D)
        """
        K, L, D = W_TcX.shape
        freqs = self._freqs(D)
        self.freqs_ = freqs

        if freqs.size == 0 or self.pos_scale_ == 0.0:
            return W_TcX.reshape(K, L * D)

        theta = Pos_Tc[:, :, None] * self.pos_scale_ * freqs[None, None, :]
        Y_TcX = self._rotate_matrix_rows(W_TcX, theta)
        return Y_TcX.reshape(K, L * D)

    def _rope_window_vectors(self, H_TZ: np.ndarray, Pos_T: np.ndarray) -> np.ndarray:
        """
        Apply RoPE once to flattened Hankel vectors.

        H_TZ: (K,Z)
        Pos_T: (K,)
        returns Y: (K,Z)
        """
        K, Z = H_TZ.shape
        freqs = self._freqs(Z)
        self.freqs_ = freqs

        if freqs.size == 0 or self.pos_scale_ == 0.0:
            return H_TZ.copy()

        theta = Pos_T[:, None] * self.pos_scale_ * freqs[None, :]
        return self._rotate_matrix_rows(H_TZ, theta)

    def _transform_windows(self, W_TcX: np.ndarray, Pos) -> np.ndarray:
        """
        Center/scale must already be applied to W_TcX.
        """
        if self.rope_mode_ == "sample":
            return self._rope_sample_windows(W_TcX, Pos)

        H = W_TcX.reshape(W_TcX.shape[0], -1)
        return self._rope_window_vectors(H, Pos)

    # -------------------------
    # training
    # -------------------------

    def _choose_beta(self, D2: np.ndarray) -> float:
        """
        Pick beta via median heuristic on sampled off-diagonal distances,
        unless beta was provided.
        """
        if self.beta_in_ is not None:
            return float(self.beta_in_)

        K = self.K_
        M_max = K * (K - 1) // 2
        M = min(self.beta_sample_pairs_, M_max)
        if M <= 0:
            return 1.0

        ii = self.rng_.integers(0, K, size=M)
        jj = self.rng_.integers(0, K, size=M)
        mask = ii != jj
        ii, jj = ii[mask], jj[mask]
        if ii.size == 0:
            return 1.0

        med = np.median(D2[ii, jj])
        return 1.0 / (med + 1e-12)

    def fit(self) -> "ROPE":
        # Hankel windows from centered/scaled data
        self.W_TcX_ = make_hankel_windows(self.R_, self.L, step=self.step)

        # Window/sample positions
        self.Pos_ = make_window_positions(
            self.sample_positions_,
            self.L,
            step=self.step,
            mode=self.rope_mode_,
            window_position=self.window_position_,
        )

        # Pre-kernel RoPE transform
        self.Y_ = self._transform_windows(self.W_TcX_, self.Pos_)

        # Dense Gram/distances
        self.G_ = self.Y_ @ self.Y_.T
        self.r2_T_ = np.sum(self.Y_ * self.Y_, axis=1)

        D2 = self.r2_T_[:, None] + self.r2_T_[None, :] - 2.0 * self.G_
        np.maximum(D2, 0.0, out=D2)

        # beta
        self.beta_ = self._choose_beta(D2)

        # Gaussian kernel on RoPE-transformed windows
        Kmat = np.exp(-self.beta_ * D2)

        # Diffusion maps alpha normalization
        K_T = Kmat.sum(axis=1) + 1e-18
        KTa = K_T ** self.alpha_
        Kalpha = Kmat / (KTa[:, None] * KTa[None, :])

        d_T = Kalpha.sum(axis=1) + 1e-18
        inv_sqrt_d = 1.0 / np.sqrt(d_T)

        Psym = (inv_sqrt_d[:, None] * Kalpha) * inv_sqrt_d[None, :]

        # Top eigenpairs of symmetric operator
        k = min(self.rank_req_ + (1 if self.drop_first_ else 0), self.K_ - 1)
        if k <= 0:
            self.lam_ = np.zeros((0,), dtype=float)
            self.phi_ = np.zeros((self.K_, 0), dtype=float)
            self.psi_ = np.zeros((self.K_, 0), dtype=float)
            self.Kmat_ = Kmat
            self.K_T_ = K_T
            self.d_T_ = d_T
            self.inv_sqrt_d_ = inv_sqrt_d
            return self

        w, V = eigsh(Psym, k=k, which="LA")

        idx = np.argsort(w)[::-1]
        w = w[idx]
        V = V[:, idx]

        if self.drop_first_ and w.size > 0:
            w = w[1:]
            V = V[:, 1:]

        self.lam_ = w
        self.phi_ = V
        self.Kmat_ = Kmat
        self.K_T_ = K_T
        self.d_T_ = d_T
        self.inv_sqrt_d_ = inv_sqrt_d

        self.psi_ = inv_sqrt_d[:, None] * V
        return self

    # -------------------------
    # encoding / Nyström
    # -------------------------

    def _prepare_query_windows(
        self,
        W_BLX: np.ndarray,
        positions=None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Center/scale query windows and build query positions.
        """
        W = np.asarray(W_BLX, dtype=float)
        if W.ndim == 2:
            W = W[None, :, :]
        if W.ndim != 3:
            raise ValueError("Expected query windows with shape (B,L,D) or (L,D).")
        if W.shape[1:] != (self.L, self.D_):
            raise ValueError(f"Expected window shape (*,{self.L},{self.D_}), got {W.shape}")

        Wc = (W - self.mu_) / self.sigma_

        B = W.shape[0]

        if self.rope_mode_ == "sample":
            if positions is None:
                # Local positions default. For real use, pass absolute positions.
                Pos = np.tile(np.arange(self.L, dtype=float), (B, 1))
            else:
                Pos = np.asarray(positions, dtype=float)
                if Pos.ndim == 1 and Pos.shape[0] == self.L:
                    Pos = np.tile(Pos[None, :], (B, 1))
                if Pos.shape != (B, self.L):
                    raise ValueError(f"For sample mode, positions must have shape ({B},{self.L}) or ({self.L},).")
            return Wc, Pos

        # window mode
        if positions is None:
            Pos = np.zeros((B,), dtype=float)
        else:
            Pos = np.asarray(positions, dtype=float)
            if Pos.ndim == 0:
                Pos = np.full((B,), float(Pos))
            if Pos.shape != (B,):
                raise ValueError(f"For window mode, positions must have shape ({B},) or scalar.")
        return Wc, Pos

    def encode_windows(self, W_BLX: np.ndarray, positions=None) -> np.ndarray:
        """
        Encode a batch of query windows using Nyström.

        Inputs:
          W_BLX: (B,L,D) or (L,D)
          positions:
            sample mode: (B,L), or (L,), absolute positions inside query windows
            window mode: (B,), or scalar, one position per query window

        Returns:
          Psi_Br: (B,r)
        """
        if self.psi_ is None or self.lam_ is None or self.psi_.shape[1] == 0:
            B = 1 if np.asarray(W_BLX).ndim == 2 else np.asarray(W_BLX).shape[0]
            return np.zeros((B, 0), dtype=float)

        Wc, Pos = self._prepare_query_windows(W_BLX, positions=positions)
        Yq = self._transform_windows(Wc, Pos)

        Gq = Yq @ self.Y_.T
        r2q = np.sum(Yq * Yq, axis=1)

        D2 = r2q[:, None] + self.r2_T_[None, :] - 2.0 * Gq
        np.maximum(D2, 0.0, out=D2)

        k_qT = np.exp(-self.beta_ * D2)

        # alpha normalization query -> train
        Kq = k_qT.sum(axis=1) + 1e-18
        k_qT_alpha = k_qT / ((Kq[:, None] ** self.alpha_) * (self.K_T_[None, :] ** self.alpha_))

        dq = k_qT_alpha.sum(axis=1) + 1e-18
        P_qT = k_qT_alpha / dq[:, None]

        lam_safe = np.maximum(self.lam_, 1e-12)
        psi_q = P_qT @ (self.psi_ / lam_safe[None, :])
        return psi_q

    def encode_window(self, W_cX: np.ndarray, positions=None) -> np.ndarray:
        """
        Encode one query window.

        Returns:
          psi_q: (r,)
        """
        return self.encode_windows(np.asarray(W_cX)[None, :, :], positions=positions)[0]

    def encode_series(
        self,
        F_aX: np.ndarray,
        sample_positions: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Encode all Hankel windows of a novel series.

        F_aX: (N,D), N >= (L-1)*step+1

        sample_positions:
          optional length-N positions. If omitted, uses arange(N).

        Returns:
          Psi: (K_n,r)
        """
        F = np.asarray(F_aX, dtype=float)
        if F.ndim == 1:
            F = F[:, None]
        N, D = F.shape
        if D != self.D_:
            raise ValueError(f"Expected D={self.D_}, got D={D}.")

        if sample_positions is None:
            sample_positions = np.arange(N, dtype=float)
        sample_positions = np.asarray(sample_positions, dtype=float)
        if sample_positions.shape != (N,):
            raise ValueError(f"sample_positions must have shape ({N},).")

        # Build raw query windows and matching positions.
        W = make_hankel_windows(F, self.L, step=self.step)

        Pos = make_window_positions(
            sample_positions,
            self.L,
            step=self.step,
            mode=self.rope_mode_,
            window_position=self.window_position_,
        )

        return self.encode_windows(W, positions=Pos)

    # -------------------------
    # convenience
    # -------------------------

    @property
    def embedding_(self) -> np.ndarray:
        """
        Training diffusion coordinates, shape (K,rank).
        """
        return self.psi_

    def fit_transform(self) -> np.ndarray:
        """
        Return training diffusion coordinates.
        """
        return self.psi_


__all__ = ["ROPE", "make_hankel_windows", "make_window_positions"]
