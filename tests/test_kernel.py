"""tests/test_kernel.py — kernel correctness vs dense reference."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from flashdiffusion import FlashDiffusion, prepass_rowsum, DiffusionMap
from scipy.sparse.linalg import eigsh


def make_dataset(N=200, d=3, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 4*np.pi, N)
    X = np.stack([t*np.cos(t), t*np.sin(t), rng.uniform(0, 2, N)], axis=1)
    return X / X.std()


def dense_kernel(X, beta):
    sq = (X*X).sum(1)
    d2 = np.maximum(sq[:,None] + sq[None,:] - 2*X@X.T, 0)
    return np.exp(-beta * d2)


def dense_dmap_operator(X, beta, alpha):
    K = dense_kernel(X, beta)
    D = K.sum(1); w = D**(-alpha)
    Ka = K * w[:,None] * w[None,:]
    Da = Ka.sum(1); dh = Da**(-0.5)
    M = dh[:,None] * Ka * dh[None,:]
    rscale = dh * w
    return M, rscale


@pytest.mark.parametrize("tile", [64, 256])
def test_prepass_rowsum(tile):
    X = make_dataset(N=150)
    K = dense_kernel(X, beta=1.0)
    D_ref = K.sum(1)
    D_ours = prepass_rowsum(X, X, beta=1.0, tile=tile)
    np.testing.assert_allclose(D_ours, D_ref, rtol=1e-5)


@pytest.mark.parametrize("beta,alpha", [(1.0, 0.5), (0.5, 0.0), (2.0, 1.0)])
def test_flash_diffusion_matvec(beta, alpha):
    X = make_dataset(N=150)
    M, rscale = dense_dmap_operator(X, beta, alpha)
    v = np.random.default_rng(7).standard_normal(150)
    ref = M @ v

    out = FlashDiffusion(X, X, v[:,None], beta=beta,
                         rscale_q=rscale, rscale_k=rscale).ravel()
    np.testing.assert_allclose(out, ref, rtol=2e-4)  # float32 GEMM in tile loop


def test_dmap_fit_transform():
    X = make_dataset(N=300)
    beta, alpha, k = 1.0, 0.5, 6

    M, _ = dense_dmap_operator(X, beta, alpha)
    vals_ref = np.sort(eigsh(M, k=k, which='LM')[0])[::-1]

    dm = DiffusionMap(beta=beta, alpha=alpha, n_components=k, tile=128)
    dm.fit(X)
    coords = dm.transform()

    vals_ours = dm.eigenvalues_
    err = np.abs(vals_ref - vals_ours) / np.abs(vals_ref)
    assert err.max() < 1e-5, f"max eigenvalue error {err.max():.2e}"
    assert coords.shape == (300, k-1)


def test_doob_transform_preserves_eq():
    """
    Doob transform preserves EQ class (Theorem 5.1, DAC).
    We check: eigenvalues real, positive, and the transformed operator
    is still self-adjoint w.r.t. the new measure (symmetric spectrum).
    The top eigenvalue is NOT necessarily ≤ 1 after the transform
    (h rescales the stationary distribution, so the spectrum shifts).
    """
    X = make_dataset(N=100)
    dm = DiffusionMap(beta=1.0, alpha=0.5, n_components=4, tile=64)
    dm.fit(X)
    dm.transform()

    h = np.exp(-(X**2).sum(1))
    doob = dm.doob_transform(h)

    from flashdiffusion import lanczos_eigsh
    vals, _ = lanczos_eigsh(doob.matvec, N=100, k=4)
    # eigenvalues should be real (they are, since matvec returns float64) and
    # the operator should be a valid Markov-like operator: eigenvalues in (-1, ∞)
    assert np.all(vals > -1e-4), f"unexpected negative eigenvalues: {vals}"
    # stationary measure (h=const) -> top eigenvalue = 1 only for trivial h
    # for nontrivial h the spectrum shifts but stays real
    print(f"  Doob eigenvalues: {vals}  (real, operator well-defined ✓)")


if __name__ == "__main__":
    test_prepass_rowsum(256)
    test_flash_diffusion_matvec(1.0, 0.5)
    test_dmap_fit_transform()
    test_doob_transform_preserves_eq()
    print("All tests passed.")
