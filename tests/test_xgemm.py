import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xgemm import (
    matvec, matmat, matvec_rbf_from_x, matmat_rbf_from_x,
    matvec_rbf, matmat_rbf, prepare_rbf_query, prepare_rbf_key,
)


def sample(nq, nk, d, b, seed=42):
    rng = np.random.default_rng(seed)
    Xq = rng.normal(size=(nq, d)).astype(np.float32) / np.sqrt(d)
    Xk = rng.normal(size=(nk, d)).astype(np.float32) / np.sqrt(d)
    V = rng.normal(size=(nk, b)).astype(np.float32)
    return Xq, Xk, V


@pytest.mark.parametrize("shape", [(17, 23, 3, 1), (83, 71, 7, 5), (65, 101, 37, 21), (64, 128, 32, 16)])
def test_dot(shape):
    Xq, Xk, V = sample(*shape)
    ref = np.exp(Xq.astype(np.float64) @ Xk.astype(np.float64).T) @ V.astype(np.float64)
    got = np.asarray(matmat(Xq, Xk, V, backend="jax", tile=32))
    np.testing.assert_allclose(got, ref, rtol=8e-5, atol=8e-5)
    v = V[:, 0]
    got_vec = np.asarray(matvec(Xq, Xk, v, backend="jax", tile=32))
    np.testing.assert_allclose(got_vec, ref[:, 0], rtol=8e-5, atol=8e-5)


@pytest.mark.parametrize("shape", [(17, 23, 3, 1), (83, 71, 7, 5), (65, 101, 37, 21), (64, 128, 32, 16)])
def test_rbf(shape):
    Xq, Xk, V = sample(*shape)
    beta = np.float32(0.75)
    d2 = ((Xq.astype(np.float64)[:, None, :] - Xk.astype(np.float64)[None, :, :]) ** 2).sum(-1)
    ref = np.exp(-float(beta) * d2) @ V.astype(np.float64)
    got = np.asarray(matmat_rbf_from_x(Xq, Xk, V, beta=beta, backend="jax", tile=32))
    np.testing.assert_allclose(got, ref, rtol=2e-4, atol=2e-4)
    got_vec = np.asarray(matvec_rbf_from_x(Xq, Xk, V[:, 0], beta=beta, backend="jax", tile=32))
    np.testing.assert_allclose(got_vec, ref[:, 0], rtol=2e-4, atol=2e-4)

    Qs, qi = prepare_rbf_query(Xq, beta)
    Q, qk = prepare_rbf_key(Xk)
    low = np.asarray(matmat_rbf(Qs, Q, V, qi, qk, beta=beta, backend="jax", tile=32))
    np.testing.assert_allclose(low, got, atol=1e-5, rtol=1e-5)
    low_vec = np.asarray(matvec_rbf(Qs, Q, V[:, 0], qi, qk, beta=beta, backend="jax", tile=32))
    np.testing.assert_allclose(low_vec, got_vec, atol=1e-5, rtol=1e-5)


def test_jit_and_dynamic_beta():
    Xq, Xk, V = sample(33, 49, 9, 3)
    f = jax.jit(lambda beta: matmat_rbf_from_x(Xq, Xk, V, beta=beta, backend="jax", tile=32))
    y1 = np.asarray(f(jnp.float32(0.2)))
    y2 = np.asarray(f(jnp.float32(1.2)))
    assert np.linalg.norm(y1 - y2) > 0.01
    np.testing.assert_allclose(y1, np.asarray(matmat_rbf_from_x(Xq, Xk, V, beta=0.2, backend="jax", tile=32)), rtol=1e-5, atol=1e-5)


def test_reference_beta_grad():
    Xq, Xk, V = sample(21, 22, 3, 2)
    f = lambda beta: jnp.sum(matmat_rbf_from_x(Xq, Xk, V, beta=beta, backend="jax", tile=32) ** 2)
    b = jnp.float32(0.8)
    g = float(jax.grad(f)(b))
    h = 5e-3
    gfd = float((f(b + h) - f(b - h)) / (2*h))
    np.testing.assert_allclose(g, gfd, rtol=1e-2, atol=1e-2)


def test_zero_query_and_keys():
    Xq = np.zeros((0, 3), np.float32)
    Xk = np.zeros((4, 3), np.float32)
    v = np.ones((4,), np.float32)
    assert matvec(Xq, Xk, v).shape == (0,)
    y = matmat_rbf_from_x(np.ones((2, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros((0, 6), np.float32), beta=0.5)
    assert y.shape == (2, 6)
    assert np.all(np.asarray(y) == 0)


def test_invalid_shape():
    Xq, Xk, V = sample(10, 12, 3, 2)
    with pytest.raises(ValueError, match="V length"):
        matmat(Xq, Xk, V[:3])
    with pytest.raises(ValueError, match="nonzero D"):
        matvec(Xq[:, :2], Xk, V[:, 0])
