"""Run explicitly on an A100/A10 after compiling the optional extension."""
import os

import jax
import numpy as np
import pytest

from xgemm import matvec, matmat, matvec_rbf_from_x, matmat_rbf_from_x
from xgemm import _ffi

if os.getenv("XGEMM_TEST_CUDA") != "1":
    pytest.skip("set XGEMM_TEST_CUDA=1 to enable SM80 GPU tests", allow_module_level=True)
if jax.default_backend() != "gpu" or not _ffi.available():
    pytest.skip("requires JAX CUDA backend and compiled xgemm._sm80 extension", allow_module_level=True)


@pytest.mark.parametrize("nq,nk,d,b", [(73, 141, 3, 5), (128, 64, 32, 16), (65, 107, 47, 21)])
def test_all_four_ops(nq, nk, d, b):
    rng = np.random.default_rng(11)
    Xq = rng.normal(size=(nq, d)).astype(np.float32) / np.sqrt(d)
    Xk = rng.normal(size=(nk, d)).astype(np.float32) / np.sqrt(d)
    V = rng.normal(size=(nk, b)).astype(np.float32)
    v = V[:, 0]

    calls = [
        lambda backend: matvec(Xq, Xk, v, backend=backend),
        lambda backend: matmat(Xq, Xk, V, backend=backend),
        lambda backend: matvec_rbf_from_x(Xq, Xk, v, beta=0.65, backend=backend),
        lambda backend: matmat_rbf_from_x(Xq, Xk, V, beta=0.65, backend=backend),
    ]
    for call in calls:
        reference = np.asarray(call("jax"))
        actual = np.asarray(call("ffi"))
        np.testing.assert_allclose(actual, reference, rtol=2e-3, atol=2e-3)


def test_ffi_dynamic_beta_jit():
    X = np.arange(65 * 3, dtype=np.float32).reshape(65, 3) / 100.
    v = np.ones((65,), np.float32)
    fn = jax.jit(lambda b: matvec_rbf_from_x(X, X, v, beta=b, backend="ffi"))
    a = np.asarray(fn(np.float32(0.2)))
    b = np.asarray(fn(np.float32(1.2)))
    assert np.linalg.norm(a-b) > 0.01
