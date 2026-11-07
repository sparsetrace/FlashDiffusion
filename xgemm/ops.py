"""Standalone xgemm public operations: exponential dot-product and Gaussian RBF."""
from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp

from . import _ffi
from ._reference import tiled_exp_gemm

Backend = Literal["auto", "jax", "ffi"]


def _float32(x, name, ndim):
    a = jnp.asarray(x, dtype=jnp.float32)
    if a.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got shape {a.shape}")
    return a


def _validate(Qs, Q, V):
    if Qs.shape[1] != Q.shape[1] or Qs.shape[1] == 0:
        raise ValueError("query and key features must have equal, nonzero D")
    if V.shape[0] != Q.shape[0]:
        raise ValueError("V length must equal number of key rows")


def _choice(backend: Backend):
    if backend not in ("auto", "jax", "ffi"):
        raise ValueError("backend must be one of: 'auto', 'jax', 'ffi'")
    if backend == "jax":
        return "jax"
    if backend == "ffi":
        if jax.default_backend() != "gpu":
            raise RuntimeError("SM80 FFI requires JAX CUDA backend; use backend='jax' on CPU")
        if not _ffi.available():
            raise RuntimeError("SM80 CUDA extension unavailable; run `python tools/build_sm80.py`")
        return "ffi"
    if jax.default_backend() == "gpu" and _ffi.available():
        return "ffi"
    return "jax"


def _apply(Qs, Q, V, qi, qk, beta, *, backend: Backend, tile: int, rbf: bool, vector: bool):
    if tile < 1:
        raise ValueError("tile must be positive")
    if Qs.shape[0] == 0 or Q.shape[0] == 0 or V.shape[1] == 0:
        return jnp.zeros((Qs.shape[0], V.shape[1]), dtype=jnp.float32)
    choice = _choice(backend)
    if choice == "jax":
        return tiled_exp_gemm(Qs, Q, V, qi, qk, beta, tile=tile, rbf=rbf)
    if vector:
        y = _ffi.call_vec(Qs, Q, V[:, 0], qi, qk, beta, rbf=rbf)
        return y[:, None]
    if V.shape[1] > 16:
        return jnp.concatenate(
            [_ffi.call(Qs, Q, V[:, i:i+16], qi, qk, beta, rbf=rbf)
             for i in range(0, V.shape[1], 16)], axis=1)
    return _ffi.call(Qs, Q, V, qi, qk, beta, rbf=rbf)


def matvec(Q_, Q, v, *, backend: Backend = "auto", tile: int = 128):
    """exp(Q_ @ Q.T) @ v; arbitrary D, rectangular Q_/Q, O(tile^2) scratch."""
    Q_, Q, v = (_float32(Q_, "Q_", 2), _float32(Q, "Q", 2),
                 _float32(v, "v", 1))
    _validate(Q_, Q, v[:, None])
    zi, zj = jnp.zeros((Q_.shape[0],), jnp.float32), jnp.zeros((Q.shape[0],), jnp.float32)
    return _apply(Q_, Q, v[:, None], zi, zj, jnp.float32(0),
                  backend=backend, tile=tile, rbf=False, vector=True)[:, 0]


def matmat(Q_, Q, V, *, backend: Backend = "auto", tile: int = 128):
    """exp(Q_ @ Q.T) @ V; no Nq-by-Nk materialization."""
    Q_, Q, V = (_float32(Q_, "Q_", 2), _float32(Q, "Q", 2),
                _float32(V, "V", 2))
    _validate(Q_, Q, V)
    zi, zj = jnp.zeros((Q_.shape[0],), jnp.float32), jnp.zeros((Q.shape[0],), jnp.float32)
    return _apply(Q_, Q, V, zi, zj, jnp.float32(0),
                  backend=backend, tile=tile, rbf=False, vector=False)


def matvec_rbf(Q_, Q, v, q_query, q_key, *, beta, backend: Backend = "auto", tile: int = 128):
    """Low-level RBF using Q_=2*beta*Xq, Q=Xk, q_{query,key}=||X||^2.

    beta is a runtime JAX scalar, not a static FFI attribute.
    """
    Q_, Q, v = _float32(Q_, "Q_", 2), _float32(Q, "Q", 2), _float32(v, "v", 1)
    qi, qk = _float32(q_query, "q_query", 1), _float32(q_key, "q_key", 1)
    beta = jnp.asarray(beta, jnp.float32)
    _validate(Q_, Q, v[:, None])
    if qi.shape != (Q_.shape[0],) or qk.shape != (Q.shape[0],):
        raise ValueError("q_query and q_key must match corresponding row counts")
    return _apply(Q_, Q, v[:, None], qi, qk, beta,
                  backend=backend, tile=tile, rbf=True, vector=True)[:, 0]


def matmat_rbf(Q_, Q, V, q_query, q_key, *, beta, backend: Backend = "auto", tile: int = 128):
    """RBF matmat with pre-scaled query and unscaled squared-norm buffers."""
    Q_, Q, V = _float32(Q_, "Q_", 2), _float32(Q, "Q", 2), _float32(V, "V", 2)
    qi, qk = _float32(q_query, "q_query", 1), _float32(q_key, "q_key", 1)
    beta = jnp.asarray(beta, jnp.float32)
    _validate(Q_, Q, V)
    if qi.shape != (Q_.shape[0],) or qk.shape != (Q.shape[0],):
        raise ValueError("q_query and q_key must match corresponding row counts")
    return _apply(Q_, Q, V, qi, qk, beta,
                  backend=backend, tile=tile, rbf=True, vector=False)


def prepare_rbf_query(Xq, beta):
    """Return (2*beta*Xq, squared norms of unscaled Xq)."""
    Xq = _float32(Xq, "Xq", 2)
    beta = jnp.asarray(beta, jnp.float32)
    return (jnp.float32(2) * beta) * Xq, jnp.sum(Xq * Xq, axis=1)


def prepare_rbf_key(Xk):
    """Return (Xk, squared norms)."""
    Xk = _float32(Xk, "Xk", 2)
    return Xk, jnp.sum(Xk * Xk, axis=1)


def matvec_rbf_from_x(Xq, Xk, v, *, beta, backend: Backend = "auto", tile: int = 128):
    """Gaussian exp(-beta*||x_i-y_j||^2) @ v; recommended RBF API."""
    Q_, qi = prepare_rbf_query(Xq, beta)
    Q, qk = prepare_rbf_key(Xk)
    return matvec_rbf(Q_, Q, v, qi, qk, beta=beta, backend=backend, tile=tile)


def matmat_rbf_from_x(Xq, Xk, V, *, beta, backend: Backend = "auto", tile: int = 128):
    """Gaussian exp(-beta*||x_i-y_j||^2) @ V; recommended RBF API."""
    Q_, qi = prepare_rbf_query(Xq, beta)
    Q, qk = prepare_rbf_key(Xk)
    return matmat_rbf(Q_, Q, V, qi, qk, beta=beta, backend=backend, tile=tile)


def backend_info():
    """Return backend availability without compiling or importing CUDA extensions."""
    return {"jax_backend": jax.default_backend(), "sm80_extension_importable": _ffi.available(),
            "selected_backend": _choice("auto")}
