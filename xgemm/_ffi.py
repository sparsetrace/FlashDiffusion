"""Optional SM80 JAX typed-FFI binding; no build or import-time side effects."""
from __future__ import annotations

import importlib
import threading

import jax
import jax.numpy as jnp
from jax import ffi

# XLA minor-to-major convention: (1, 0) is C-contiguous/row-major 2-D.
_LAYOUT_1D = (0,)
_LAYOUT_2D = (1, 0)
_TILE = 64
_KCHUNK = 32
_BV = 16
_REGISTERED = False
_LOCK = threading.Lock()

_TARGETS = {
    "dot_vec": "xgemm_sm80_fp32_matvec",
    "dot_mat": "xgemm_sm80_fp32_matmat",
    "rbf_vec": "xgemm_sm80_fp32_matvec_rbf",
    "rbf_mat": "xgemm_sm80_fp32_matmat_rbf",
}


def available() -> bool:
    """True iff the compiled extension is importable (not GPU detection)."""
    try:
        importlib.import_module("xgemm._sm80")
        return True
    except ImportError:
        return False


def _register():
    global _REGISTERED
    if _REGISTERED:
        return
    with _LOCK:
        if _REGISTERED:
            return
        ext = importlib.import_module("xgemm._sm80")
        registrations = ext.registrations()
        missing = set(_TARGETS.values()) - set(registrations)
        if missing:
            raise RuntimeError(f"SM80 extension missing FFI targets: {sorted(missing)}")
        for target, capsule in registrations.items():
            ffi.register_ffi_target(target, capsule, platform="CUDA", api_version=1)
        _REGISTERED = True


def _pad_n(A):
    n = A.shape[0]
    p = (-n) % _TILE
    if A.ndim == 1:
        return jnp.pad(A, ((0, p),)), n
    return jnp.pad(A, ((0, p), (0, 0))), n


def _pad_d(A):
    p = (-A.shape[1]) % _KCHUNK
    return jnp.pad(A, ((0, 0), (0, p)))


def call(Qs, Q, V, q_query, q_key, beta, *, rbf):
    """V is rank-2 and 1 <= width <= 16; returns rank-2."""
    _register()
    if not (1 <= V.shape[1] <= _BV):
        raise ValueError("SM80 matmat supports 1..16 RHS columns per call")

    Qs, n_query = _pad_n(_pad_d(Qs))
    Q, _ = _pad_n(_pad_d(Q))
    V, _ = _pad_n(V)
    q_query, _ = _pad_n(q_query)
    q_key, _ = _pad_n(q_key)

    width = V.shape[1]
    out_type = jax.ShapeDtypeStruct((Qs.shape[0], width), jnp.float32)

    if rbf:
        fn = ffi.ffi_call(
            _TARGETS["rbf_mat"], out_type,
            input_layouts=(_LAYOUT_2D, _LAYOUT_2D, _LAYOUT_2D,
                           _LAYOUT_1D, _LAYOUT_1D, ()),
            output_layouts=_LAYOUT_2D,
            vmap_method="sequential",
        )
        out = fn(Qs, Q, V, q_query, q_key, jnp.asarray(beta, dtype=jnp.float32))
    else:
        fn = ffi.ffi_call(
            _TARGETS["dot_mat"], out_type,
            input_layouts=(_LAYOUT_2D, _LAYOUT_2D, _LAYOUT_2D),
            output_layouts=_LAYOUT_2D,
            vmap_method="sequential",
        )
        out = fn(Qs, Q, V)

    return out[:n_query]


def call_vec(Qs, Q, v, q_query, q_key, beta, *, rbf):
    """Dedicated vector FFI targets: v and output are rank-1."""
    _register()
    Qs, n_query = _pad_n(_pad_d(Qs))
    Q, _ = _pad_n(_pad_d(Q))
    v, _ = _pad_n(v)
    q_query, _ = _pad_n(q_query)
    q_key, _ = _pad_n(q_key)
    out_type = jax.ShapeDtypeStruct((Qs.shape[0],), jnp.float32)
    if rbf:
        fn = ffi.ffi_call(
            _TARGETS["rbf_vec"], out_type,
            input_layouts=(_LAYOUT_2D, _LAYOUT_2D, _LAYOUT_1D,
                           _LAYOUT_1D, _LAYOUT_1D, ()),
            output_layouts=_LAYOUT_1D,
            vmap_method="sequential",
        )
        out = fn(Qs, Q, v, q_query, q_key, jnp.asarray(beta, dtype=jnp.float32))
    else:
        fn = ffi.ffi_call(
            _TARGETS["dot_vec"], out_type,
            input_layouts=(_LAYOUT_2D, _LAYOUT_2D, _LAYOUT_1D),
            output_layouts=_LAYOUT_1D,
            vmap_method="sequential",
        )
        out = fn(Qs, Q, v)
    return out[:n_query]
