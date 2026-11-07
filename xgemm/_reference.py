"""Pure JAX, matrix-free, block-tiled reference for exponential GEMM.

No Nq-by-Nk distance/affinity tensor is constructed. The active tile is
B-by-B. The row-major input arrays are expected to be float32.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax


@partial(jax.jit, static_argnames=("tile", "rbf"))
def tiled_exp_gemm(Qs, Q, V, q_query, q_key, beta, *, tile: int, rbf: bool):
    """Compute exp(Qs@Q.T - beta*(q_query+q_key))@V when rbf=True.

    For rbf=False, compute exp(Qs@Q.T)@V. Qs is PRE-SCALED
    to 2*beta*Xq in RBF mode; q_query/q_key are UNSCALED squared norms.
    """
    n_query, d = Qs.shape
    n_key = Q.shape[0]
    b = V.shape[1]

    pad_q = (-n_query) % tile
    pad_k = (-n_key) % tile
    Qs = jnp.pad(Qs, ((0, pad_q), (0, 0)))
    Q = jnp.pad(Q, ((0, pad_k), (0, 0)))
    V = jnp.pad(V, ((0, pad_k), (0, 0)))
    q_query = jnp.pad(q_query, (0, pad_q))
    q_key = jnp.pad(q_key, (0, pad_k))

    nqp = n_query + pad_q
    nkp = n_key + pad_k
    out0 = jnp.zeros((nqp, b), dtype=jnp.float32)

    def row_step(i, out):
        i0 = i * tile
        Xi = lax.dynamic_slice(Qs, (i0, 0), (tile, d))
        qi = lax.dynamic_slice(q_query, (i0,), (tile,))
        acc0 = jnp.zeros((tile, b), dtype=jnp.float32)

        def key_step(j, acc):
            j0 = j * tile
            Xj = lax.dynamic_slice(Q, (j0, 0), (tile, d))
            Vj = lax.dynamic_slice(V, (j0, 0), (tile, b))
            dot = jnp.matmul(Xi, Xj.T, precision=lax.Precision.HIGHEST)
            if rbf:
                qj = lax.dynamic_slice(q_key, (j0,), (tile,))
                logit = jnp.minimum(dot - beta * (qi[:, None] + qj[None, :]), 0.)
            else:
                logit = dot
            return acc + jnp.exp(logit) @ Vj

        result = lax.fori_loop(0, nkp // tile, key_step, acc0)
        return lax.dynamic_update_slice(out, result, (i0, 0))

    full = lax.fori_loop(0, nqp // tile, row_step, out0)
    return full[:n_query]
