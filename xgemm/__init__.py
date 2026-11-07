"""xgemm: matrix-free exponential GEMM, Gaussian RBF and JAX FFI for SM80.

Importing xgemm never builds CUDA extensions or downloads dependencies.
"""
from .ops import (
    matvec, matmat,
    matvec_rbf, matmat_rbf,
    matvec_rbf_from_x, matmat_rbf_from_x,
    prepare_rbf_query, prepare_rbf_key,
    backend_info,
)

__version__ = "0.1.0"
__all__ = [
    "matvec", "matmat", "matvec_rbf", "matmat_rbf",
    "matvec_rbf_from_x", "matmat_rbf_from_x",
    "prepare_rbf_query", "prepare_rbf_key", "backend_info",
]
