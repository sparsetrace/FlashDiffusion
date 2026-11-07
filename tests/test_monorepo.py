"""The single PyPI distribution exposes two importable packages."""
import importlib.metadata as md

import flashdiffusion
import xgemm


def test_imports():
    assert flashdiffusion.__version__
    assert callable(xgemm.matmat_rbf_from_x)


def test_one_distribution():
    dist = md.distribution("flashdiffusion")
    assert dist.metadata["Name"].lower() == "flashdiffusion"
