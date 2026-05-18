# src/flashdiffusion/__init__.py

import sys, os
_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

import os, torch
def _jit_init():
    """On first import: detect GPU, JIT-build kernel, inject into sys.modules."""
    if not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))   # src/flashdiffusion → repo root

    from .kernel_jit import _build_kernel
    try:
        _build_kernel(sm, repo_root)
    except Exception as e:
        print(f'[FlashDiffusion] JIT build failed ({e}), CPU fallback only')

_jit_init()
