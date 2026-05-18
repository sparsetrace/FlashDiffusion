# src/flashdiffusion/__init__.py

import sys, os

# ensure src/ is on path for editable installs
_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

import torch

def _jit_init():
    """On first import: detect GPU, JIT-build kernel, inject into sys.modules."""
    if not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor

    # package_dir = .../flashdiffusion/  (works for both pip and editable)
    package_dir = os.path.dirname(os.path.abspath(__file__))

    from .kernel_jit import _build_kernel
    try:
        _build_kernel(sm, package_dir)
    except Exception as e:
        print(f'[FlashDiffusion] JIT build failed ({e}), CPU fallback only')

_jit_init()
