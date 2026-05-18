# src/flashdiffusion/kernel_jit.py
# Called on first import if no compiled .so found.
# Clones CUTLASS if needed, builds the right kernel for the current GPU.

import os, sys, glob, subprocess, sysconfig, time
import torch
from torch.utils.cpp_extension import load

_CACHE_DIR = os.path.expanduser('~/.cache/flashdiffusion')


def _find_csrc(package_dir):
    """
    Find the csrc directory containing .cu files.
    Works for both pip install and editable install:
      pip install:      site-packages/flashdiffusion/csrc/
      editable install: repo/src/flashdiffusion/csrc/
    package_dir is os.path.dirname(__file__) from __init__.py
    """
    # pip install: csrc/ sits next to __init__.py
    p1 = os.path.join(package_dir, 'csrc')
    if os.path.exists(p1):
        return p1
    # editable install: package_dir = repo/src/flashdiffusion
    # csrc is already there — same as p1, but check repo root layout too
    p2 = os.path.join(package_dir, 'src', 'flashdiffusion', 'csrc')
    if os.path.exists(p2):
        return p2
    raise FileNotFoundError(
        f'FlashDiffusion csrc/ not found near {package_dir}. '
        f'Reinstall with: pip install flashdiffusion')


def _build_kernel(sm, package_dir, verbose=False):
    """JIT-build the right kernel for this GPU. Returns loaded module."""

    import importlib
    if importlib.util.find_spec('ninja') is None:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '-q', 'ninja'])

    _SOURCES = {
        80:  ('flash_diffusion_sm80.cu',  'sm_80',  False, 'flash_diffusion_cuda'),
        86:  ('flash_diffusion_sm80.cu',  'sm_80',  False, 'flash_diffusion_cuda'),
        89:  ('flash_diffusion_sm80.cu',  'sm_80',  False, 'flash_diffusion_cuda'),
        90:  ('flash_diffusion_sm80.cu',  'sm_80',  False, 'flash_diffusion_cuda'),
        100: ('flash_diffusion_sm80.cu',  'sm_80',  False, 'flash_diffusion_cuda'),
        103: ('flash_diffusion_sm80.cu',  'sm_80',  False, 'flash_diffusion_cuda'),
        120: ('flash_diffusion_sm120.cu', 'sm_120', False, 'flash_diffusion_sm120'),
        121: ('flash_diffusion_sm120.cu', 'sm_120', False, 'flash_diffusion_sm120'),
    }
    cu_file, arch, needs_cutlass, mod_name = _SOURCES.get(sm, _SOURCES[80])

    # find csrc — works for pip install and editable install
    csrc_dir = _find_csrc(package_dir)
    cu_path  = os.path.join(csrc_dir, cu_file)

    if not os.path.exists(cu_path):
        raise FileNotFoundError(
            f'CUDA source not found: {cu_path}\n'
            f'Reinstall with: pip install --force-reinstall flashdiffusion')

    build_dir = os.path.join(_CACHE_DIR, f'sm{sm}')
    os.makedirs(build_dir, exist_ok=True)

    # check if already built
    existing = glob.glob(f'{build_dir}/{mod_name}*.so')
    if existing:
        return _load_so(existing[0], mod_name)

    # CUTLASS clone if needed
    cutlass_inc = None
    if needs_cutlass:
        cutlass_inc = _ensure_cutlass(package_dir)

    # gather nvidia includes
    py = f'python{sys.version_info.major}.{sys.version_info.minor}'
    nvidia_incs = glob.glob(
        f'/usr/local/lib/{py}/site-packages/nvidia/*/include')
    includes = ([cutlass_inc] if cutlass_inc else []) + \
               [sysconfig.get_path('include')] + nvidia_incs

    extra_defines = []
    if sm in (90,):
        extra_defines = ['-DCUTLASS_ARCH_MMA_SM90_SUPPORTED=1',
                         '-DCUTE_ARCH_MMA_SM90A_ENABLED=1']
    if sm in (100, 103):
        extra_defines = ['-DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1',
                         '-DCUTE_ARCH_MMA_SM100A_ENABLED=1']

    compute = arch.replace('sm_', 'compute_').rstrip('a')

    print(f'[FlashDiffusion] Building SM{sm} kernel ({arch})...')
    t0 = time.perf_counter()
    load(
        name                = mod_name,
        sources             = [cu_path],
        extra_include_paths = includes,
        extra_cuda_cflags   = [
            f'-arch={arch}', '-std=c++17', '-O3', '--use_fast_math',
            f'-gencode=arch={compute},code={arch}',
            '-U__CUDA_NO_HALF_OPERATORS__',
            '-U__CUDA_NO_HALF_CONVERSIONS__',
            '-U__CUDA_NO_HALF2_OPERATORS__',
            '--expt-relaxed-constexpr',
            '--expt-extended-lambda',
        ] + extra_defines,
        extra_cflags        = ['-O3'],
        build_directory     = build_dir,
        verbose             = verbose,
    )
    print(f'[FlashDiffusion] Built in {time.perf_counter()-t0:.0f}s')

    so = glob.glob(f'{build_dir}/*.so')[0]
    return _load_so(so, mod_name)


def _load_so(so_path, mod_name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(mod_name, so_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # inject under mod_name and any legacy name kernel_cuda_*.py expects
    _LEGACY = {
        'flash_diffusion_sm80':  'flash_diffusion_cuda',
        'flash_diffusion_sm90':  'flash_diffusion_sm90',
        'flash_diffusion_sm100': 'flash_diffusion_sm100',
        'flash_diffusion_sm120': 'flash_diffusion_sm120',
    }
    for name in set([mod_name, _LEGACY.get(mod_name, mod_name)]):
        sys.modules[name] = mod

    return mod


def _ensure_cutlass(package_dir):
    """Find or clone CUTLASS headers."""
    # check third_party submodule relative to repo root
    # package_dir = repo/src/flashdiffusion (editable) or site-packages/flashdiffusion (pip)
    for base in [package_dir,
                 os.path.dirname(package_dir),
                 os.path.dirname(os.path.dirname(package_dir))]:
        sub = os.path.join(base, 'third_party', 'cutlass', 'include')
        if os.path.exists(os.path.join(sub, 'cute', 'tensor.hpp')):
            return sub

    # check /tmp session clone
    tmp = '/tmp/cutlass_main/include'
    if os.path.exists(os.path.join(tmp, 'cute', 'tensor.hpp')):
        return tmp

    # auto-clone
    target = '/tmp/cutlass_main'
    print('[FlashDiffusion] Cloning CUTLASS (one-time, ~2min)...')
    subprocess.check_call(
        ['git', 'clone', '--depth=1',
         'https://github.com/NVIDIA/cutlass', target],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmp
