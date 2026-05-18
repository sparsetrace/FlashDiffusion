# src/flashdiffusion/kernel_jit.py
# Called on first import if no compiled .so found.
# Clones CUTLASS if needed, builds the right kernel for the current GPU.

import os, sys, glob, subprocess, sysconfig, time
import torch
from torch.utils.cpp_extension import load

_CACHE_DIR = os.path.expanduser('~/.cache/flashdiffusion')

def _build_kernel(sm, repo_root, verbose=False):
    """JIT-build the right kernel for this GPU. Returns loaded module."""

    # fix 1: no local import of sys — use module-level sys
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
    cu_file, arch, needs_cutlass, mod_name = _SOURCES.get(sm, _SOURCES[80])  # fix 2: keep mod_name from tuple

    cu_path   = os.path.join(repo_root, 'src', 'flashdiffusion', 'csrc', cu_file)
    # removed: mod_name = os.path.splitext(cu_file)[0]  ← this was overwriting the correct name

    build_dir = os.path.join(_CACHE_DIR, f'sm{sm}')
    os.makedirs(build_dir, exist_ok=True)

    # check if already built
    existing = glob.glob(f'{build_dir}/{mod_name}*.so')
    if existing:
        return _load_so(existing[0], mod_name)

    # CUTLASS clone if needed
    cutlass_inc = None
    if needs_cutlass:
        cutlass_inc = _ensure_cutlass(repo_root)

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
        name             = mod_name,
        sources          = [cu_path],
        extra_include_paths = includes,
        extra_cuda_cflags = [
            f'-arch={arch}', '-std=c++17', '-O3', '--use_fast_math',
            f'-gencode=arch={compute},code={arch}',
            '-U__CUDA_NO_HALF_OPERATORS__',
            '-U__CUDA_NO_HALF_CONVERSIONS__',
            '-U__CUDA_NO_HALF2_OPERATORS__',
            '--expt-relaxed-constexpr',
            '--expt-extended-lambda',
        ] + extra_defines,
        extra_cflags    = ['-O3'],
        build_directory = build_dir,
        verbose         = verbose,
    )
    print(f'[FlashDiffusion] Built in {time.perf_counter()-t0:.0f}s')

    so = glob.glob(f'{build_dir}/*.so')[0]
    return _load_so(so, mod_name)


def _load_so(so_path, mod_name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(mod_name, so_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[mod_name] = mod

    # legacy names that kernel_cuda_*.py files expect to import
    _LEGACY = {
        'flash_diffusion_sm80':  'flash_diffusion_cuda',   # kernel_cuda.py
        'flash_diffusion_sm90':  'flash_diffusion_sm90',   # kernel_cuda_sm90.py
        'flash_diffusion_sm100': 'flash_diffusion_sm100',  # kernel_cuda_sm100.py
        'flash_diffusion_sm120': 'flash_diffusion_sm120',  # kernel_cuda_sm120.py
    }
    for name in set([mod_name, _LEGACY.get(mod_name, mod_name)]):
        sys.modules[name] = mod

    return mod


def _ensure_cutlass(repo_root):
    # prefer submodule → /tmp/cutlass_main → auto-clone
    sub = os.path.join(repo_root, 'third_party', 'cutlass', 'include')
    if os.path.exists(os.path.join(sub, 'cute', 'tensor.hpp')):
        return sub
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
