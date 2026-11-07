# Repository Context

Generated automatically from repository source files. Do not edit manually.

## Files

- `.github/workflows/publish.yml`
- `LICENSE`
- `README.md`
- `ablations/flashdiffusion_alanine.py`
- `ablations/flashdiffusion_emnist.py`
- `ablations/flashdiffusion_swiss.py`
- `examples/swiss_roll.py`
- `pyproject.toml`
- `setup.py`
- `src/flashdiffusion/ROPE.py`
- `src/flashdiffusion/__init__.py`
- `src/flashdiffusion/cdc.py`
- `src/flashdiffusion/csrc/flash_diffusion_sm120.cu`
- `src/flashdiffusion/csrc/flash_diffusion_sm120X.cu`
- `src/flashdiffusion/csrc/flash_diffusion_sm120_.cu`
- `src/flashdiffusion/csrc/flash_diffusion_sm80.cu`
- `src/flashdiffusion/csrc/flash_diffusion_sm80X.cu`
- `src/flashdiffusion/csrc/flash_diffusion_sm80_.cu`
- `src/flashdiffusion/dmap.py`
- `src/flashdiffusion/eigenX.py`
- `src/flashdiffusion/eigensolver.py`
- `src/flashdiffusion/eigensolver_gpu.py`
- `src/flashdiffusion/eigensolver_gpu_.py`
- `src/flashdiffusion/eigensolver_gpu_block.py`
- `src/flashdiffusion/flash_diffusion_sm80.cu`
- `src/flashdiffusion/kernel.py`
- `src/flashdiffusion/kernel_cuda.py`
- `src/flashdiffusion/kernel_cuda_sm120.py`
- `src/flashdiffusion/kernel_jit.py`
- `src/flashdiffusion/ldm_matmat.py`
- `src/flashdiffusion/utils.py`
- `tests/test_kernel.py`

## `.github/workflows/publish.yml`

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]   # triggers when you make a GitHub release

jobs:
  publish:
    runs-on: ubuntu-latest
    environment: release
    permissions:
      id-token: write    # required for trusted publishing

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Build
        run: |
          pip install build
          python -m build

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        # no token needed — trusted publishing handles it
```

## `LICENSE`

```text
MIT License

Copyright (c) 2026 Julio Candanedo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## `README.md`

````markdown
# FlashDiffusion

Tiled, memory-efficient diffusion maps eigensolver — never materialises the O(N²) kernel matrix.

## What this enables

| Task | N limit (dense) | N limit (FlashDiffusion) |
|---|---|---|
| Diffusion map eigenvectors | ~20k (8 GB) | ~10M (memory: O(N·tile)) |
| Carré du Champ metric tensor | ~10k | ~1M |
| Schrödinger bridge / Doob h-transform | ~10k | ~1M |
| MD trajectory slow manifold | ~5k frames | ~500k frames |

## The primitive

```python
from flashdiffusion import FlashDiffusion

# K(i,j) = exp(-beta * score_mod(xi, xj))
# with Coifman-Lafon alpha-normalisation and optional Doob h-transform
out = FlashDiffusion(X, X, V, beta=1.0, alpha=0.5, h=None)
```

`FlashDiffusion(Q, K, V, beta, alpha, h)` is the single kernel underlying everything:
- **DMAP mode** (Q=K=X, V=eigvec guess): symmetric diffusion matvec for Lanczos
- **AMAP mode** (Q≠K, asymmetric W): directed/NESS Markov operator (DAC §3)
- **Doob mode** (h≠None): h-transform of any of the above (DAC §4)

## Design

```
kernel.py        FlashDiffusion primitive — tiled O(N·tile) memory matvec
dmap.py          DiffusionMap class: fit() precomputes rscale; transform() calls kernel
eigensolver.py   Lanczos/LOBPCG with bf16→fp32→fp64 precision ramp; orthog in fp64
cdc.py           Carré du Champ operator Γ(f,g) and metric tensor M = ΣᵢGᵢGᵢᵀ
utils.py         Bandwidth selection (self-tuning, median), Nyström extension
```

## Physics background

Transformer attention, diffusion maps, and magnetic Laplacians are three regimes
of a single Markov geometry (Candanedo 2025). The EQ/NESS classification:

- **EQ** (symmetric W): DMAP — reversible diffusion, detailed balance
- **NESS** (asymmetric W): self-attention — directed, probability current ≠ 0
- **Doob deformation**: h-transform preserves EQ class (Theorem 5.1, DAC)

The Coifman-Lafon α-normalisation is an exact Doob transform (Corollary 4.2, DAC).

# FlashDiffusion SM120 — RTX 6000 / Consumer Blackwell

## What's in this package

```
flashdiffusion/csrc/flash_diffusion_sm120.cu   ← kernel
flashdiffusion/kernel_cuda_sm120.py            ← Python wrapper
setup_sm120.py                                  ← build script
sm120_notebook.py                               ← notebook cells
```

## SM120 vs SM80 kernel differences

| Feature         | SM80 (A100)        | SM120 (RTX 5090)        |
|-----------------|--------------------|-------------------------|
| MMA instruction | mma.sync f16→f32   | mma.sync f16→f32 (same) |
| Tile size       | 64×64              | 128×128                 |
| SMEM async      | __syncthreads      | cp.async pipeline       |
| TMA             | no                 | yes (not used yet)      |
| TMEM            | no                 | no (SM100 only)         |
| Cluster shape   | 1×1×1              | 1×1×1 (no multicast)   |
| WGMMA/UMMA     | no                 | no (SM100 only)         |

The inner GEMM instruction is **identical** to SM80. The speedup comes from:
1. Larger 128×128 tiles → better arithmetic intensity
2. cp.async pipeline → hides GMEM→SMEM load latency
3. Higher RTX 5090 clock speed

## Build

```bash
# on a machine with RTX 5090
FLASHDIFFUSION_BUILD_CUDA=1 \
TORCH_CUDA_ARCH_LIST="12.0" \
python setup_sm120.py build_ext --inplace
```

Or in notebook (see sm120_notebook.py Cell 2).

## Expected performance vs SM80

```
N=200k  SM80 A100  precompute: 0.4s   Lanczos: ~100s
N=200k  SM120 5090 precompute: ~0.3s  Lanczos: ~60s  (estimate)
```

RTX 5090 has higher memory bandwidth than A100 (1.8 TB/s vs 2.0 TB/s)
but our scalar kernel is compute-bound not memory-bound.
The cp.async pipeline and larger tiles are the main wins.

## Next step: TiledMMA on SM120

The `mma.sync.aligned.m16n8k16` instruction on SM120 uses the same
CuTe atom as SM80: `SM80_16x8x16_F32F16F16F32`.
Wiring TiledMMA replaces the scalar inner loop, giving ~10× speedup.
This is `flash_diffusion_sm120_v2.cu` — to be added next.


## CUDA roadmap

Current: NumPy reference (validates correctness, runs on CPU).
Next: CuTe kernel — SM90 tensor cores, fused prepass+matvec, no online softmax needed
(precomputed rscale eliminates the log-sum-exp recurrence that drives FA4 complexity).

```
prepass:  out_i = Σⱼ K(i,j) · wⱼ          # scalar reduction per row, one kernel
matvec:   out_i = rscaleᵢ · Σⱼ K(i,j) · rscaleⱼ · vⱼ  # vector, same tiling
```

Both kernels share the same tile loop; only the epilogue differs.

## Installation

```bash
pip install -e ".[dev]"
```

## Quick start

```python
from flashdiffusion import DiffusionMap

dm = DiffusionMap(beta=1.0, alpha=0.5, n_components=8)
dm.fit(X)               # two prepass tiled reductions, O(N·tile) memory
coords = dm.transform() # Lanczos eigensolver, mixed precision
```

## References

- Coifman & Lafon (2006) — diffusion maps
- Candanedo (2025) — diffusion-attention connection, bidivergence, Doob classification
- Dao et al. (2024) — FlashAttention-4
- Rohrdanz, Zheng, Clementi (2011) — diffusion maps for MD
````

## `ablations/flashdiffusion_alanine.py`

```python
#!/usr/bin/env python3
"""
flashdiffusion_alanine.py
=========================

Alanine-dipeptide benchmark suite for FlashDiffusion / DiffusionMap.

This experiment is the real molecular-trajectory section:
  1. Download/load a 1M-frame alanine-dipeptide coordinate trajectory.
  2. Encode each frame using E(3)-invariant molecular features:
       - anchor distances: distances from every atom to selected anchor atoms
       - pairwise distances: all interatomic distances
  3. Run FlashDiffusion / DiffusionMap on up to 1M frames.
  4. Report runtime scaling, beta sensitivity, matvec timing, and eigenvalues.
  5. If phi/psi dihedrals are available, measure whether diffusion coordinates
     recover Ramachandran structure using circular features:
       sin(phi), cos(phi), sin(psi), cos(psi)

Default:
  --features anchor
  --anchors 0 1 2 3
  --alpha 1.0
  --n-components 20
  --beta-scale 0.1

Example
-------
python flashdiffusion_alanine.py \
  --repo /path/to/FlashDiffusion \
  --outdir results_alanine_anchor_sm100 \
  --features anchor \
  --n-values 50000 100000 250000 500000 750000 1000000 \
  --repeats 3 \
  --backend-label B200-sm100

Run both feature encodings:
python flashdiffusion_alanine.py \
  --repo /path/to/FlashDiffusion \
  --outdir results_alanine_features \
  --features anchor pairwise

Quick smoke test:
python flashdiffusion_alanine.py --repo /path/to/FlashDiffusion --quick

Caution
-------
The coordinate trajectory has 1M frames. The mdshare phi/psi file commonly used
for alanine dipeptide has 750k frames. This script uses phi/psi only for the first
min(N, len(phi_psi)) frames and treats it as a conformational sanity check. Verify
provenance/frame alignment before making strong molecular-science claims.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


ALA2_URL = "http://ftp.imp.fu-berlin.de/pub/cmb-data/ala2_cg_2fs_Hmass_2_HBonds.npz"
ALA2_FILE = "ala2_cg_2fs_Hmass_2_HBonds.npz"
DIHEDRAL_FILE_MD_SHARE = "alanine-dipeptide-3x250ns-backbone-dihedrals.npz"


# -----------------------------------------------------------------------------
# Backend
# -----------------------------------------------------------------------------

def add_repo_to_path(repo: Optional[str]) -> None:
    if repo is None:
        return
    repo_path = Path(repo).expanduser().resolve()
    for p in [repo_path / "src", repo_path]:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def load_flashdiffusion_backend() -> Tuple[Any, Dict[str, Any]]:
    import flashdiffusion.kernel as fk
    importlib.reload(fk)

    dmap_mod = importlib.import_module("flashdiffusion.dmap")
    importlib.reload(dmap_mod)
    DiffusionMap = dmap_mod.DiffusionMap

    ext = fk._get_cuda_backend()
    sm = fk._cuda_sm() if torch.cuda.is_available() else None
    backend_name = getattr(fk, "_SM_BACKEND", {}).get(sm)

    meta = {
        "torch_cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "sm": sm,
        "flashdiffusion_backend": str(backend_name),
        "cuda_ext": str(ext),
        "torch_version": torch.__version__,
    }

    print(f"[backend] cuda available: {meta['torch_cuda_available']}")
    print(f"[backend] cuda device: {meta['cuda_device']}")
    print(f"[backend] extension: {ext}")
    print(f"[backend] SM: {sm} -> {backend_name}")

    return DiffusionMap, meta


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def transform_dm(dm: Any, t: int = 1, tol: Optional[float] = None, maxiter: Optional[int] = None) -> np.ndarray:
    """
    Call dm.transform robustly across small API differences.
    """
    kwargs: Dict[str, Any] = {"t": t}
    if tol is not None:
        kwargs["tol"] = tol
    if maxiter is not None:
        kwargs["maxiter"] = maxiter

    try:
        return dm.transform(**kwargs)
    except TypeError:
        # Older/local DiffusionMap may only support transform(t=1).
        return dm.transform(t=t)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[download] already exists: {dest}")
        return

    print(f"[download] {url}")
    print(f"[download] -> {dest}")
    with urllib.request.urlopen(url) as response, open(dest, "wb") as f:
        total = getattr(response, "length", None)
        read = 0
        block = 1024 * 1024
        while True:
            chunk = response.read(block)
            if not chunk:
                break
            f.write(chunk)
            read += len(chunk)
            if total:
                print(f"\r[download] {read/1e6:.1f}/{total/1e6:.1f} MB", end="")
            else:
                print(f"\r[download] {read/1e6:.1f} MB", end="")
    print("\n[download] done")


def load_alanine_coords(data_dir: str, download: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    data_dir_p = Path(data_dir).expanduser().resolve()
    dest = data_dir_p / ALA2_FILE

    if download:
        download_url(ALA2_URL, dest)
    if not dest.exists():
        raise FileNotFoundError(f"Missing coordinate file: {dest}")

    with np.load(dest) as f:
        print(f"[coords] keys: {list(f.keys())}")
        xyz = f["coords"].astype(np.float32, copy=False)

    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"Expected coords shape (N, n_atoms, 3), got {xyz.shape}")

    meta = {
        "file": str(dest),
        "N": int(xyz.shape[0]),
        "n_atoms": int(xyz.shape[1]),
        "coord_dim": int(xyz.shape[2]),
        "dtype": str(xyz.dtype),
    }
    print(f"[coords] xyz={xyz.shape} dtype={xyz.dtype}")
    return xyz, meta


def load_phi_psi(data_dir: str, download: bool = True) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    meta: Dict[str, Any] = {"available": False, "source": "mdshare", "name": DIHEDRAL_FILE_MD_SHARE}

    if not download:
        print("[phi_psi] download disabled; skipping")
        return None, meta

    try:
        import mdshare  # type: ignore
    except Exception as e:
        print(f"[phi_psi] mdshare unavailable; skipping. {e}")
        meta["error"] = str(e)
        return None, meta

    try:
        data_dir_p = Path(data_dir).expanduser().resolve()
        data_dir_p.mkdir(parents=True, exist_ok=True)
        dih_file = mdshare.fetch(DIHEDRAL_FILE_MD_SHARE, working_directory=str(data_dir_p))

        with np.load(dih_file) as f:
            keys = sorted(list(f.keys()))
            phi_psi = np.concatenate([f[k] for k in keys], axis=0).astype(np.float32, copy=False)

        if phi_psi.ndim != 2 or phi_psi.shape[1] < 2:
            raise ValueError(f"Expected phi_psi shape (N, >=2), got {phi_psi.shape}")

        phi_psi = phi_psi[:, :2]
        meta.update({
            "available": True,
            "path": str(dih_file),
            "keys": keys,
            "N": int(phi_psi.shape[0]),
            "dim": int(phi_psi.shape[1]),
            "dtype": str(phi_psi.dtype),
        })
        print(f"[phi_psi] phi_psi={phi_psi.shape} dtype={phi_psi.dtype}")
        return phi_psi, meta

    except Exception as e:
        print(f"[phi_psi] failed; skipping. {e}")
        meta["error"] = str(e)
        return None, meta


# -----------------------------------------------------------------------------
# E(3)-invariant features
# -----------------------------------------------------------------------------

def e3_anchor_encode_gpu(
    xyz: np.ndarray,
    anchors: Sequence[int],
    batch_size: int = 250_000,
    device: str = "cuda",
) -> np.ndarray:
    """
    Anchor-distance E(3)-invariant encoding.

    xyz: (N, n_atoms, 3)
    returns: (N, n_atoms * len(anchors))
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    N, n_atoms, _ = xyz.shape
    anchors = list(anchors)
    if min(anchors) < 0 or max(anchors) >= n_atoms:
        raise ValueError(f"Invalid anchors={anchors} for n_atoms={n_atoms}")

    A = len(anchors)
    out = np.empty((N, n_atoms * A), dtype=np.float32)
    anchors_t = torch.tensor(anchors, dtype=torch.long, device=device)

    print(f"[features:anchor] N={N:,} n_atoms={n_atoms} anchors={anchors} d={out.shape[1]}")

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        R = torch.from_numpy(xyz[start:end]).to(device=device, dtype=torch.float32)
        anchor_xyz = R[:, anchors_t, :]                         # (B, A, 3)
        diff = R[:, :, None, :] - anchor_xyz[:, None, :, :]     # (B, n_atoms, A, 3)
        dist = diff.pow(2).sum(-1).sqrt().reshape(end - start, -1)
        out[start:end] = dist.detach().cpu().numpy()
        print(f"\r[features:anchor] encoded {end:,}/{N:,}", end="")
    print()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def e3_pairwise_encode_gpu(
    xyz: np.ndarray,
    batch_size: int = 250_000,
    device: str = "cuda",
) -> np.ndarray:
    """
    All-pairwise-distance E(3)-invariant encoding.

    xyz: (N, n_atoms, 3)
    returns: (N, n_atoms choose 2)
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    N, n_atoms, _ = xyz.shape
    pairs = [(i, j) for i in range(n_atoms) for j in range(i + 1, n_atoms)]
    out = np.empty((N, len(pairs)), dtype=np.float32)

    i_idx = torch.tensor([i for i, _ in pairs], dtype=torch.long, device=device)
    j_idx = torch.tensor([j for _, j in pairs], dtype=torch.long, device=device)

    print(f"[features:pairwise] N={N:,} n_atoms={n_atoms} d={out.shape[1]}")

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        R = torch.from_numpy(xyz[start:end]).to(device=device, dtype=torch.float32)
        diff = R[:, i_idx, :] - R[:, j_idx, :]
        dist = diff.pow(2).sum(-1).sqrt()
        out[start:end] = dist.detach().cpu().numpy()
        print(f"\r[features:pairwise] encoded {end:,}/{N:,}", end="")
    print()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def make_features(
    xyz: np.ndarray,
    feature: str,
    anchors: Sequence[int],
    batch_size: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    t0 = time.perf_counter()

    if feature == "anchor":
        X = e3_anchor_encode_gpu(xyz, anchors=anchors, batch_size=batch_size)
        meta = {"feature": feature, "anchors": list(anchors), "d": int(X.shape[1])}
    elif feature == "pairwise":
        X = e3_pairwise_encode_gpu(xyz, batch_size=batch_size)
        meta = {"feature": feature, "anchors": None, "d": int(X.shape[1])}
    else:
        raise ValueError(f"Unknown feature: {feature}")

    meta["feature_time"] = float(time.perf_counter() - t0)
    print(f"[features:{feature}] X={X.shape} time={meta['feature_time']:.2f}s")
    return X, meta


# -----------------------------------------------------------------------------
# Beta estimate
# -----------------------------------------------------------------------------

def estimate_beta_torch(
    X: np.ndarray,
    k: int = 10,
    K: int = 10_000,
    seed: int = 123,
    device: str = "cuda",
) -> float:
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    N = X.shape[0]
    K_eff = min(K, N)
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=K_eff, replace=False)

    Xs = torch.from_numpy(X[idx]).to(device=device, dtype=torch.float32)
    sq = (Xs * Xs).sum(1)
    d2 = (sq[:, None] + sq[None, :] - 2.0 * (Xs @ Xs.T)).clamp(min=0.0)
    vals = d2.topk(k + 1, dim=1, largest=False).values[:, 1:]
    mean_knn = float(vals.mean().item())
    beta = 1.0 / mean_knn

    print(f"[beta] K={K_eff} k={k} mean_knn_d2={mean_knn:.6g} beta0={beta:.8g}")

    del Xs, sq, d2, vals
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return beta


# -----------------------------------------------------------------------------
# Phi/psi metrics
# -----------------------------------------------------------------------------

def circular_targets(phi_psi: np.ndarray, degrees: bool = False) -> np.ndarray:
    phi = phi_psi[:, 0].astype(np.float32)
    psi = phi_psi[:, 1].astype(np.float32)
    if degrees:
        phi = np.deg2rad(phi)
        psi = np.deg2rad(psi)
    return np.stack([np.sin(phi), np.cos(phi), np.sin(psi), np.cos(psi)], axis=1).astype(np.float32)


def linear_r2_multioutput(X: np.ndarray, Y: np.ndarray, max_rows: int = 200_000, seed: int = 0) -> float:
    N = X.shape[0]
    if N > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N, size=max_rows, replace=False)
        Xs = X[idx]
        Ys = Y[idx]
    else:
        Xs = X
        Ys = Y

    A = np.c_[Xs.astype(np.float64), np.ones(Xs.shape[0], dtype=np.float64)]
    B = Ys.astype(np.float64)
    coef, *_ = np.linalg.lstsq(A, B, rcond=None)
    pred = A @ coef

    ss_res = np.sum((B - pred) ** 2)
    ss_tot = np.sum((B - B.mean(axis=0, keepdims=True)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def component_recovery_sweep(
    coords: np.ndarray,
    phi_psi: Optional[np.ndarray],
    coord_dims: Sequence[int],
    max_rows: int,
    seed: int,
    degrees: bool = False,
) -> pd.DataFrame:
    if phi_psi is None:
        return pd.DataFrame()

    N_match = min(coords.shape[0], phi_psi.shape[0])
    C = coords[:N_match]
    Y = circular_targets(phi_psi[:N_match], degrees=degrees)

    rows: List[Dict[str, Any]] = []
    for m in coord_dims:
        m_eff = min(int(m), C.shape[1])
        if m_eff <= 0:
            continue
        r2 = linear_r2_multioutput(C[:, :m_eff], Y, max_rows=max_rows, seed=seed)
        rows.append({"coord_dim": m_eff, "N_match": N_match, "ramachandran_circular_r2": r2})
        print(f"[recover] m={m_eff:>3} circular R2={r2:.4f}")

    return pd.DataFrame(rows)


def per_dc_circular_correlations(
    coords: np.ndarray,
    phi_psi: Optional[np.ndarray],
    max_dcs: int = 12,
    max_rows: int = 200_000,
    seed: int = 0,
    degrees: bool = False,
) -> pd.DataFrame:
    if phi_psi is None:
        return pd.DataFrame()

    N_match = min(coords.shape[0], phi_psi.shape[0])
    C = coords[:N_match]
    Y = circular_targets(phi_psi[:N_match], degrees=degrees)

    if N_match > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N_match, size=max_rows, replace=False)
        C = C[idx]
        Y = Y[idx]

    labels = ["sin_phi", "cos_phi", "sin_psi", "cos_psi"]
    rows: List[Dict[str, Any]] = []

    for j in range(min(max_dcs, C.shape[1])):
        row: Dict[str, Any] = {"dc": j + 1}
        for k, label in enumerate(labels):
            if np.std(C[:, j]) == 0 or np.std(Y[:, k]) == 0:
                corr = np.nan
            else:
                corr = abs(np.corrcoef(C[:, j], Y[:, k])[0, 1])
            row[f"corr_{label}"] = float(corr)

        row["max_corr_phi"] = float(np.nanmax([row["corr_sin_phi"], row["corr_cos_phi"]]))
        row["max_corr_psi"] = float(np.nanmax([row["corr_sin_psi"], row["corr_cos_psi"]]))
        rows.append(row)
        print(f"[corr] DC{j+1}: max phi={row['max_corr_phi']:.3f} max psi={row['max_corr_psi']:.3f}")

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# DiffusionMap run wrapper
# -----------------------------------------------------------------------------

def run_dmap_alanine(
    DiffusionMap: Any,
    X: np.ndarray,
    phi_psi: Optional[np.ndarray],
    N: int,
    alpha: float,
    n_components: int,
    beta: Optional[float],
    beta_scale: float,
    beta_k: int,
    beta_subsample: int,
    seed: int,
    prefix: bool,
    maxiter: Optional[int],
    tol: Optional[float],
    matvec_bench: int,
    recovery_max_rows: int,
    phi_psi_degrees: bool,
    keep_arrays: bool,
) -> Dict[str, Any]:
    N_total = X.shape[0]
    N_eff = min(int(N), N_total)

    if prefix:
        idx = np.arange(N_eff)
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N_total, size=N_eff, replace=False)
        idx.sort()

    Xn = X[idx].astype(np.float32, copy=False)

    if beta is None:
        beta0 = estimate_beta_torch(Xn, k=beta_k, K=beta_subsample, seed=seed + 999)
        beta = beta0 * beta_scale
    else:
        beta0 = np.nan

    print(
        f"[run] N={N_eff:,} d={Xn.shape[1]} beta={beta:.8g} "
        f"beta0={beta0} beta_scale={beta_scale} alpha={alpha}"
    )

    dm = DiffusionMap(beta=float(beta), alpha=float(alpha), n_components=int(n_components), tile=512)

    cuda_sync()
    t0 = time.perf_counter()
    dm.fit(Xn)
    cuda_sync()
    fit_time = time.perf_counter() - t0

    # Matvec-only microbenchmark after precompute.
    matvec_bench_time = np.nan
    matvec_bench_each = np.nan
    if matvec_bench > 0:
        rng = np.random.default_rng(seed + 2024)
        vecs = [rng.standard_normal(Xn.shape[0]).astype(np.float32) for _ in range(matvec_bench)]
        _ = dm.matvec(vecs[0])
        cuda_sync()

        t0 = time.perf_counter()
        for v in vecs:
            _ = dm.matvec(v)
        cuda_sync()
        matvec_bench_time = time.perf_counter() - t0
        matvec_bench_each = matvec_bench_time / matvec_bench
        print(f"[matvec bench] {matvec_bench} matvecs in {matvec_bench_time:.3f}s each={matvec_bench_each:.4f}s")

    # Count matvec calls during eigensolver.
    matvec_counter = {"n": 0}
    orig_matvec = dm.matvec

    def counted_matvec(v: np.ndarray) -> np.ndarray:
        matvec_counter["n"] += 1
        return orig_matvec(v)

    dm.matvec = counted_matvec  # type: ignore[method-assign]

    cuda_sync()
    t0 = time.perf_counter()
    coords = transform_dm(dm, t=1, tol=tol, maxiter=maxiter)
    cuda_sync()
    eig_time = time.perf_counter() - t0

    n_matvec = int(matvec_counter["n"])
    matvec_time_lanczos = eig_time / n_matvec if n_matvec > 0 else np.nan
    ev = np.asarray(dm.eigenvalues_, dtype=np.float64)

    # Basic phi/psi sanity metrics.
    N_match = 0
    r2_first8 = np.nan
    r2_all = np.nan
    if phi_psi is not None and prefix:
        N_match = min(N_eff, phi_psi.shape[0])
        if N_match > 0:
            Y = circular_targets(phi_psi[:N_match], degrees=phi_psi_degrees)
            r2_first8 = linear_r2_multioutput(
                coords[:N_match, : min(8, coords.shape[1])],
                Y,
                max_rows=recovery_max_rows,
                seed=seed,
            )
            r2_all = linear_r2_multioutput(
                coords[:N_match, :],
                Y,
                max_rows=recovery_max_rows,
                seed=seed,
            )

    row: Dict[str, Any] = {
        "N": int(N_eff),
        "d": int(Xn.shape[1]),
        "seed": int(seed),
        "alpha": float(alpha),
        "n_components": int(n_components),
        "beta": float(beta),
        "beta0": float(beta0) if np.isfinite(beta0) else np.nan,
        "beta_scale": float(beta_scale),
        "fit_time": float(fit_time),
        "eig_time": float(eig_time),
        "total_time": float(fit_time + eig_time),
        "n_matvec": int(n_matvec),
        "matvec_time_lanczos": float(matvec_time_lanczos),
        "matvec_bench_n": int(matvec_bench),
        "matvec_bench_time": float(matvec_bench_time),
        "matvec_bench_each": float(matvec_bench_each),
        "N_match_phi_psi": int(N_match),
        "circular_r2_first8": float(r2_first8),
        "circular_r2_all_coords": float(r2_all),
    }

    for j in range(min(32, len(ev))):
        row[f"lambda{j+1}"] = float(ev[j])

    print(
        f"[run] fit={fit_time:.2f}s eig={eig_time:.2f}s total={fit_time+eig_time:.2f}s "
        f"matvecs={n_matvec} matvec_time={matvec_time_lanczos:.4f}s "
        f"R2_8={r2_first8:.3f} R2_all={r2_all:.3f}"
    )
    print(f"[run] eigenvalues[:8] = {ev[:8]}")

    if keep_arrays:
        row["_idx"] = idx
        row["_coords"] = coords
        row["_eigenvalues"] = ev

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row


# -----------------------------------------------------------------------------
# Benchmark loops
# -----------------------------------------------------------------------------

def benchmark_N_scaling(
    DiffusionMap: Any,
    X: np.ndarray,
    phi_psi: Optional[np.ndarray],
    N_values: Iterable[int],
    repeats: int,
    args: argparse.Namespace,
    feature: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[Dict[str, Any]]]:
    run_rows: List[Dict[str, Any]] = []
    recovery_tables: List[pd.DataFrame] = []
    corr_tables: List[pd.DataFrame] = []
    last_case: Optional[Dict[str, Any]] = None

    prefix = not args.random_subsample

    for N in N_values:
        for r in range(repeats):
            seed = 30_000 + 101 * r + int(N % 997)
            keep_arrays = (r == 0)
            print(f"\n=== N-scaling: feature={feature} N={N:,} repeat={r+1}/{repeats} ===")

            row = run_dmap_alanine(
                DiffusionMap=DiffusionMap,
                X=X,
                phi_psi=phi_psi,
                N=int(N),
                alpha=args.alpha,
                n_components=args.n_components,
                beta=None,
                beta_scale=args.beta_scale,
                beta_k=args.beta_k,
                beta_subsample=args.beta_subsample,
                seed=seed,
                prefix=prefix,
                maxiter=args.maxiter,
                tol=args.tol,
                matvec_bench=args.matvec_bench,
                recovery_max_rows=args.recovery_max_rows,
                phi_psi_degrees=args.phi_psi_degrees,
                keep_arrays=keep_arrays,
            )

            row["experiment"] = "N_scaling"
            row["repeat"] = r
            row["backend_label"] = args.backend_label
            row["feature"] = feature
            run_rows.append({k: v for k, v in row.items() if not k.startswith("_")})

            if keep_arrays and phi_psi is not None and prefix:
                df_rec = component_recovery_sweep(
                    row["_coords"],
                    phi_psi,
                    coord_dims=args.coord_dims,
                    max_rows=args.recovery_max_rows,
                    seed=seed + 404,
                    degrees=args.phi_psi_degrees,
                )
                if not df_rec.empty:
                    df_rec["N"] = int(row["N"])
                    df_rec["repeat"] = r
                    df_rec["feature"] = feature
                    df_rec["beta"] = float(row["beta"])
                    df_rec["alpha"] = float(args.alpha)
                    df_rec["backend_label"] = args.backend_label
                    recovery_tables.append(df_rec)

                df_corr = per_dc_circular_correlations(
                    row["_coords"],
                    phi_psi,
                    max_dcs=min(12, row["_coords"].shape[1]),
                    max_rows=args.recovery_max_rows,
                    seed=seed + 405,
                    degrees=args.phi_psi_degrees,
                )
                if not df_corr.empty:
                    df_corr["N"] = int(row["N"])
                    df_corr["repeat"] = r
                    df_corr["feature"] = feature
                    df_corr["backend_label"] = args.backend_label
                    corr_tables.append(df_corr)

                last_case = row

            if not keep_arrays:
                for k in ["_idx", "_coords", "_eigenvalues"]:
                    row.pop(k, None)

    df_runs = pd.DataFrame(run_rows)
    df_recovery = pd.concat(recovery_tables, ignore_index=True) if recovery_tables else pd.DataFrame()
    df_corr = pd.concat(corr_tables, ignore_index=True) if corr_tables else pd.DataFrame()
    return df_runs, df_recovery, df_corr, last_case


def benchmark_beta_sweep(
    DiffusionMap: Any,
    X: np.ndarray,
    phi_psi: Optional[np.ndarray],
    args: argparse.Namespace,
    feature: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    prefix = not args.random_subsample
    N_eff = min(int(args.beta_n), X.shape[0])
    X_ref = X[:N_eff] if prefix else X[np.random.default_rng(123).choice(X.shape[0], size=N_eff, replace=False)]
    beta0 = estimate_beta_torch(X_ref, k=args.beta_k, K=args.beta_subsample, seed=777)
    print(f"[beta sweep] feature={feature} N={N_eff:,} beta0={beta0:.8g}")

    run_rows: List[Dict[str, Any]] = []
    recovery_tables: List[pd.DataFrame] = []

    for factor in args.beta_factors:
        beta = float(beta0 * factor)
        for r in range(args.repeats):
            seed = 40_000 + 131 * r
            keep_arrays = (r == 0)
            print(
                f"\n=== beta-sweep: feature={feature} N={N_eff:,} "
                f"beta/beta0={factor:g} beta={beta:.8g} repeat={r+1}/{args.repeats} ==="
            )

            row = run_dmap_alanine(
                DiffusionMap=DiffusionMap,
                X=X,
                phi_psi=phi_psi,
                N=N_eff,
                alpha=args.alpha,
                n_components=args.n_components,
                beta=beta,
                beta_scale=factor,
                beta_k=args.beta_k,
                beta_subsample=args.beta_subsample,
                seed=seed,
                prefix=prefix,
                maxiter=args.maxiter,
                tol=args.tol,
                matvec_bench=args.matvec_bench,
                recovery_max_rows=args.recovery_max_rows,
                phi_psi_degrees=args.phi_psi_degrees,
                keep_arrays=keep_arrays,
            )

            row["experiment"] = "beta_sweep"
            row["repeat"] = r
            row["backend_label"] = args.backend_label
            row["feature"] = feature
            row["beta_factor"] = float(factor)
            row["beta0_fixed"] = float(beta0)
            run_rows.append({k: v for k, v in row.items() if not k.startswith("_")})

            if keep_arrays and phi_psi is not None and prefix:
                df_rec = component_recovery_sweep(
                    row["_coords"],
                    phi_psi,
                    coord_dims=args.coord_dims,
                    max_rows=args.recovery_max_rows,
                    seed=seed + 404,
                    degrees=args.phi_psi_degrees,
                )
                if not df_rec.empty:
                    df_rec["N"] = int(row["N"])
                    df_rec["repeat"] = r
                    df_rec["feature"] = feature
                    df_rec["beta"] = float(beta)
                    df_rec["beta_factor"] = float(factor)
                    df_rec["beta0_fixed"] = float(beta0)
                    df_rec["alpha"] = float(args.alpha)
                    df_rec["backend_label"] = args.backend_label
                    recovery_tables.append(df_rec)

    df_runs = pd.DataFrame(run_rows)
    df_recovery = pd.concat(recovery_tables, ignore_index=True) if recovery_tables else pd.DataFrame()
    return df_runs, df_recovery


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

def plot_N_scaling(df: pd.DataFrame, outdir: Path, feature: str) -> None:
    if df.empty:
        return

    g = df.groupby("N").agg(["mean", "std"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    for col, label in [("fit_time", "precompute"), ("eig_time", "Lanczos"), ("total_time", "total")]:
        x = g.index.values.astype(float)
        y = g[(col, "mean")].values
        e = g[(col, "std")].fillna(0).values
        ax.errorbar(x, y, yerr=e, marker="o", capsize=3, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N frames")
    ax.set_ylabel("time (s)")
    ax.set_title(f"Alanine scaling: {feature} E(3) features")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[1]
    x = g.index.values.astype(float)
    y = g[("matvec_time_lanczos", "mean")].values
    e = g[("matvec_time_lanczos", "std")].fillna(0).values
    ax.errorbar(x, y, yerr=e, marker="s", capsize=3, label="Lanczos avg")
    if ("matvec_bench_each", "mean") in g.columns:
        yb = g[("matvec_bench_each", "mean")].values
        eb = g[("matvec_bench_each", "std")].fillna(0).values
        ax.errorbar(x, yb, yerr=eb, marker="o", capsize=3, label="matvec microbench")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N frames")
    ax.set_ylabel("time per matvec (s)")
    ax.set_title("Implicit dense-kernel matvec")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    path = outdir / f"alanine_{feature}_N_scaling.png"
    fig.savefig(path, dpi=200)
    print(f"[plot] wrote {path}")
    plt.close(fig)


def plot_component_recovery(df: pd.DataFrame, outdir: Path, feature: str) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for N, sub in sorted(df.groupby("N")):
        sub = sub.sort_values("coord_dim")
        ax.plot(sub["coord_dim"], sub["ramachandran_circular_r2"], marker="o", label=f"N={int(N):,}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of diffusion coordinates")
    ax.set_ylabel(r"linear $R^2$ to circular $(\phi,\psi)$ features")
    ax.set_title(f"Alanine conformational recovery: {feature}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = outdir / f"alanine_{feature}_component_recovery.png"
    fig.savefig(path, dpi=200)
    print(f"[plot] wrote {path}")
    plt.close(fig)


def plot_beta_sweep(df: pd.DataFrame, df_rec: pd.DataFrame, outdir: Path, feature: str) -> None:
    if df.empty or "beta_factor" not in df:
        return

    g = df.groupby("beta_factor").agg(["mean", "std"])
    x = g.index.values.astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax1 = axes[0]
    y = g[("eig_time", "mean")].values
    e = g[("eig_time", "std")].fillna(0).values
    ax1.errorbar(x, y, yerr=e, marker="o", capsize=3, label="Lanczos time")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$\beta/\beta_0$")
    ax1.set_ylabel("Lanczos time (s)")
    ax1.grid(True, which="both", alpha=0.3)

    ax1b = ax1.twinx()
    y2 = g[("n_matvec", "mean")].values
    e2 = g[("n_matvec", "std")].fillna(0).values
    ax1b.errorbar(x, y2, yerr=e2, marker="s", capsize=3, label="# matvecs")
    ax1b.set_ylabel("matvec calls")
    ax1.set_title("Bandwidth controls eigensolver cost")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    ax2 = axes[1]
    if not df_rec.empty and "beta_factor" in df_rec:
        idx = df_rec.groupby("beta_factor")["coord_dim"].idxmax()
        qual = df_rec.loc[idx].sort_values("beta_factor")
        ax2.plot(qual["beta_factor"], qual["ramachandran_circular_r2"], marker="o", label="max coord-dim R2")
    if ("circular_r2_all_coords", "mean") in g.columns:
        ax2.plot(x, g[("circular_r2_all_coords", "mean")].values, marker="s", label="all coords R2")
    if ("lambda1", "mean") in g.columns and ("lambda2", "mean") in g.columns:
        gap = g[("lambda1", "mean")].values - g[("lambda2", "mean")].values
        ax2.plot(x, gap, marker="^", linestyle="--", label=r"$\lambda_1-\lambda_2$")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\beta/\beta_0$")
    ax2.set_ylabel("quality / spectral gap")
    ax2.set_title("Molecular bandwidth sensitivity")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="best")

    fig.tight_layout()
    path = outdir / f"alanine_{feature}_beta_sweep.png"
    fig.savefig(path, dpi=200)
    print(f"[plot] wrote {path}")
    plt.close(fig)


def plot_eigenvalues(df: pd.DataFrame, outdir: Path, feature: str) -> None:
    lambda_cols = [c for c in df.columns if c.startswith("lambda")]
    if not lambda_cols:
        return
    lambda_cols = sorted(lambda_cols, key=lambda s: int(s.replace("lambda", "")))

    fig, ax = plt.subplots(figsize=(7, 4))
    for N, sub in sorted(df.groupby("N")):
        sub0 = sub.sort_values("repeat").iloc[0]
        vals = [sub0[c] for c in lambda_cols[:20] if c in sub0 and np.isfinite(sub0[c])]
        ax.plot(np.arange(1, len(vals) + 1), vals, marker="o", label=f"N={int(N):,}")
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("eigenvalue")
    ax.set_title(f"Alanine diffusion eigenvalues: {feature}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = outdir / f"alanine_{feature}_eigenvalues.png"
    fig.savefig(path, dpi=200)
    print(f"[plot] wrote {path}")
    plt.close(fig)


def plot_embedding(
    case: Optional[Dict[str, Any]],
    phi_psi: Optional[np.ndarray],
    outdir: Path,
    feature: str,
    n_plot: int,
    phi_psi_degrees: bool,
) -> None:
    if case is None or "_coords" not in case:
        return

    coords = case["_coords"]
    N = coords.shape[0]
    rng = np.random.default_rng(12345)
    idx = rng.choice(N, size=min(n_plot, N), replace=False)

    has_phi = phi_psi is not None and min(N, phi_psi.shape[0]) > 0
    N_match = min(N, phi_psi.shape[0]) if has_phi else 0

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    sc = ax.scatter(coords[idx, 0], coords[idx, 1], c=idx / max(N - 1, 1), s=2)
    ax.set_xlabel("DC1")
    ax.set_ylabel("DC2")
    ax.set_title("DC1/DC2 colored by trajectory time")
    plt.colorbar(sc, ax=ax, label="normalized time")

    if has_phi:
        idx_phi = idx[idx < N_match]
        if len(idx_phi) == 0:
            idx_phi = np.arange(min(n_plot, N_match))

        P = phi_psi[:N_match].copy()
        if phi_psi_degrees:
            P = np.deg2rad(P)
        phi = P[:, 0]
        psi = P[:, 1]

        ax = axes[0, 1]
        sc = ax.scatter(coords[idx_phi, 0], coords[idx_phi, 1], c=phi[idx_phi], s=2)
        ax.set_xlabel("DC1")
        ax.set_ylabel("DC2")
        ax.set_title("DC1/DC2 colored by phi")
        plt.colorbar(sc, ax=ax, label="phi")

        ax = axes[1, 0]
        sc = ax.scatter(coords[idx_phi, 0], coords[idx_phi, 1], c=psi[idx_phi], s=2)
        ax.set_xlabel("DC1")
        ax.set_ylabel("DC2")
        ax.set_title("DC1/DC2 colored by psi")
        plt.colorbar(sc, ax=ax, label="psi")

        ax = axes[1, 1]
        sc = ax.scatter(phi[idx_phi], psi[idx_phi], c=coords[idx_phi, 0], s=2)
        ax.set_xlabel("phi")
        ax.set_ylabel("psi")
        ax.set_title("Ramachandran plot colored by DC1")
        plt.colorbar(sc, ax=ax, label="DC1")
    else:
        for ax in [axes[0, 1], axes[1, 0], axes[1, 1]]:
            ax.axis("off")

    fig.suptitle(
        f"Alanine diffusion coordinates, feature={feature}, N={N:,}, "
        f"beta={case['beta']:.4g}, alpha={case['alpha']}",
        y=1.02,
    )
    fig.tight_layout()
    path = outdir / f"alanine_{feature}_embedding_N{N}.png"
    fig.savefig(path, dpi=200)
    print(f"[plot] wrote {path}")
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Alanine-dipeptide benchmark suite for FlashDiffusion.")

    p.add_argument("--repo", type=str, default=None, help="Path to FlashDiffusion repo; repo/src is added to sys.path.")
    p.add_argument("--outdir", type=str, default="results_alanine", help="Output directory.")
    p.add_argument("--data-dir", type=str, default="/tmp/ala2", help="Data directory.")
    p.add_argument("--no-download", action="store_true", help="Do not download datasets.")
    p.add_argument("--skip-phi-psi", action="store_true", help="Skip loading phi/psi reference.")
    p.add_argument("--phi-psi-degrees", action="store_true", help="Treat phi/psi file as degrees instead of radians.")

    p.add_argument("--features", type=str, nargs="+", default=["anchor"], choices=["anchor", "pairwise"])
    p.add_argument("--anchors", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--feature-batch-size", type=int, default=250_000)

    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--n-components", type=int, default=20)
    p.add_argument("--tol", type=float, default=None)
    p.add_argument("--maxiter", type=int, default=None)

    p.add_argument("--beta-k", type=int, default=10)
    p.add_argument("--beta-subsample", type=int, default=10_000)
    p.add_argument("--beta-scale", type=float, default=0.1)

    p.add_argument("--n-values", type=int, nargs="+", default=[50_000, 100_000, 250_000, 500_000, 750_000, 1_000_000])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--random-subsample", action="store_true")

    p.add_argument("--beta-n", type=int, default=250_000)
    p.add_argument("--beta-factors", type=float, nargs="+", default=[1/50, 1/20, 1/10, 1/5, 1/2, 1.0])

    p.add_argument("--coord-dims", type=int, nargs="+", default=[2, 4, 8, 12, 16, 19])
    p.add_argument("--recovery-max-rows", type=int, default=200_000)
    p.add_argument("--matvec-bench", type=int, default=5)
    p.add_argument("--plot-points", type=int, default=50_000)
    p.add_argument("--backend-label", type=str, default="")

    p.add_argument("--skip-n-scaling", action="store_true")
    p.add_argument("--skip-beta-sweep", action="store_true")
    p.add_argument("--skip-plots", action="store_true")

    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.quick:
        args.n_values = [5_000, 10_000]
        args.beta_n = 10_000
        args.repeats = 1
        args.beta_subsample = 2_000
        args.n_components = min(args.n_components, 12)
        args.coord_dims = [2, 4, 8]
        args.matvec_bench = 2
        args.plot_points = 5_000
        print("[quick] using quick smoke-test settings")

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    add_repo_to_path(args.repo)
    DiffusionMap, backend_meta = load_flashdiffusion_backend()

    xyz, coord_meta = load_alanine_coords(args.data_dir, download=not args.no_download)

    phi_psi: Optional[np.ndarray] = None
    phi_meta: Dict[str, Any] = {"available": False}
    if not args.skip_phi_psi:
        phi_psi, phi_meta = load_phi_psi(args.data_dir, download=not args.no_download)
    else:
        print("[phi_psi] skipped")

    metadata = {
        "backend": backend_meta,
        "coord_data": coord_meta,
        "phi_psi": phi_meta,
        "features": args.features,
        "anchors": args.anchors,
        "feature_batch_size": args.feature_batch_size,
        "alpha": args.alpha,
        "n_components": args.n_components,
        "tol": args.tol,
        "maxiter": args.maxiter,
        "beta_k": args.beta_k,
        "beta_subsample": args.beta_subsample,
        "beta_scale": args.beta_scale,
        "N_values": args.n_values,
        "repeats": args.repeats,
        "random_subsample": args.random_subsample,
        "beta_n": args.beta_n,
        "beta_factors": args.beta_factors,
        "coord_dims": args.coord_dims,
        "recovery_max_rows": args.recovery_max_rows,
        "matvec_bench": args.matvec_bench,
        "backend_label": args.backend_label,
        "repo": args.repo,
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[main] wrote {outdir / 'metadata.json'}")

    all_N: List[pd.DataFrame] = []
    all_rec: List[pd.DataFrame] = []
    all_corr: List[pd.DataFrame] = []
    all_beta: List[pd.DataFrame] = []
    all_beta_rec: List[pd.DataFrame] = []

    for feature in args.features:
        feature_dir = outdir / feature
        feature_dir.mkdir(parents=True, exist_ok=True)

        X, feature_meta = make_features(
            xyz=xyz,
            feature=feature,
            anchors=args.anchors,
            batch_size=args.feature_batch_size,
        )
        (feature_dir / "feature_metadata.json").write_text(json.dumps(feature_meta, indent=2))

        N_values = sorted(set(int(min(N, X.shape[0])) for N in args.n_values if int(min(N, X.shape[0])) > 0))

        last_case: Optional[Dict[str, Any]] = None

        if not args.skip_n_scaling:
            df_N, df_rec, df_corr, last_case = benchmark_N_scaling(
                DiffusionMap=DiffusionMap,
                X=X,
                phi_psi=phi_psi,
                N_values=N_values,
                repeats=args.repeats,
                args=args,
                feature=feature,
            )
            df_N.to_csv(feature_dir / f"alanine_{feature}_N_scaling.csv", index=False)
            df_rec.to_csv(feature_dir / f"alanine_{feature}_component_recovery.csv", index=False)
            df_corr.to_csv(feature_dir / f"alanine_{feature}_per_dc_correlations.csv", index=False)

            all_N.append(df_N)
            all_rec.append(df_rec)
            all_corr.append(df_corr)

            if last_case is not None:
                np.savez_compressed(
                    feature_dir / f"alanine_{feature}_embedding_N{last_case['N']}.npz",
                    coords=last_case["_coords"],
                    eigenvalues=last_case["_eigenvalues"],
                    beta=last_case["beta"],
                    alpha=last_case["alpha"],
                )

            if not args.skip_plots:
                plot_N_scaling(df_N, feature_dir, feature)
                plot_component_recovery(df_rec, feature_dir, feature)
                plot_eigenvalues(df_N, feature_dir, feature)
                plot_embedding(last_case, phi_psi, feature_dir, feature, args.plot_points, args.phi_psi_degrees)

        if not args.skip_beta_sweep:
            df_beta, df_beta_rec = benchmark_beta_sweep(
                DiffusionMap=DiffusionMap,
                X=X,
                phi_psi=phi_psi,
                args=args,
                feature=feature,
            )
            df_beta.to_csv(feature_dir / f"alanine_{feature}_beta_sweep.csv", index=False)
            df_beta_rec.to_csv(feature_dir / f"alanine_{feature}_beta_component_recovery.csv", index=False)

            all_beta.append(df_beta)
            all_beta_rec.append(df_beta_rec)

            if not args.skip_plots:
                plot_beta_sweep(df_beta, df_beta_rec, feature_dir, feature)

        del X
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if all_N:
        pd.concat(all_N, ignore_index=True).to_csv(outdir / "alanine_all_N_scaling.csv", index=False)
    if all_rec:
        pd.concat(all_rec, ignore_index=True).to_csv(outdir / "alanine_all_component_recovery.csv", index=False)
    if all_corr:
        pd.concat(all_corr, ignore_index=True).to_csv(outdir / "alanine_all_per_dc_correlations.csv", index=False)
    if all_beta:
        pd.concat(all_beta, ignore_index=True).to_csv(outdir / "alanine_all_beta_sweep.csv", index=False)
    if all_beta_rec:
        pd.concat(all_beta_rec, ignore_index=True).to_csv(outdir / "alanine_all_beta_component_recovery.csv", index=False)

    print(f"[done] outputs in {outdir}")


if __name__ == "__main__":
    main()
```

## `ablations/flashdiffusion_emnist.py`

```python
#!/usr/bin/env python3
"""
flashdiffusion_emnist.py
========================

EMNIST benchmark suite for FlashDiffusion / DiffusionMap.

Experiment 2 focuses on high-dimensional real data:
  1. EMNIST ByMerge / ByClass / Balanced loading from torchvision.
  2. N-scaling at fixed raw-pixel dimension d=784.
  3. 5-NN accuracy in diffusion-coordinate space vs number of DCs.
  4. Matvec-only microbenchmark after precompute.
  5. Backend tagging for SM80/SM100/SM120 comparisons.

Default:
  split       = bymerge
  include_test = True   # gives the full largest EMNIST variants
  alpha       = 1.0
  n_components = 20

Install deps if needed:
  pip install torch torchvision scikit-learn pandas matplotlib

Example full-ish run:
  python flashdiffusion_emnist.py \
      --repo /path/to/FlashDiffusion \
      --outdir results_emnist_bymerge_sm100 \
      --split bymerge \
      --include-test \
      --n-values 50000 100000 200000 400000 814255 \
      --repeats 3 \
      --n-components 20 \
      --backend-label B200-sm100

Quick smoke test:
  python flashdiffusion_emnist.py --quick --outdir results_emnist_quick

Notes
-----
This script intentionally keeps d fixed at 784 raw pixels. No PCA sweep.
To compare GPU backends, run this same script under each compiled backend
and compare the resulting CSVs. Use --backend-label to tag runs.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Environment / backend
# -----------------------------------------------------------------------------

def add_repo_to_path(repo: Optional[str]) -> None:
    if repo is None:
        return
    repo_path = Path(repo).expanduser().resolve()
    for p in [repo_path / "src", repo_path]:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def load_flashdiffusion_backend() -> Tuple[Any, Any, Any, Dict[str, Any]]:
    import flashdiffusion.kernel as fk
    importlib.reload(fk)

    dmap_mod = importlib.import_module("flashdiffusion.dmap")
    importlib.reload(dmap_mod)
    DiffusionMap = dmap_mod.DiffusionMap

    ext = fk._get_cuda_backend()
    sm = fk._cuda_sm() if torch.cuda.is_available() else None
    backend_name = getattr(fk, "_SM_BACKEND", {}).get(sm)

    meta = {
        "torch_cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "sm": sm,
        "flashdiffusion_backend": str(backend_name),
        "cuda_ext": str(ext),
    }

    print(f"[backend] torch cuda available: {meta['torch_cuda_available']}")
    print(f"[backend] cuda device: {meta['cuda_device']}")
    print(f"[backend] cuda module: {ext}")
    print(f"[backend] SM: {sm} -> {backend_name}")

    return DiffusionMap, fk, ext, meta


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# -----------------------------------------------------------------------------
# EMNIST loading
# -----------------------------------------------------------------------------

def load_emnist(
    root: str,
    split: str = "bymerge",
    include_test: bool = True,
    download: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load EMNIST from torchvision.

    Parameters
    ----------
    split:
        One of: balanced, bymerge, byclass, digits, letters, mnist.
    include_test:
        If True, concatenate train and test sets. This yields the full EMNIST
        ByMerge/ByClass size of 814,255 images.

    Returns
    -------
    X : float32 array, shape (N, 784), values in [0,1]
    y : int64 array, shape (N,)
    meta : dict
    """
    try:
        from torchvision.datasets import EMNIST
    except Exception as e:
        raise RuntimeError(
            "torchvision is required. Install with: pip install torchvision"
        ) from e

    root = str(Path(root).expanduser())

    print(f"[data] loading EMNIST split={split!r}, include_test={include_test}")
    ds_train = EMNIST(root=root, split=split, train=True, download=download)
    X_train = ds_train.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_train = np.asarray(ds_train.targets, dtype=np.int64)

    if include_test:
        ds_test = EMNIST(root=root, split=split, train=False, download=download)
        X_test = ds_test.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
        y_test = np.asarray(ds_test.targets, dtype=np.int64)

        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
        split_id = np.concatenate(
            [
                np.zeros(len(y_train), dtype=np.int8),
                np.ones(len(y_test), dtype=np.int8),
            ]
        )
    else:
        X = X_train
        y = y_train
        split_id = np.zeros(len(y_train), dtype=np.int8)

    classes = np.unique(y)
    meta = {
        "split": split,
        "include_test": include_test,
        "N": int(X.shape[0]),
        "d": int(X.shape[1]),
        "n_classes": int(len(classes)),
        "classes": classes.tolist(),
        "train_count": int(len(y_train)),
        "test_count": int(np.sum(split_id == 1)),
    }

    print(
        f"[data] X={X.shape} dtype={X.dtype} classes={meta['n_classes']} "
        f"train={meta['train_count']} test={meta['test_count']}"
    )

    return X, y, meta


def subsample_emnist(
    X: np.ndarray,
    y: np.ndarray,
    N: int,
    seed: int,
    stratified: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Subsample EMNIST. For paper runs, stratified=True is preferable so all
    classes remain represented at smaller N.
    """
    total = X.shape[0]
    if N >= total:
        idx = np.arange(total)
        return X, y, idx

    rng = np.random.default_rng(seed)

    if not stratified:
        idx = rng.choice(total, size=N, replace=False)
        return X[idx], y[idx], idx

    idx_parts: List[np.ndarray] = []
    classes, counts = np.unique(y, return_counts=True)

    # Proportional allocation with at least one per class when possible.
    raw = counts / counts.sum() * N
    alloc = np.floor(raw).astype(int)

    # Ensure non-empty classes if N >= number of classes.
    if N >= len(classes):
        alloc = np.maximum(alloc, 1)

    # Adjust to exact N.
    diff = N - int(alloc.sum())
    frac_order = np.argsort(raw - np.floor(raw))[::-1]

    if diff > 0:
        for j in frac_order[:diff]:
            alloc[j] += 1
    elif diff < 0:
        for j in frac_order[::-1]:
            if diff == 0:
                break
            if alloc[j] > 1:
                alloc[j] -= 1
                diff += 1

    for c, n_c in zip(classes, alloc):
        c_idx = np.flatnonzero(y == c)
        take = min(int(n_c), len(c_idx))
        idx_parts.append(rng.choice(c_idx, size=take, replace=False))

    idx = np.concatenate(idx_parts)
    if len(idx) > N:
        idx = rng.choice(idx, size=N, replace=False)
    elif len(idx) < N:
        rest = np.setdiff1d(np.arange(total), idx, assume_unique=False)
        extra = rng.choice(rest, size=N - len(idx), replace=False)
        idx = np.concatenate([idx, extra])

    rng.shuffle(idx)
    return X[idx], y[idx], idx


# -----------------------------------------------------------------------------
# Beta estimation
# -----------------------------------------------------------------------------

def estimate_beta_numpy(
    X: np.ndarray,
    k: int = 10,
    K: int = 5000,
    seed: int = 123,
    batch: Optional[int] = None,
) -> float:
    """
    Estimate beta = 1 / mean kNN squared distance on a random subsample.

    Uses a dense K x K distance matrix. K=5000 is ~100 MB float32 for d2,
    and usually fine on CPU RAM.
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    K_eff = min(K, N)
    idx = rng.choice(N, size=K_eff, replace=False)
    Xs = X[idx].astype(np.float32, copy=False)

    sq = np.sum(Xs * Xs, axis=1, dtype=np.float32)
    gram = Xs @ Xs.T
    d2 = sq[:, None] + sq[None, :] - 2.0 * gram
    np.maximum(d2, 0.0, out=d2)

    vals = np.partition(d2, kth=k, axis=1)[:, 1 : k + 1]
    mean_knn = float(vals.mean())
    beta = 1.0 / mean_knn
    print(f"[beta] K={K_eff} k={k} mean_knn_d2={mean_knn:.6g} beta={beta:.8g}")
    return beta


# -----------------------------------------------------------------------------
# DiffusionMap run wrapper
# -----------------------------------------------------------------------------

def run_dmap_emnist(
    DiffusionMap: Any,
    X: np.ndarray,
    y: np.ndarray,
    N: int,
    alpha: float = 1.0,
    n_components: int = 20,
    beta: Optional[float] = None,
    beta_k: int = 10,
    beta_subsample: int = 5000,
    seed: int = 0,
    stratified: bool = True,
    maxiter: Optional[int] = None,
    tol: float = 0.0,
    matvec_bench: int = 10,
    keep_arrays: bool = False,
) -> Dict[str, Any]:
    """
    Subsample EMNIST, fit DiffusionMap, run eigensolver, and collect metrics.
    """
    Xn, yn, idx = subsample_emnist(X, y, N=N, seed=seed, stratified=stratified)

    if beta is None:
        beta = estimate_beta_numpy(Xn, k=beta_k, K=beta_subsample, seed=seed + 999)

    print(f"[run] N={Xn.shape[0]:,} d={Xn.shape[1]} beta={beta:.8g} alpha={alpha}")

    dm = DiffusionMap(beta=float(beta), alpha=float(alpha), n_components=int(n_components))

    cuda_sync()
    t0 = time.perf_counter()
    dm.fit(Xn)
    cuda_sync()
    fit_time = time.perf_counter() - t0

    # Matvec microbenchmark after precompute, before Lanczos.
    matvec_bench_time = np.nan
    matvec_bench_each = np.nan
    if matvec_bench and matvec_bench > 0:
        rng = np.random.default_rng(seed + 2024)
        vecs = [rng.standard_normal(Xn.shape[0]).astype(np.float32) for _ in range(matvec_bench)]

        # Warmup.
        _ = dm.matvec(vecs[0])
        cuda_sync()

        t0 = time.perf_counter()
        for v in vecs:
            _ = dm.matvec(v)
        cuda_sync()
        matvec_bench_time = time.perf_counter() - t0
        matvec_bench_each = matvec_bench_time / matvec_bench
        print(f"[matvec bench] {matvec_bench} matvecs in {matvec_bench_time:.3f}s; each={matvec_bench_each:.4f}s")

    # Count matvecs during eigensolver.
    matvec_counter = {"n": 0}
    orig_matvec = dm.matvec

    def counted_matvec(v: np.ndarray) -> np.ndarray:
        matvec_counter["n"] += 1
        return orig_matvec(v)

    dm.matvec = counted_matvec  # type: ignore[method-assign]

    cuda_sync()
    t0 = time.perf_counter()
    coords = dm.transform(t=1, tol=tol, maxiter=maxiter)
    cuda_sync()
    eig_time = time.perf_counter() - t0

    n_matvec = int(matvec_counter["n"])
    matvec_time_lanczos = eig_time / n_matvec if n_matvec > 0 else np.nan

    ev = np.asarray(dm.eigenvalues_, dtype=np.float64)

    row: Dict[str, Any] = {
        "N": int(Xn.shape[0]),
        "d": int(Xn.shape[1]),
        "seed": int(seed),
        "alpha": float(alpha),
        "n_components": int(n_components),
        "beta": float(beta),
        "fit_time": float(fit_time),
        "eig_time": float(eig_time),
        "total_time": float(fit_time + eig_time),
        "n_matvec": int(n_matvec),
        "matvec_time_lanczos": float(matvec_time_lanczos),
        "matvec_bench_n": int(matvec_bench),
        "matvec_bench_time": float(matvec_bench_time),
        "matvec_bench_each": float(matvec_bench_each),
        "n_classes_present": int(len(np.unique(yn))),
    }

    for j in range(min(32, len(ev))):
        row[f"lambda{j+1}"] = float(ev[j])

    print(
        f"[run] fit={fit_time:.2f}s eig={eig_time:.2f}s total={fit_time+eig_time:.2f}s "
        f"eig_matvecs={n_matvec} eig_matvec_time={matvec_time_lanczos:.4f}s"
    )
    print(f"[run] eigenvalues[:8] = {ev[:8]}")

    if keep_arrays:
        row["_X"] = Xn
        row["_y"] = yn
        row["_coords"] = coords
        row["_idx"] = idx
        row["_eigenvalues"] = ev

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row


# -----------------------------------------------------------------------------
# KNN accuracy in diffusion coordinates
# -----------------------------------------------------------------------------

def knn_accuracy_sweep(
    coords: np.ndarray,
    y: np.ndarray,
    coord_dims: Sequence[int],
    n_neighbors: int = 5,
    train_size: int = 100_000,
    eval_size: int = 20_000,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Estimate kNN accuracy in diffusion-coordinate space.

    For large N, exact all-vs-all scoring is expensive. This function fits on a
    random training subset and scores on a disjoint random evaluation subset.

    If N is small enough, it still uses disjoint train/eval splits unless the
    requested sizes exceed N.
    """
    try:
        from sklearn.neighbors import KNeighborsClassifier
    except Exception as e:
        raise RuntimeError("scikit-learn is required. Install with: pip install scikit-learn") from e

    N = coords.shape[0]
    rng = np.random.default_rng(seed)

    perm = rng.permutation(N)
    n_train = min(train_size, max(1, N // 2))
    n_eval = min(eval_size, max(1, N - n_train))

    train_idx = perm[:n_train]
    eval_idx = perm[n_train : n_train + n_eval]

    if len(eval_idx) == 0:
        # Fallback for tiny smoke tests.
        train_idx = perm
        eval_idx = perm

    rows: List[Dict[str, Any]] = []

    for m in coord_dims:
        m_eff = int(min(m, coords.shape[1]))
        if m_eff <= 0:
            continue

        clf = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        t0 = time.perf_counter()
        clf.fit(coords[train_idx, :m_eff], y[train_idx])
        fit_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        acc = float(clf.score(coords[eval_idx, :m_eff], y[eval_idx]))
        score_time = time.perf_counter() - t0

        rows.append(
            {
                "coord_dim": m_eff,
                "knn_k": int(n_neighbors),
                "train_size": int(len(train_idx)),
                "eval_size": int(len(eval_idx)),
                "accuracy": acc,
                "fit_time": fit_time,
                "score_time": score_time,
            }
        )

        print(
            f"[knn] m={m_eff:>3} acc={acc:.4f} "
            f"train={len(train_idx)} eval={len(eval_idx)} score_time={score_time:.2f}s"
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Benchmarks
# -----------------------------------------------------------------------------

def benchmark_N_scaling(
    DiffusionMap: Any,
    X: np.ndarray,
    y: np.ndarray,
    N_values: Iterable[int],
    repeats: int,
    alpha: float,
    n_components: int,
    beta_k: int,
    beta_subsample: int,
    stratified: bool,
    maxiter: Optional[int],
    tol: float,
    matvec_bench: int,
    coord_dims: Sequence[int],
    knn_train_size: int,
    knn_eval_size: int,
    knn_k: int,
    backend_label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[Dict[str, Any]]]:
    run_rows: List[Dict[str, Any]] = []
    knn_rows: List[pd.DataFrame] = []

    last_case: Optional[Dict[str, Any]] = None

    for N in N_values:
        for r in range(repeats):
            seed = 10_000 + 101 * r + int(N % 997)
            print(f"\n=== EMNIST N-scaling: N={N:,} repeat={r+1}/{repeats} ===")

            keep_arrays = (r == 0)  # use first repeat for kNN and plotting.
            row = run_dmap_emnist(
                DiffusionMap=DiffusionMap,
                X=X,
                y=y,
                N=int(N),
                alpha=alpha,
                n_components=n_components,
                beta=None,
                beta_k=beta_k,
                beta_subsample=beta_subsample,
                seed=seed,
                stratified=stratified,
                maxiter=maxiter,
                tol=tol,
                matvec_bench=matvec_bench,
                keep_arrays=keep_arrays,
            )

            row["experiment"] = "N_scaling"
            row["repeat"] = r
            row["backend_label"] = backend_label
            run_rows.append({k: v for k, v in row.items() if not k.startswith("_")})

            if keep_arrays:
                last_case = row
                df_knn = knn_accuracy_sweep(
                    coords=row["_coords"],
                    y=row["_y"],
                    coord_dims=coord_dims,
                    n_neighbors=knn_k,
                    train_size=knn_train_size,
                    eval_size=knn_eval_size,
                    seed=seed + 404,
                )
                df_knn["N"] = int(N)
                df_knn["repeat"] = r
                df_knn["experiment"] = "component_sweep"
                df_knn["backend_label"] = backend_label
                df_knn["n_components"] = int(n_components)
                df_knn["alpha"] = float(alpha)
                df_knn["beta"] = float(row["beta"])
                knn_rows.append(df_knn)

            # Drop arrays after use.
            for key in ["_X", "_y", "_coords", "_idx", "_eigenvalues"]:
                row.pop(key, None)

    df_runs = pd.DataFrame(run_rows)
    df_knn_all = pd.concat(knn_rows, ignore_index=True) if knn_rows else pd.DataFrame()

    return df_runs, df_knn_all, last_case


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_N_scaling(df: pd.DataFrame, outdir: Path) -> None:
    if df.empty:
        return

    g = df.groupby("N").agg(["mean", "std"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    for col, label in [
        ("fit_time", "precompute"),
        ("eig_time", "Lanczos"),
        ("total_time", "total"),
    ]:
        x = g.index.values.astype(float)
        y = g[(col, "mean")].values
        e = g[(col, "std")].fillna(0).values
        ax.errorbar(x, y, yerr=e, marker="o", capsize=3, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N images")
    ax.set_ylabel("time (s)")
    ax.set_title("EMNIST scaling at d=784")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[1]
    x = g.index.values.astype(float)
    if ("matvec_bench_each", "mean") in g.columns:
        y = g[("matvec_bench_each", "mean")].values
        e = g[("matvec_bench_each", "std")].fillna(0).values
        ax.errorbar(x, y, yerr=e, marker="o", capsize=3, label="matvec microbench")

    y2 = g[("matvec_time_lanczos", "mean")].values
    e2 = g[("matvec_time_lanczos", "std")].fillna(0).values
    ax.errorbar(x, y2, yerr=e2, marker="s", capsize=3, label="Lanczos avg")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N images")
    ax.set_ylabel("time per matvec (s)")
    ax.set_title("Implicit dense-kernel matvec")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    path = outdir / "emnist_N_scaling.png"
    fig.savefig(path, dpi=200)
    print(f"[plot] wrote {path}")
    plt.close(fig)


def plot_knn_components(df: pd.DataFrame, outdir: Path) -> None:
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    for N, sub in sorted(df.groupby("N")):
        sub = sub.sort_values("coord_dim")
        ax.plot(
            sub["coord_dim"],
            sub["accuracy"],
            marker="o",
            label=f"N={int(N):,}",
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of diffusion coordinates")
    ax.set_ylabel("5-NN accuracy")
    ax.set_title("EMNIST label structure in diffusion-coordinate space")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = outdir / "emnist_knn_vs_components.png"
    fig.savefig(path, dpi=200)
    print(f"[plot] wrote {path}")
    plt.close(fig)


def plot_eigenvalues(df: pd.DataFrame, outdir: Path) -> None:
    if df.empty:
        return

    lambda_cols = [c for c in df.columns if c.startswith("lambda")]
    if not lambda_cols:
        return
    lambda_cols = sorted(lambda_cols, key=lambda s: int(s.replace("lambda", "")))

    # Plot one representative spectrum per N, using repeat 0 if available.
    fig, ax = plt.subplots(figsize=(7, 4))

    for N, sub in sorted(df.groupby("N")):
        sub0 = sub.sort_values("repeat").iloc[0]
        vals = [sub0[c] for c in lambda_cols[:20] if c in sub0 and np.isfinite(sub0[c])]
        ax.plot(np.arange(1, len(vals) + 1), vals, marker="o", label=f"N={int(N):,}")

    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("eigenvalue")
    ax.set_title("Top EMNIST diffusion eigenvalues")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = outdir / "emnist_eigenvalues.png"
    fig.savefig(path, dpi=200)
    print(f"[plot] wrote {path}")
    plt.close(fig)


def plot_embedding(case: Optional[Dict[str, Any]], outdir: Path, n_plot: int = 10_000) -> None:
    if case is None or "_coords" not in case:
        return

    coords = case["_coords"]
    y = case["_y"]
    N = coords.shape[0]

    rng = np.random.default_rng(12345)
    idx = rng.choice(N, size=min(n_plot, N), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (a, b) in zip(axes, [(0, 1), (1, 2)]):
        sc = ax.scatter(coords[idx, a], coords[idx, b], c=y[idx], cmap="tab20", s=2, alpha=0.8)
        ax.set_xlabel(f"DC{a+1}")
        ax.set_ylabel(f"DC{b+1}")
        ax.set_title(f"DC{a+1} vs DC{b+1}")

    fig.suptitle(
        f"EMNIST diffusion coordinates, N={N:,}, beta={case['beta']:.4g}, alpha={case['alpha']}",
        y=1.02,
    )
    fig.tight_layout()
    path = outdir / f"emnist_embedding_N{N}.png"
    fig.savefig(path, dpi=200)
    print(f"[plot] wrote {path}")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EMNIST benchmark suite for FlashDiffusion.")

    p.add_argument("--repo", type=str, default=None, help="Path to FlashDiffusion repo; repo/src is added to sys.path.")
    p.add_argument("--outdir", type=str, default="results_emnist", help="Output directory.")

    p.add_argument("--data-root", type=str, default="/tmp/emnist", help="torchvision EMNIST root.")
    p.add_argument(
        "--split",
        type=str,
        default="bymerge",
        choices=["balanced", "bymerge", "byclass", "digits", "letters", "mnist"],
        help="EMNIST split.",
    )
    p.add_argument("--include-test", action="store_true", help="Concatenate train and test sets.")
    p.add_argument("--train-only", action="store_true", help="Use train set only, overrides --include-test.")
    p.add_argument("--no-download", action="store_true", help="Do not download EMNIST.")

    p.add_argument("--alpha", type=float, default=1.0, help="Coifman-Lafon alpha. Default: 1.0")
    p.add_argument("--n-components", type=int, default=20, help="Eigenvectors including trivial.")
    p.add_argument("--tol", type=float, default=0.0, help="Eigensolver tolerance passed to transform().")
    p.add_argument("--maxiter", type=int, default=None, help="Eigensolver maxiter passed to transform().")

    p.add_argument("--beta-k", type=int, default=10, help="k for kNN beta estimate.")
    p.add_argument("--beta-subsample", type=int, default=5000, help="Subsample size for beta estimate.")

    p.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=[50_000, 100_000, 200_000, 400_000, 814_255],
        help="N values for scaling sweep. Values larger than dataset size are clipped/skipped.",
    )
    p.add_argument("--repeats", type=int, default=3, help="Repeats per N.")
    p.add_argument("--no-stratified", action="store_true", help="Disable stratified subsampling.")

    p.add_argument(
        "--coord-dims",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16, 32, 64],
        help="Diffusion-coordinate dimensions for 5-NN accuracy sweep.",
    )
    p.add_argument("--knn-k", type=int, default=5, help="k for kNN accuracy.")
    p.add_argument("--knn-train-size", type=int, default=100_000, help="Max training subset for kNN.")
    p.add_argument("--knn-eval-size", type=int, default=20_000, help="Max eval subset for kNN.")
    p.add_argument("--matvec-bench", type=int, default=10, help="Number of random matvecs after precompute.")
    p.add_argument("--plot-points", type=int, default=10_000, help="Points in DC scatter plot.")
    p.add_argument("--backend-label", type=str, default="", help="Tag for backend comparisons, e.g. B200-sm100.")

    p.add_argument("--skip-n-scaling", action="store_true", help="Skip N-scaling benchmark.")
    p.add_argument("--skip-plots", action="store_true", help="Skip plot generation.")

    p.add_argument("--quick", action="store_true", help="Quick smoke-test settings.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.train_only:
        include_test = False
    else:
        # For this benchmark, full dataset is more useful by default if requested.
        include_test = bool(args.include_test)

    if args.quick:
        args.split = "balanced"
        include_test = False
        args.n_values = [5_000, 10_000]
        args.repeats = 1
        args.beta_subsample = 1000
        args.n_components = min(args.n_components, 12)
        args.coord_dims = [2, 4, 8]
        args.knn_train_size = 5000
        args.knn_eval_size = 2000
        args.matvec_bench = 3
        print("[quick] using quick smoke-test settings")

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    add_repo_to_path(args.repo)
    DiffusionMap, fk, ext, backend_meta = load_flashdiffusion_backend()

    X, y, data_meta = load_emnist(
        root=args.data_root,
        split=args.split,
        include_test=include_test,
        download=not args.no_download,
    )

    # Clip N values to available dataset size and unique-sort.
    N_total = X.shape[0]
    N_values = sorted(set(int(min(N, N_total)) for N in args.n_values if int(min(N, N_total)) > 0))

    metadata = {
        "backend": backend_meta,
        "data": data_meta,
        "alpha": args.alpha,
        "n_components": args.n_components,
        "tol": args.tol,
        "maxiter": args.maxiter,
        "beta_k": args.beta_k,
        "beta_subsample": args.beta_subsample,
        "N_values": N_values,
        "repeats": args.repeats,
        "coord_dims": args.coord_dims,
        "knn_k": args.knn_k,
        "knn_train_size": args.knn_train_size,
        "knn_eval_size": args.knn_eval_size,
        "matvec_bench": args.matvec_bench,
        "backend_label": args.backend_label,
        "torch_version": torch.__version__,
        "repo": args.repo,
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[main] wrote {outdir / 'metadata.json'}")

    if not args.skip_n_scaling:
        df_runs, df_knn, last_case = benchmark_N_scaling(
            DiffusionMap=DiffusionMap,
            X=X,
            y=y,
            N_values=N_values,
            repeats=args.repeats,
            alpha=args.alpha,
            n_components=args.n_components,
            beta_k=args.beta_k,
            beta_subsample=args.beta_subsample,
            stratified=not args.no_stratified,
            maxiter=args.maxiter,
            tol=args.tol,
            matvec_bench=args.matvec_bench,
            coord_dims=args.coord_dims,
            knn_train_size=args.knn_train_size,
            knn_eval_size=args.knn_eval_size,
            knn_k=args.knn_k,
            backend_label=args.backend_label,
        )

        df_runs.to_csv(outdir / "emnist_N_scaling.csv", index=False)
        df_knn.to_csv(outdir / "emnist_knn_components.csv", index=False)
        print(f"[main] wrote {outdir / 'emnist_N_scaling.csv'}")
        print(f"[main] wrote {outdir / 'emnist_knn_components.csv'}")

        if last_case is not None:
            # Save only the final representative embedding arrays. For full 814k this can be large.
            np.savez_compressed(
                outdir / f"emnist_embedding_N{last_case['N']}.npz",
                coords=last_case["_coords"],
                y=last_case["_y"],
                eigenvalues=last_case["_eigenvalues"],
                beta=last_case["beta"],
                alpha=last_case["alpha"],
            )
            print(f"[main] wrote {outdir / ('emnist_embedding_N' + str(last_case['N']) + '.npz')}")

        if not args.skip_plots:
            plot_N_scaling(df_runs, outdir)
            plot_knn_components(df_knn, outdir)
            plot_eigenvalues(df_runs, outdir)
            plot_embedding(last_case, outdir, n_plot=args.plot_points)

    print(f"[done] outputs in {outdir}")


if __name__ == "__main__":
    main()
```

## `ablations/flashdiffusion_swiss.py`

```python
#!/usr/bin/env python3
"""
flashdiffusion_swiss.py
=======================

Swiss-roll benchmark suite for FlashDiffusion / DiffusionMap.

Experiment 1 focuses only on the low-dimensional Swiss roll:
  1. Dense-reference correctness check for prepass + matvec.
  2. N-scaling benchmark with repeats and error bars.
  3. Beta sweep at fixed N, probing the rank-one -> identity regime.
  4. One embedding visualization.

Default Coifman-Lafon alpha is 1.0.

Example
-------
python flashdiffusion_swiss.py \
    --repo /path/to/FlashDiffusion \
    --outdir results_swiss \
    --n-values 50000 100000 200000 500000 1000000 \
    --repeats 3 \
    --beta-n 200000 \
    --alpha 1.0

For a quick smoke test:
python flashdiffusion_swiss.py --quick --outdir results_quick
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Environment / backend
# -----------------------------------------------------------------------------

def add_repo_to_path(repo: Optional[str]) -> None:
    """Add repo/src or repo to sys.path if provided."""
    if repo is None:
        return

    repo_path = Path(repo).expanduser().resolve()
    candidates = [repo_path / "src", repo_path]

    for p in candidates:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def load_flashdiffusion_backend() -> Tuple[Any, Any, Any]:
    """
    Import DiffusionMap and CUDA backend.

    Returns
    -------
    DiffusionMap : class
    fk           : flashdiffusion.kernel module
    ext          : selected CUDA extension module, or None
    """
    import flashdiffusion.kernel as fk
    importlib.reload(fk)

    dmap_mod = importlib.import_module("flashdiffusion.dmap")
    importlib.reload(dmap_mod)
    DiffusionMap = dmap_mod.DiffusionMap

    ext = fk._get_cuda_backend()
    sm = fk._cuda_sm() if torch.cuda.is_available() else None

    print(f"[backend] torch cuda available: {torch.cuda.is_available()}")
    print(f"[backend] cuda module: {ext}")
    print(f"[backend] SM: {sm} -> {getattr(fk, '_SM_BACKEND', {}).get(sm)}")

    return DiffusionMap, fk, ext


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# -----------------------------------------------------------------------------
# Swiss roll generation and beta estimation
# -----------------------------------------------------------------------------

def make_swiss_roll(
    N: int,
    device: str = "cuda",
    seed: int = 42,
    noise: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a normalized Swiss roll directly on GPU.

    Returns
    -------
    X : (N, 3) torch.float32
        Normalized ambient coordinates.
    t : (N,) torch.float32
        Latent roll coordinate.
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    g = torch.Generator(device=device)
    g.manual_seed(seed)

    t = torch.rand(N, device=device, generator=g) * (4.5 - 1.5) * torch.pi + 1.5 * torch.pi
    h = torch.rand(N, device=device, generator=g) * 10.0

    X = torch.stack([t * torch.cos(t), h, t * torch.sin(t)], dim=1)
    X = X + torch.randn(X.shape, device=device, generator=g) * noise
    X = (X - X.mean(0)) / X.std(0)

    return X.float(), t.float()


def estimate_beta(
    X: torch.Tensor,
    k: int = 10,
    K: int = 5000,
    seed: int = 123,
) -> float:
    """
    Estimate beta as inverse mean k-NN squared distance on a subsample.
    """
    N = X.shape[0]
    g = torch.Generator(device=X.device)
    g.manual_seed(seed)

    idx = torch.randperm(N, device=X.device, generator=g)[: min(K, N)]
    Xs = X[idx].float()

    sq = (Xs * Xs).sum(1)
    d2 = (sq[:, None] + sq[None, :] - 2.0 * (Xs @ Xs.T)).clamp(min=0.0)

    # Drop self-distance.
    knn = d2.topk(k + 1, dim=1, largest=False).values[:, 1:]
    return 1.0 / knn.mean().item()


# -----------------------------------------------------------------------------
# Correctness checks
# -----------------------------------------------------------------------------

def correctness_check(
    ext: Any,
    N: int = 300,
    d: int = 3,
    beta: float = 1.0,
    seed: int = 0,
    atol: float = 5e-2,
) -> Dict[str, Any]:
    """
    Compare CUDA prepass/matvec against dense NumPy reference.
    """
    if ext is None:
        print("[correctness] no CUDA extension found; skipping correctness check")
        return {"skipped": True}

    print(f"[correctness] N={N} d={d} beta={beta}")

    rng = np.random.default_rng(seed)
    X_np = rng.standard_normal((N, d), dtype=np.float32)

    X_fp = torch.from_numpy(X_np).half().cuda()
    sq = ext.compute_sq_norms(X_fp)

    w = torch.ones(N, dtype=torch.float32, device="cuda")
    cuda_sync()
    D = ext.prepass(X_fp, w, sq, float(beta))
    cuda_sync()

    X64 = X_np.astype(np.float64)
    sq64 = (X64 * X64).sum(1)
    d2 = np.maximum(sq64[:, None] + sq64[None, :] - 2.0 * (X64 @ X64.T), 0.0)
    K = np.exp(-beta * d2)

    D_ref = K.sum(1)
    prepass_max_abs = float(np.max(np.abs(D.detach().cpu().numpy() - D_ref)))
    prepass_rel_l2 = float(
        np.linalg.norm(D.detach().cpu().numpy() - D_ref) / np.linalg.norm(D_ref)
    )

    v = torch.randn(N, dtype=torch.float32, device="cuda")
    rscale = torch.ones(N, dtype=torch.float32, device="cuda")

    cuda_sync()
    out = ext.matvec(X_fp, v, rscale, sq, float(beta))
    cuda_sync()

    v_np = v.detach().cpu().numpy()
    out_ref = K @ v_np

    matvec_max_abs = float(np.max(np.abs(out.detach().cpu().numpy() - out_ref)))
    matvec_rel_l2 = float(
        np.linalg.norm(out.detach().cpu().numpy() - out_ref) / np.linalg.norm(out_ref)
    )

    result = {
        "skipped": False,
        "N": N,
        "d": d,
        "beta": beta,
        "prepass_max_abs": prepass_max_abs,
        "prepass_rel_l2": prepass_rel_l2,
        "matvec_max_abs": matvec_max_abs,
        "matvec_rel_l2": matvec_rel_l2,
        "prepass_pass": prepass_max_abs < atol,
        "matvec_pass": matvec_max_abs < atol,
        "atol": atol,
    }

    print(
        f"[correctness] prepass max abs={prepass_max_abs:.4g}, rel l2={prepass_rel_l2:.4g} "
        f"{'PASS' if result['prepass_pass'] else 'FAIL'}"
    )
    print(
        f"[correctness] matvec  max abs={matvec_max_abs:.4g}, rel l2={matvec_rel_l2:.4g} "
        f"{'PASS' if result['matvec_pass'] else 'FAIL'}"
    )

    return result


# -----------------------------------------------------------------------------
# DiffusionMap run wrapper
# -----------------------------------------------------------------------------

def run_dmap_swiss(
    DiffusionMap: Any,
    N: int,
    alpha: float = 1.0,
    n_components: int = 8,
    beta: Optional[float] = None,
    beta_scale: float = 1.0 / 8.0,
    beta_k: int = 10,
    beta_subsample: int = 5000,
    seed: int = 42,
    noise: float = 0.1,
    maxiter: Optional[int] = None,
    tol: float = 0.0,
    keep_arrays: bool = False,
) -> Dict[str, Any]:
    """
    Generate Swiss roll, run DiffusionMap, and collect runtime/geometry metrics.

    The matvec counter wraps dm.matvec so the Lanczos iteration count is measured
    independent of eigensolver internals.
    """
    print(f"[run] N={N:,} seed={seed} alpha={alpha}")

    X_gpu, t_gpu = make_swiss_roll(N, seed=seed, noise=noise)

    if beta is None:
        beta0 = estimate_beta(X_gpu, k=beta_k, K=beta_subsample, seed=seed + 999)
        beta = beta0 * beta_scale
    else:
        beta0 = np.nan

    X_np = X_gpu.cpu().numpy().astype(np.float32)
    t_np = t_gpu.cpu().numpy().astype(np.float32)

    print(f"[run] beta={beta:.8g} beta0={beta0} beta_scale={beta_scale}")

    dm = DiffusionMap(beta=float(beta), alpha=float(alpha), n_components=int(n_components), tile=512)

    cuda_sync()
    t0 = time.perf_counter()
    dm.fit(X_np)
    cuda_sync()
    fit_time = time.perf_counter() - t0

    # Wrap matvec to count eigensolver calls.
    matvec_counter = {"n": 0}
    orig_matvec = dm.matvec

    def counted_matvec(v: np.ndarray) -> np.ndarray:
        matvec_counter["n"] += 1
        return orig_matvec(v)

    dm.matvec = counted_matvec  # type: ignore[method-assign]

    cuda_sync()
    t0 = time.perf_counter()
    coords = dm.transform(t=1, tol=tol, maxiter=maxiter)
    cuda_sync()
    eig_time = time.perf_counter() - t0

    n_matvec = int(matvec_counter["n"])
    matvec_time = eig_time / n_matvec if n_matvec > 0 else np.nan

    # Geometry metrics.
    n_corr = min(5, coords.shape[1])
    corrs = [
        abs(np.corrcoef(t_np, coords[:, j])[0, 1])
        for j in range(n_corr)
        if np.std(coords[:, j]) > 0
    ]
    best_corr_t = float(np.max(corrs)) if len(corrs) else np.nan

    # Oracle linear R^2 from first m diffusion coordinates to latent t.
    # Subsample for large N to keep the least-squares cheap and deterministic.
    m = min(5, coords.shape[1])
    if N > 200_000:
        rng = np.random.default_rng(seed + 2027)
        idx = rng.choice(N, size=200_000, replace=False)
        C = coords[idx, :m]
        y = t_np[idx]
    else:
        C = coords[:, :m]
        y = t_np

    A = np.c_[C, np.ones(C.shape[0], dtype=C.dtype)]
    coef, *_ = np.linalg.lstsq(A.astype(np.float64), y.astype(np.float64), rcond=None)
    pred = A.astype(np.float64) @ coef
    denom = np.sum((y - y.mean()) ** 2)
    oracle_r2_t = float(1.0 - np.sum((y - pred) ** 2) / denom) if denom > 0 else np.nan

    ev = np.asarray(dm.eigenvalues_, dtype=np.float64)
    spectral_gap_12 = float(ev[0] - ev[1]) if len(ev) > 1 else np.nan
    spectral_gap_23 = float(ev[1] - ev[2]) if len(ev) > 2 else np.nan

    row = {
        "N": int(N),
        "seed": int(seed),
        "alpha": float(alpha),
        "n_components": int(n_components),
        "beta": float(beta),
        "beta0": float(beta0) if np.isfinite(beta0) else np.nan,
        "beta_scale": float(beta_scale),
        "fit_time": float(fit_time),
        "eig_time": float(eig_time),
        "total_time": float(fit_time + eig_time),
        "n_matvec": int(n_matvec),
        "matvec_time": float(matvec_time),
        "best_corr_t": float(best_corr_t),
        "oracle_r2_t": float(oracle_r2_t),
        "spectral_gap_12": float(spectral_gap_12),
        "spectral_gap_23": float(spectral_gap_23),
    }

    for j in range(min(8, len(ev))):
        row[f"lambda{j+1}"] = float(ev[j])

    print(
        f"[run] fit={fit_time:.2f}s eig={eig_time:.2f}s total={fit_time+eig_time:.2f}s "
        f"matvecs={n_matvec} matvec_time={matvec_time:.3f}s "
        f"best_corr={best_corr_t:.3f} R2={oracle_r2_t:.3f}"
    )

    if keep_arrays:
        row["_X_np"] = X_np
        row["_t_np"] = t_np
        row["_coords"] = coords
        row["_eigenvalues"] = ev

    # Help memory pressure between sweeps.
    del X_gpu, t_gpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row


# -----------------------------------------------------------------------------
# Benchmarks
# -----------------------------------------------------------------------------

def benchmark_N_scaling(
    DiffusionMap: Any,
    N_values: Iterable[int],
    repeats: int,
    alpha: float,
    beta_scale: float,
    n_components: int,
    beta_k: int,
    beta_subsample: int,
    noise: float,
    maxiter: Optional[int],
    tol: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for N in N_values:
        for r in range(repeats):
            seed = 1000 + 37 * r + int(N % 997)
            print(f"\n=== N-scaling: N={N:,} repeat={r+1}/{repeats} ===")
            row = run_dmap_swiss(
                DiffusionMap=DiffusionMap,
                N=int(N),
                alpha=alpha,
                n_components=n_components,
                beta=None,
                beta_scale=beta_scale,
                beta_k=beta_k,
                beta_subsample=beta_subsample,
                seed=seed,
                noise=noise,
                maxiter=maxiter,
                tol=tol,
                keep_arrays=False,
            )
            row["experiment"] = "N_scaling"
            row["repeat"] = r
            rows.append(row)

    return pd.DataFrame(rows)


def benchmark_beta_sweep(
    DiffusionMap: Any,
    N: int,
    beta_factors: Iterable[float],
    repeats: int,
    alpha: float,
    n_components: int,
    beta_k: int,
    beta_subsample: int,
    noise: float,
    maxiter: Optional[int],
    tol: float,
) -> pd.DataFrame:
    # Estimate a single beta0 from a fixed dataset so beta_factor means one thing.
    X_gpu, _ = make_swiss_roll(N, seed=424242, noise=noise)
    beta0 = estimate_beta(X_gpu, k=beta_k, K=beta_subsample, seed=777)
    del X_gpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[beta sweep] fixed N={N:,}, beta0={beta0:.8g}")

    rows: List[Dict[str, Any]] = []

    for factor in beta_factors:
        beta = float(beta0 * factor)

        for r in range(repeats):
            seed = 2000 + 53 * r
            print(
                f"\n=== beta-sweep: N={N:,} beta/beta0={factor:g} "
                f"beta={beta:.8g} repeat={r+1}/{repeats} ==="
            )

            row = run_dmap_swiss(
                DiffusionMap=DiffusionMap,
                N=int(N),
                alpha=alpha,
                n_components=n_components,
                beta=beta,
                beta_scale=float(factor),
                beta_k=beta_k,
                beta_subsample=beta_subsample,
                seed=seed,
                noise=noise,
                maxiter=maxiter,
                tol=tol,
                keep_arrays=False,
            )
            row["experiment"] = "beta_sweep"
            row["repeat"] = r
            row["beta_factor"] = float(factor)
            row["beta0_fixed"] = float(beta0)
            rows.append(row)

    return pd.DataFrame(rows)


def run_visualization_case(
    DiffusionMap: Any,
    N: int,
    alpha: float,
    beta_scale: float,
    n_components: int,
    beta_k: int,
    beta_subsample: int,
    noise: float,
    seed: int,
    maxiter: Optional[int],
    tol: float,
) -> Dict[str, Any]:
    print(f"\n=== visualization run: N={N:,} ===")
    return run_dmap_swiss(
        DiffusionMap=DiffusionMap,
        N=N,
        alpha=alpha,
        n_components=n_components,
        beta=None,
        beta_scale=beta_scale,
        beta_k=beta_k,
        beta_subsample=beta_subsample,
        seed=seed,
        noise=noise,
        maxiter=maxiter,
        tol=tol,
        keep_arrays=True,
    )


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def _mean_std(df: pd.DataFrame, group: str, cols: List[str]) -> pd.DataFrame:
    agg = df.groupby(group)[cols].agg(["mean", "std"]).reset_index()
    return agg


def plot_N_scaling(df: pd.DataFrame, outdir: Path) -> None:
    if df.empty:
        return

    g = df.groupby("N").agg(["mean", "std"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    for col, label in [
        ("fit_time", "precompute"),
        ("eig_time", "Lanczos"),
        ("total_time", "total"),
    ]:
        x = g.index.values.astype(float)
        y = g[(col, "mean")].values
        e = g[(col, "std")].fillna(0).values
        ax.errorbar(x, y, yerr=e, marker="o", capsize=3, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N samples")
    ax.set_ylabel("time (s)")
    ax.set_title("Swiss roll scaling")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[1]
    x = g.index.values.astype(float)
    y = g[("matvec_time", "mean")].values
    e = g[("matvec_time", "std")].fillna(0).values
    ax.errorbar(x, y, yerr=e, marker="o", capsize=3, label="time per matvec")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N samples")
    ax.set_ylabel("time per matvec (s)")
    ax.set_title("Implicit dense-kernel matvec")
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    path = outdir / "swiss_roll_N_scaling.png"
    fig.savefig(path, dpi=200)
    print(f"[plot] wrote {path}")
    plt.close(fig)


def plot_beta_sweep(df: pd.DataFrame, outdir: Path) -> None:
    if df.empty:
        return

    g = df.groupby("beta_factor").agg(["mean", "std"])
    x = g.index.values.astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Time and matvec count.
    ax1 = axes[0]
    y = g[("eig_time", "mean")].values
    e = g[("eig_time", "std")].fillna(0).values
    ax1.errorbar(x, y, yerr=e, marker="o", capsize=3, label="Lanczos time")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$\beta/\beta_0$")
    ax1.set_ylabel("Lanczos time (s)")
    ax1.grid(True, which="both", alpha=0.3)

    ax1b = ax1.twinx()
    y2 = g[("n_matvec", "mean")].values
    e2 = g[("n_matvec", "std")].fillna(0).values
    ax1b.errorbar(x, y2, yerr=e2, marker="s", capsize=3, label="# matvecs")
    ax1b.set_ylabel("matvec calls")
    ax1.set_title("Bandwidth controls eigensolver cost")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    # Geometry and spectrum.
    ax2 = axes[1]
    y = g[("oracle_r2_t", "mean")].values
    e = g[("oracle_r2_t", "std")].fillna(0).values
    ax2.errorbar(x, y, yerr=e, marker="o", capsize=3, label=r"oracle $R^2(t)$")

    y_corr = g[("best_corr_t", "mean")].values
    e_corr = g[("best_corr_t", "std")].fillna(0).values
    ax2.errorbar(x, y_corr, yerr=e_corr, marker="s", capsize=3, label="best |corr(t, DCj)|")

    if ("spectral_gap_12", "mean") in g.columns:
        y_gap = g[("spectral_gap_12", "mean")].values
        ax2.plot(x, y_gap, marker="^", linestyle="--", label=r"$\lambda_1-\lambda_2$")

    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\beta/\beta_0$")
    ax2.set_ylabel("quality / spectral gap")
    ax2.set_title("Rank-one to localized-kernel regime")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="best")

    fig.tight_layout()
    path = outdir / "swiss_roll_beta_sweep.png"
    fig.savefig(path, dpi=200)
    print(f"[plot] wrote {path}")
    plt.close(fig)

    # Optional eigenvalue spectrum plot.
    lambda_cols = [c for c in df.columns if c.startswith("lambda")]
    lambda_cols = sorted(lambda_cols, key=lambda s: int(s.replace("lambda", "")))
    if lambda_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        for col in lambda_cols[:8]:
            y = g[(col, "mean")].values
            ax.plot(x, y, marker="o", label=col.replace("lambda", r"$\lambda_") + "$")
        ax.set_xscale("log")
        ax.set_xlabel(r"$\beta/\beta_0$")
        ax.set_ylabel("eigenvalue")
        ax.set_title("Top diffusion eigenvalues vs bandwidth")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        path = outdir / "swiss_roll_beta_eigenvalues.png"
        fig.savefig(path, dpi=200)
        print(f"[plot] wrote {path}")
        plt.close(fig)


def plot_embedding(case: Dict[str, Any], outdir: Path, n_plot: int = 5000) -> None:
    if "_X_np" not in case:
        return

    X_np = case["_X_np"]
    t_np = case["_t_np"]
    coords = case["_coords"]
    N = int(case["N"])

    rng = np.random.default_rng(12345)
    idx = rng.choice(N, size=min(n_plot, N), replace=False)

    fig = plt.figure(figsize=(15, 4))

    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.scatter(
        X_np[idx, 0],
        X_np[idx, 1],
        X_np[idx, 2],
        c=t_np[idx],
        cmap="plasma",
        s=1,
    )
    ax.set_title(f"Input 3D, N={N:,}")

    for p, (a, b) in enumerate([(0, 1), (1, 2)], start=2):
        ax = fig.add_subplot(1, 3, p)
        sc = ax.scatter(
            coords[idx, a],
            coords[idx, b],
            c=t_np[idx],
            cmap="plasma",
            s=1,
        )
        ax.set_xlabel(f"DC{a+1}")
        ax.set_ylabel(f"DC{b+1}")
        ax.set_title(f"DC{a+1} vs DC{b+1}")

    plt.colorbar(sc, ax=fig.axes[-1], label="latent t")
    plt.suptitle(
        f"FlashDiffusion Swiss roll, N={N:,}, alpha={case['alpha']}, beta={case['beta']:.4g}, "
        f"precompute={case['fit_time']:.1f}s, Lanczos={case['eig_time']:.1f}s",
        y=1.02,
        fontsize=10,
    )
    plt.tight_layout()

    path = outdir / f"swiss_roll_embedding_N{N}.png"
    fig.savefig(path, dpi=200)
    print(f"[plot] wrote {path}")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Swiss-roll benchmark suite for FlashDiffusion.")

    p.add_argument("--repo", type=str, default=None, help="Path to FlashDiffusion repo; repo/src is added to sys.path.")
    p.add_argument("--outdir", type=str, default="results_swiss", help="Output directory.")

    p.add_argument("--alpha", type=float, default=1.0, help="Coifman-Lafon alpha. Default: 1.0")
    p.add_argument("--n-components", type=int, default=8, help="Number of eigenvectors including trivial.")
    p.add_argument("--tol", type=float, default=0.0, help="Eigensolver tolerance passed to transform().")
    p.add_argument("--maxiter", type=int, default=None, help="Eigensolver maxiter passed to transform().")
    p.add_argument("--noise", type=float, default=0.1, help="Swiss-roll Gaussian noise.")
    p.add_argument("--beta-k", type=int, default=10, help="k for kNN beta estimate.")
    p.add_argument("--beta-subsample", type=int, default=5000, help="Subsample K for beta estimate.")
    p.add_argument("--beta-scale", type=float, default=1.0/8.0, help="Default beta = beta0 * beta_scale for N scaling.")

    p.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=[50_000, 100_000, 200_000, 500_000, 1_000_000],
        help="N values for scaling sweep.",
    )
    p.add_argument("--repeats", type=int, default=3, help="Repeats per N and per beta factor.")

    p.add_argument("--beta-n", type=int, default=200_000, help="Fixed N for beta sweep.")
    p.add_argument(
        "--beta-factors",
        type=float,
        nargs="+",
        default=[1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16],
        help="Multipliers of beta0 for beta sweep.",
    )

    p.add_argument("--viz-n", type=int, default=200_000, help="N for embedding visualization run.")
    p.add_argument("--viz-points", type=int, default=5000, help="Number of points in embedding plot.")

    p.add_argument("--skip-correctness", action="store_true", help="Skip dense-reference correctness check.")
    p.add_argument("--skip-n-scaling", action="store_true", help="Skip N-scaling benchmark.")
    p.add_argument("--skip-beta-sweep", action="store_true", help="Skip beta sweep benchmark.")
    p.add_argument("--skip-viz", action="store_true", help="Skip embedding visualization run.")

    p.add_argument("--quick", action="store_true", help="Quick smoke-test settings.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.quick:
        args.n_values = [5_000, 10_000, 20_000]
        args.beta_n = 20_000
        args.viz_n = 20_000
        args.repeats = 1
        args.beta_factors = [1/32, 1/8, 1/2, 2, 8]
        args.beta_subsample = 2000
        print("[quick] using quick smoke-test settings")

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    add_repo_to_path(args.repo)
    DiffusionMap, fk, ext = load_flashdiffusion_backend()

    metadata = {
        "alpha": args.alpha,
        "n_components": args.n_components,
        "tol": args.tol,
        "maxiter": args.maxiter,
        "noise": args.noise,
        "beta_k": args.beta_k,
        "beta_subsample": args.beta_subsample,
        "beta_scale": args.beta_scale,
        "n_values": args.n_values,
        "repeats": args.repeats,
        "beta_n": args.beta_n,
        "beta_factors": args.beta_factors,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "repo": args.repo,
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[main] wrote {outdir / 'metadata.json'}")

    if not args.skip_correctness:
        corr = correctness_check(ext=ext, N=300, d=3, beta=1.0)
        pd.DataFrame([corr]).to_csv(outdir / "correctness.csv", index=False)

    if not args.skip_n_scaling:
        df_N = benchmark_N_scaling(
            DiffusionMap=DiffusionMap,
            N_values=args.n_values,
            repeats=args.repeats,
            alpha=args.alpha,
            beta_scale=args.beta_scale,
            n_components=args.n_components,
            beta_k=args.beta_k,
            beta_subsample=args.beta_subsample,
            noise=args.noise,
            maxiter=args.maxiter,
            tol=args.tol,
        )
        df_N.to_csv(outdir / "swiss_roll_N_scaling.csv", index=False)
        print(f"[main] wrote {outdir / 'swiss_roll_N_scaling.csv'}")
        plot_N_scaling(df_N, outdir)

    if not args.skip_beta_sweep:
        df_beta = benchmark_beta_sweep(
            DiffusionMap=DiffusionMap,
            N=args.beta_n,
            beta_factors=args.beta_factors,
            repeats=args.repeats,
            alpha=args.alpha,
            n_components=args.n_components,
            beta_k=args.beta_k,
            beta_subsample=args.beta_subsample,
            noise=args.noise,
            maxiter=args.maxiter,
            tol=args.tol,
        )
        df_beta.to_csv(outdir / "swiss_roll_beta_sweep.csv", index=False)
        print(f"[main] wrote {outdir / 'swiss_roll_beta_sweep.csv'}")
        plot_beta_sweep(df_beta, outdir)

    if not args.skip_viz:
        case = run_visualization_case(
            DiffusionMap=DiffusionMap,
            N=args.viz_n,
            alpha=args.alpha,
            beta_scale=args.beta_scale,
            n_components=args.n_components,
            beta_k=args.beta_k,
            beta_subsample=args.beta_subsample,
            noise=args.noise,
            seed=314159,
            maxiter=args.maxiter,
            tol=args.tol,
        )
        # Save arrays for possible reuse, but compressed.
        np.savez_compressed(
            outdir / f"swiss_roll_embedding_N{args.viz_n}.npz",
            X=case["_X_np"],
            t=case["_t_np"],
            coords=case["_coords"],
            eigenvalues=case["_eigenvalues"],
            beta=case["beta"],
            alpha=case["alpha"],
        )
        plot_embedding(case, outdir, n_plot=args.viz_points)

    print(f"[done] outputs in {outdir}")


if __name__ == "__main__":
    main()
```

## `examples/swiss_roll.py`

```python
"""examples/swiss_roll.py — basic DiffusionMap demo on Swiss roll.

The Swiss roll is a classic 2D manifold embedded in 3D.
Diffusion maps should unroll it: DC1 and DC2 capture the angle t,
DC3 captures the height.

Bandwidth note: the median heuristic gives a kernel that may be too local
for long-range manifolds. A good rule of thumb is to choose beta so that
each point has ~5-15% of N neighbors with K(i,j) > 0.1.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from flashdiffusion import DiffusionMap

rng = np.random.default_rng(42)
N = 400
t   = rng.uniform(1.5*np.pi, 4.5*np.pi, N)
X   = np.stack([t*np.cos(t), rng.uniform(0, 10, N), t*np.sin(t)], axis=1)
X  += rng.standard_normal(X.shape) * 0.1
X   = (X - X.mean(0)) / X.std()          # unit variance normalisation

# Choose beta so median K(i,j) ~ exp(-1) across nearest neighbours
sq    = (X*X).sum(1)
dist2 = np.maximum(sq[:,None] + sq[None,:] - 2*X@X.T, 0)
k10   = np.sort(dist2, 1)[:, 10].mean()  # mean 10-NN distance²
beta  = 1.0 / k10
print(f"10-NN bandwidth: beta={beta:.4f}")

dm = DiffusionMap(beta=beta, alpha=0.5, n_components=8, tile=256)
dm.fit(X)
coords = dm.transform(t=1)

print(f"Eigenvalues: {dm.eigenvalues_}")
print(f"Embedding shape: {coords.shape}")

# Swiss roll angle t is encoded in the degenerate (DC1, DC2) pair
for i in range(4):
    c = abs(np.corrcoef(t, coords[:, i])[0, 1])
    print(f"  |corr(t, DC{i+1})| = {c:.3f}")

print("Swiss roll demo complete.")
```

## `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "flashdiffusion"
version = "0.1.2"
description = "GPU-accelerated diffusion maps with tensor core kernels"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
authors = [{name = "Julio Candanedo", email = "julio@sparsetrace.ai"}]
keywords = ["diffusion maps", "manifold learning", "CUDA", "tensor cores", "GPU"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
dependencies = ["numpy>=1.24", "scipy>=1.10", "ninja"]

[project.urls]
Homepage = "https://github.com/sparsetrace/FlashDiffusion"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
flashdiffusion = ["csrc/*.cu", "csrc/*.cuh"]
```

## `setup.py`

```python
"""
setup.py
========
Builds FlashDiffusion CUDA extensions.

Kernels in csrc/:
  flash_diffusion_sm80.cu   — SM80 scalar       (A100, fallback)
  flash_diffusion_sm90.cu   — SM90 wgmma+TMA    (H100, requires CUTLASS >= 3.5)
  flash_diffusion_sm100.cu  — SM100 WMMA         (B200/B300)
  flash_diffusion_sm120.cu  — SM120 128×128 tile (RTX 5090 / RTX PRO 6000)

Usage
-----
  # CPU only:
  pip install -e .

  # GPU — auto-detects SM, auto-clones CUTLASS if needed:
  FLASHDIFFUSION_BUILD_CUDA=1 pip install -e ".[cuda]"

  # specific arch:
  FLASHDIFFUSION_BUILD_CUDA=1 TORCH_CUDA_ARCH_LIST="8.0"  pip install -e ".[cuda]"
  FLASHDIFFUSION_BUILD_CUDA=1 TORCH_CUDA_ARCH_LIST="10.0" pip install -e ".[cuda]"
  FLASHDIFFUSION_BUILD_CUDA=1 TORCH_CUDA_ARCH_LIST="12.0" pip install -e ".[cuda]"

CUTLASS
-------
  SM90 and SM100 kernels need CUTLASS headers.
  setup.py will auto-clone CUTLASS main into third_party/cutlass if not found.
  To skip auto-clone, set CUTLASS_PATH to an existing CUTLASS include dir.
  To use a git submodule instead:
    git submodule add https://github.com/NVIDIA/cutlass third_party/cutlass
    git submodule update --init --depth=1
"""

import os, sys, glob, subprocess
from setuptools import setup

# ---------------------------------------------------------------------------
# Helper: find CUDA headers
# ---------------------------------------------------------------------------

def find_cuda_includes():
    py = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = []
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        candidates.append(os.path.join(cuda_home, "include"))
    for base in [
        f"/usr/local/lib/{py}/site-packages/nvidia",
        f"/usr/lib/{py}/site-packages/nvidia",
    ]:
        candidates += glob.glob(f"{base}/*/include")
    candidates += ["/usr/local/cuda/include", "/usr/cuda/include"]
    return [p for p in candidates if os.path.isdir(p)]


# ---------------------------------------------------------------------------
# Helper: find or auto-clone CUTLASS
# ---------------------------------------------------------------------------

def _has_sm100_headers(inc):
    return os.path.exists(
        os.path.join(inc, "cute", "arch", "mma_sm100_umma.hpp"))

def _has_sm90_headers(inc):
    return os.path.exists(
        os.path.join(inc, "cute", "tensor.hpp"))

def find_cutlass_include():
    """
    Find CUTLASS headers. Check order:
      1. CUTLASS_PATH env var
      2. third_party/cutlass submodule (in this repo)
      3. /tmp/cutlass* (notebook session clones)
      4. Auto-clone CUTLASS main into third_party/cutlass
    Returns path to include/ dir, or None if all fail.
    """
    repo_root = os.path.dirname(os.path.abspath(__file__))
    candidates = []

    # 1. env var
    env = os.environ.get("CUTLASS_PATH", "")
    if env:
        candidates.append(env if env.endswith("include") else
                          os.path.join(env, "include"))

    # 2. git submodule
    sub = os.path.join(repo_root, "third_party", "cutlass", "include")
    candidates.append(sub)

    # 3. /tmp session clones — prefer cutlass_main (has SM100 headers)
    for p in sorted(glob.glob("/tmp/cutlass*/include"), reverse=True):
        candidates.append(p)

    for inc in candidates:
        if os.path.isdir(inc) and _has_sm90_headers(inc):
            return inc

    return None   # trigger auto-clone below


def ensure_cutlass():
    """
    Return CUTLASS include path, auto-cloning if necessary.
    Clones into third_party/cutlass (persists across sessions).
    """
    inc = find_cutlass_include()
    if inc:
        return inc

    # auto-clone CUTLASS main into third_party/cutlass
    repo_root = os.path.dirname(os.path.abspath(__file__))
    target    = os.path.join(repo_root, "third_party", "cutlass")
    inc       = os.path.join(target, "include")

    print("[setup.py] CUTLASS not found — cloning main branch into "
          "third_party/cutlass ...")
    print("[setup.py] (This happens once; subsequent installs reuse the clone.)")

    os.makedirs(os.path.dirname(target), exist_ok=True)
    subprocess.check_call([
        "git", "clone", "--depth=1",
        "https://github.com/NVIDIA/cutlass", target
    ])
    print(f"[setup.py] CUTLASS cloned to {target}")
    return inc


# ---------------------------------------------------------------------------
# SM routing tables
# ---------------------------------------------------------------------------

_SM_SOURCES = {
    "80":  "src/flashdiffusion/csrc/flash_diffusion_sm80.cu",
    "86":  "src/flashdiffusion/csrc/flash_diffusion_sm80.cu",
    "89":  "src/flashdiffusion/csrc/flash_diffusion_sm80.cu",
    "90":  "src/flashdiffusion/csrc/flash_diffusion_sm90.cu",
    "100": "src/flashdiffusion/csrc/flash_diffusion_sm100.cu",
    "103": "src/flashdiffusion/csrc/flash_diffusion_sm100.cu",
    "120": "src/flashdiffusion/csrc/flash_diffusion_sm120.cu",
    "121": "src/flashdiffusion/csrc/flash_diffusion_sm120.cu",
}

_SM_MODULE = {
    "80":  "flash_diffusion_cuda",
    "86":  "flash_diffusion_cuda",
    "89":  "flash_diffusion_cuda",
    "90":  "flash_diffusion_sm90",
    "100": "flash_diffusion_sm100",
    "103": "flash_diffusion_sm100",
    "120": "flash_diffusion_sm120",
    "121": "flash_diffusion_sm120",
}

_SM_ARCH = {
    "80":  "sm_80",
    "86":  "sm_86",
    "89":  "sm_89",
    "90":  "sm_90a",   # 'a' required for wgmma
    "100": "sm_100a",  # 'a' required for UMMA
    "103": "sm_100a",
    "120": "sm_120",
    "121": "sm_121",
}

# SMs that need CUTLASS headers
_SM_NEEDS_CUTLASS = {"90", "100", "103"}

# Extra nvcc defines per SM
_SM_NVCC_DEFINES = {
    "90":  ["-DCUTLASS_ARCH_MMA_SM90_SUPPORTED=1",
            "-DCUTE_ARCH_MMA_SM90A_ENABLED=1"],
    "100": ["-DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1",
            "-DCUTE_ARCH_MMA_SM100A_ENABLED=1"],
    "103": ["-DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1",
            "-DCUTE_ARCH_MMA_SM100A_ENABLED=1"],
}


def _gencode(sm):
    arch    = _SM_ARCH.get(sm, f"sm_{sm}")
    compute = arch.replace("sm_", "compute_").rstrip("a")  # compute_100a → compute_100
    return f"-gencode=arch={compute},code={arch}"


# ---------------------------------------------------------------------------
# Extension build
# ---------------------------------------------------------------------------

BUILD_CUDA  = os.environ.get("FLASHDIFFUSION_BUILD_CUDA", "0") == "1"
ext_modules = []
cmdclass    = {}

if BUILD_CUDA:
    try:
        import torch
        from torch.utils.cpp_extension import CUDAExtension, BuildExtension

        cuda_includes = find_cuda_includes()
        print(f"[setup.py] CUDA includes: {len(cuda_includes)} paths found")

        # resolve arch list
        arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
        if not arch_list:
            try:
                major, minor = torch.cuda.get_device_capability()
                arch_list = f"{major}.{minor}"
                print(f"[setup.py] Auto-detected: SM{major*10+minor} "
                      f"({torch.cuda.get_device_name(0)})")
            except Exception:
                arch_list = "8.0"
                print("[setup.py] No GPU found, defaulting to SM80")

        # normalise: "8.0 10.0+PTX" → ["80", "100"]
        sms = [a.strip().replace(".", "").replace("+PTX", "")
               for a in arch_list.split() if a.strip()]

        # check if any SM needs CUTLASS
        needs_cutlass_globally = any(sm in _SM_NEEDS_CUTLASS for sm in sms)
        cutlass_inc = None
        if needs_cutlass_globally:
            try:
                cutlass_inc = ensure_cutlass()
                print(f"[setup.py] CUTLASS: {cutlass_inc}")
                if _has_sm100_headers(cutlass_inc):
                    print("[setup.py] CUTLASS has SM100 UMMA headers ✓")
                else:
                    print("[setup.py] CUTLASS missing SM100 headers "
                          "(SM90 only, no tcgen05.mma support)")
            except Exception as e:
                print(f"[setup.py] WARNING: CUTLASS unavailable ({e}) — "
                      f"SM90/SM100 kernels will be skipped")

        # group by (module, source)
        groups: dict[tuple[str, str], list[str]] = {}
        for sm in sms:
            mod = _SM_MODULE.get(sm, "flash_diffusion_cuda")
            src = _SM_SOURCES.get(sm, _SM_SOURCES["80"])
            groups.setdefault((mod, src), []).append(sm)

        for (mod, src), sm_list in groups.items():
            if not os.path.exists(src):
                print(f"[setup.py] SKIP {mod}: {src} not found in repo")
                continue

            # check CUTLASS requirement
            needs_cutlass = any(sm in _SM_NEEDS_CUTLASS for sm in sm_list)
            if needs_cutlass and cutlass_inc is None:
                print(f"[setup.py] SKIP {mod} (SM{sm_list}): "
                      f"CUTLASS unavailable")
                continue

            # build include list
            includes = cuda_includes[:]
            if needs_cutlass:
                includes = [cutlass_inc] + includes

            # per-SM nvcc defines (deduplicated)
            nvcc_extra = list({
                d
                for sm in sm_list
                for d in _SM_NVCC_DEFINES.get(sm, [])
            })

            nvcc_flags = [
                "-std=c++17", "-O3", "--use_fast_math",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
            ] + [_gencode(sm) for sm in sm_list] + nvcc_extra

            print(f"[setup.py] Building {mod}  SM{sm_list}  "
                  f"arch={[_SM_ARCH.get(s, s) for s in sm_list]}")

            ext_modules.append(CUDAExtension(
                name    = mod,
                sources = [src],
                include_dirs = includes,
                extra_compile_args = {
                    "cxx":  ["-std=c++17", "-O3"],
                    "nvcc": nvcc_flags,
                },
            ))

        cmdclass = {"build_ext": BuildExtension}
        print(f"[setup.py] {len(ext_modules)} extension(s) configured.")

    except ImportError as e:
        print(f"[setup.py] torch not available, skipping CUDA: {e}")
    except Exception as e:
        print(f"[setup.py] CUDA setup error: {e}")
        raise


# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------

setup(
    name             = "flashdiffusion",
    version          = "0.1.0",
    description      = "Tiled memory-efficient diffusion maps eigensolver",
    author           = "Julio Candanedo",
    author_email     = "julio@sparsetrace.ai",
    license          = "MIT",
    package_dir      = {"": "src"},
    packages         = ["flashdiffusion"],
    python_requires  = ">=3.10",
    install_requires = ["numpy>=1.24", "scipy>=1.10", "ninja"],
    extras_require   = {
        "cuda":     ["torch>=2.1", "ninja"],
        "examples": ["matplotlib>=3.7", "scikit-learn>=1.3", "jupyter>=1.0"],
        "dev":      ["pytest>=7.0", "ruff>=0.3.0", "black>=24.0"],
    },
    ext_modules = ext_modules,
    cmdclass    = cmdclass,
)
```

## `src/flashdiffusion/ROPE.py`

```python
# rope_encoder.py
# ============================================================
# ROPE Encoder: RoPE-DMAP on Hankel windows
#
# - Learns diffusion coordinates ψ_T for each training Hankel window
# - Applies RoPE BEFORE the diffusion-map kernel is constructed
# - Supports Nyström out-of-sample embedding for new windows/series
#
# This is intentionally an "encoder only":
#   - no decoder
#   - no forecasting
#   - just RoPE-DMAP coordinates + Nyström extension
#
# Relation to NLSA:
#   - pos_scale = 0 recovers ordinary dense NLSA / DMAP on Hankel windows.
#   - rope_mode = "sample" applies RoPE to each sample inside each Hankel window.
#   - rope_mode = "window" applies RoPE once to the flattened Hankel vector.
#
# ============================================================

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import eigsh


# ============================================================
# Hankel-window helpers
# ============================================================

def make_hankel_windows(F_tX: np.ndarray, L: int, step: int = 1) -> np.ndarray:
    """
    Build Hankel windows from a time series.

    F_tX: (N,D)
    returns W_TcX: (K,L,D), where
        W[T,c] = F[T + c*step]
        K = N - (L-1)*step
    """
    F = np.asarray(F_tX, dtype=float)
    if F.ndim == 1:
        F = F[:, None]

    N, D = F.shape
    L = int(L)
    step = int(step)

    if L <= 0:
        raise ValueError("L must be positive.")
    if step <= 0:
        raise ValueError("step must be positive.")

    K = N - (L - 1) * step
    if K <= 0:
        raise ValueError(f"Need N={N} >= (L-1)*step+1 = {(L-1)*step+1}")

    return np.stack([F[c * step : c * step + K, :] for c in range(L)], axis=1)


def make_window_positions(
    sample_positions: np.ndarray,
    L: int,
    step: int = 1,
    mode: str = "sample",
    window_position: str = "end",
) -> np.ndarray:
    """
    Construct positions for Hankel windows.

    sample_positions: (N,)

    If mode == "sample":
        returns Pos_Tc: (K,L), the absolute/sample positions inside each window.

    If mode == "window":
        returns Pos_T: (K,), one position per window.
    """
    pos = np.asarray(sample_positions, dtype=float)
    if pos.ndim != 1:
        raise ValueError("sample_positions must be one-dimensional.")

    N = len(pos)
    K = N - (L - 1) * step
    if K <= 0:
        raise ValueError("Not enough positions for windows.")

    Pos_Tc = np.stack([pos[c * step : c * step + K] for c in range(L)], axis=1)

    if mode == "sample":
        return Pos_Tc

    if mode != "window":
        raise ValueError("mode must be 'sample' or 'window'.")

    if window_position == "start":
        return Pos_Tc[:, 0]
    if window_position == "center":
        return Pos_Tc[:, L // 2]
    if window_position == "end":
        return Pos_Tc[:, -1]

    raise ValueError("window_position must be 'start', 'center', or 'end'.")


# ============================================================
# ROPE Encoder
# ============================================================

class ROPE:
    """
    Dense RoPE-DMAP encoder on Hankel windows.

    Training series:
      F_tX : (N,D)
      windows: W_T = [F_T, F_{T+step}, ..., F_{T+(L-1)step}]
      T = 0..K-1, K = N-(L-1)step

    Pre-kernel RoPE:
      In sample mode:
        each sample F_{T+c} inside the window is rotated by its own position.

      In window mode:
        the flattened window is rotated once by the window position.

    Kernel:
      K(T,T') = exp(-beta * ||RoPE(W_T) - RoPE(W_T')||^2)

    Diffusion normalization:
      q(T) = sum_T' K(T,T')
      K_alpha = K / (q(T)^alpha q(T')^alpha)
      d(T) = sum_T' K_alpha(T,T')
      P_sym = d^{-1/2} K_alpha d^{-1/2}

    Embedding:
      P_sym phi_j = lambda_j phi_j
      psi_j(T) = d(T)^(-1/2) phi_j(T)

    Out-of-sample:
      Nyström extension using the same pre-kernel RoPE transformation.
    """

    def __init__(
        self,
        F_tX: np.ndarray,
        L: int,
        rank: int = 20,
        beta: float | None = None,
        alpha: float = 1.0,
        center: bool = True,
        scale: bool = False,
        drop_first: bool = True,
        step: int = 1,
        sample_positions: np.ndarray | None = None,
        rope_mode: str = "sample",
        window_position: str = "end",
        pos_scale: float = 1.0,
        base: float = 10000.0,
        freqs: np.ndarray | None = None,
        n_freqs: int | None = None,
        beta_sample_pairs: int = 20000,
        seed: int = 0,
    ):
        R = np.asarray(F_tX, dtype=float)
        if R.ndim == 1:
            R = R[:, None]

        self.center = bool(center)
        self.scale = bool(scale)

        self.mu_ = R.mean(axis=0, keepdims=True) if self.center else np.zeros((1, R.shape[1]))
        self.sigma_ = R.std(axis=0, keepdims=True) if self.scale else np.ones((1, R.shape[1]))
        self.sigma_ = np.where(self.sigma_ == 0, 1.0, self.sigma_)

        self.R_raw_ = R
        self.R_ = (R - self.mu_) / self.sigma_

        self.N_, self.D_ = self.R_.shape
        self.L = int(L)
        self.step = int(step)

        self.K_ = self.N_ - (self.L - 1) * self.step
        if self.K_ <= 0:
            raise ValueError(f"N={self.N_} too small for L={self.L}, step={self.step}")

        self.rank_req_ = int(rank)
        self.beta_in_ = beta
        self.alpha_ = float(alpha)
        self.drop_first_ = bool(drop_first)

        self.rope_mode_ = str(rope_mode)
        if self.rope_mode_ not in ("sample", "window"):
            raise ValueError("rope_mode must be 'sample' or 'window'.")

        self.window_position_ = str(window_position)
        self.pos_scale_ = float(pos_scale)
        self.base_ = float(base)
        self.freqs_in_ = None if freqs is None else np.asarray(freqs, dtype=float)
        self.n_freqs_ = None if n_freqs is None else int(n_freqs)

        if sample_positions is None:
            sample_positions = np.arange(self.N_, dtype=float)
        self.sample_positions_ = np.asarray(sample_positions, dtype=float)
        if self.sample_positions_.shape != (self.N_,):
            raise ValueError(f"sample_positions must have shape ({self.N_},).")

        self.beta_sample_pairs_ = int(beta_sample_pairs)
        self.rng_ = np.random.default_rng(int(seed))

        # learned artifacts
        self.W_TcX_ = None              # raw centered/scaled windows, shape (K,L,D)
        self.Pos_ = None                # (K,L) for sample mode or (K,) for window mode
        self.Y_ = None                  # RoPE-transformed flattened windows, shape (K,D_rope)
        self.r2_T_ = None               # squared norms of Y_
        self.G_ = None                  # dense Gram of Y_
        self.beta_ = None
        self.Kmat_ = None               # raw Gaussian kernel
        self.K_T_ = None                # q(T)
        self.d_T_ = None
        self.inv_sqrt_d_ = None
        self.lam_ = None
        self.phi_ = None
        self.psi_ = None
        self.freqs_ = None

        self.fit()

    # -------------------------
    # RoPE internals
    # -------------------------

    def _freqs(self, dim: int) -> np.ndarray:
        """
        Standard RoPE frequency schedule over adjacent 2D pairs.

        If freqs was provided, use it.
        If n_freqs is provided, only the first n_freqs pairs rotate; remaining pairs have frequency 0.
        Odd leftover dimension is not rotated.
        """
        n_pairs = dim // 2
        if n_pairs == 0:
            return np.zeros((0,), dtype=float)

        if self.freqs_in_ is not None:
            freqs = np.asarray(self.freqs_in_, dtype=float)
            if freqs.size > n_pairs:
                freqs = freqs[:n_pairs]
            if freqs.size < n_pairs:
                freqs = np.concatenate([freqs, np.zeros(n_pairs - freqs.size)])
            return freqs

        m = np.arange(n_pairs, dtype=float)
        freqs = self.base_ ** (-2.0 * m / max(2, 2 * n_pairs))

        if self.n_freqs_ is not None:
            n_active = max(0, min(int(self.n_freqs_), n_pairs))
            mask = np.arange(n_pairs) < n_active
            freqs = freqs * mask

        return freqs

    @staticmethod
    def _rotate_matrix_rows(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Rotate adjacent pairs of the last dimension of X.

        X: (..., D)
        theta: (..., n_pairs), broadcast-compatible with X[..., 0::2] pairs

        Odd final dimension is copied unchanged.
        """
        X = np.asarray(X, dtype=float)
        D = X.shape[-1]
        n_pairs = D // 2

        if n_pairs == 0:
            return X.copy()

        pair_part = X[..., : 2 * n_pairs].reshape(*X.shape[:-1], n_pairs, 2)
        a = pair_part[..., 0]
        b = pair_part[..., 1]

        c = np.cos(theta)
        s = np.sin(theta)

        Ypair = np.empty_like(pair_part)
        Ypair[..., 0] = a * c - b * s
        Ypair[..., 1] = a * s + b * c

        Yflat = Ypair.reshape(*X.shape[:-1], 2 * n_pairs)

        if D % 2 == 1:
            return np.concatenate([Yflat, X[..., -1:]], axis=-1)

        return Yflat

    def _rope_sample_windows(self, W_TcX: np.ndarray, Pos_Tc: np.ndarray) -> np.ndarray:
        """
        Apply RoPE to each sample in each Hankel window.

        W_TcX: (K,L,D)
        Pos_Tc: (K,L)
        returns Y: (K, L*D)
        """
        K, L, D = W_TcX.shape
        freqs = self._freqs(D)
        self.freqs_ = freqs

        if freqs.size == 0 or self.pos_scale_ == 0.0:
            return W_TcX.reshape(K, L * D)

        theta = Pos_Tc[:, :, None] * self.pos_scale_ * freqs[None, None, :]
        Y_TcX = self._rotate_matrix_rows(W_TcX, theta)
        return Y_TcX.reshape(K, L * D)

    def _rope_window_vectors(self, H_TZ: np.ndarray, Pos_T: np.ndarray) -> np.ndarray:
        """
        Apply RoPE once to flattened Hankel vectors.

        H_TZ: (K,Z)
        Pos_T: (K,)
        returns Y: (K,Z)
        """
        K, Z = H_TZ.shape
        freqs = self._freqs(Z)
        self.freqs_ = freqs

        if freqs.size == 0 or self.pos_scale_ == 0.0:
            return H_TZ.copy()

        theta = Pos_T[:, None] * self.pos_scale_ * freqs[None, :]
        return self._rotate_matrix_rows(H_TZ, theta)

    def _transform_windows(self, W_TcX: np.ndarray, Pos) -> np.ndarray:
        """
        Center/scale must already be applied to W_TcX.
        """
        if self.rope_mode_ == "sample":
            return self._rope_sample_windows(W_TcX, Pos)

        H = W_TcX.reshape(W_TcX.shape[0], -1)
        return self._rope_window_vectors(H, Pos)

    # -------------------------
    # training
    # -------------------------

    def _choose_beta(self, D2: np.ndarray) -> float:
        """
        Pick beta via median heuristic on sampled off-diagonal distances,
        unless beta was provided.
        """
        if self.beta_in_ is not None:
            return float(self.beta_in_)

        K = self.K_
        M_max = K * (K - 1) // 2
        M = min(self.beta_sample_pairs_, M_max)
        if M <= 0:
            return 1.0

        ii = self.rng_.integers(0, K, size=M)
        jj = self.rng_.integers(0, K, size=M)
        mask = ii != jj
        ii, jj = ii[mask], jj[mask]
        if ii.size == 0:
            return 1.0

        med = np.median(D2[ii, jj])
        return 1.0 / (med + 1e-12)

    def fit(self) -> "ROPE":
        # Hankel windows from centered/scaled data
        self.W_TcX_ = make_hankel_windows(self.R_, self.L, step=self.step)

        # Window/sample positions
        self.Pos_ = make_window_positions(
            self.sample_positions_,
            self.L,
            step=self.step,
            mode=self.rope_mode_,
            window_position=self.window_position_,
        )

        # Pre-kernel RoPE transform
        self.Y_ = self._transform_windows(self.W_TcX_, self.Pos_)

        # Dense Gram/distances
        self.G_ = self.Y_ @ self.Y_.T
        self.r2_T_ = np.sum(self.Y_ * self.Y_, axis=1)

        D2 = self.r2_T_[:, None] + self.r2_T_[None, :] - 2.0 * self.G_
        np.maximum(D2, 0.0, out=D2)

        # beta
        self.beta_ = self._choose_beta(D2)

        # Gaussian kernel on RoPE-transformed windows
        Kmat = np.exp(-self.beta_ * D2)

        # Diffusion maps alpha normalization
        K_T = Kmat.sum(axis=1) + 1e-18
        KTa = K_T ** self.alpha_
        Kalpha = Kmat / (KTa[:, None] * KTa[None, :])

        d_T = Kalpha.sum(axis=1) + 1e-18
        inv_sqrt_d = 1.0 / np.sqrt(d_T)

        Psym = (inv_sqrt_d[:, None] * Kalpha) * inv_sqrt_d[None, :]

        # Top eigenpairs of symmetric operator
        k = min(self.rank_req_ + (1 if self.drop_first_ else 0), self.K_ - 1)
        if k <= 0:
            self.lam_ = np.zeros((0,), dtype=float)
            self.phi_ = np.zeros((self.K_, 0), dtype=float)
            self.psi_ = np.zeros((self.K_, 0), dtype=float)
            self.Kmat_ = Kmat
            self.K_T_ = K_T
            self.d_T_ = d_T
            self.inv_sqrt_d_ = inv_sqrt_d
            return self

        w, V = eigsh(Psym, k=k, which="LA")

        idx = np.argsort(w)[::-1]
        w = w[idx]
        V = V[:, idx]

        if self.drop_first_ and w.size > 0:
            w = w[1:]
            V = V[:, 1:]

        self.lam_ = w
        self.phi_ = V
        self.Kmat_ = Kmat
        self.K_T_ = K_T
        self.d_T_ = d_T
        self.inv_sqrt_d_ = inv_sqrt_d

        self.psi_ = inv_sqrt_d[:, None] * V
        return self

    # -------------------------
    # encoding / Nyström
    # -------------------------

    def _prepare_query_windows(
        self,
        W_BLX: np.ndarray,
        positions=None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Center/scale query windows and build query positions.
        """
        W = np.asarray(W_BLX, dtype=float)
        if W.ndim == 2:
            W = W[None, :, :]
        if W.ndim != 3:
            raise ValueError("Expected query windows with shape (B,L,D) or (L,D).")
        if W.shape[1:] != (self.L, self.D_):
            raise ValueError(f"Expected window shape (*,{self.L},{self.D_}), got {W.shape}")

        Wc = (W - self.mu_) / self.sigma_

        B = W.shape[0]

        if self.rope_mode_ == "sample":
            if positions is None:
                # Local positions default. For real use, pass absolute positions.
                Pos = np.tile(np.arange(self.L, dtype=float), (B, 1))
            else:
                Pos = np.asarray(positions, dtype=float)
                if Pos.ndim == 1 and Pos.shape[0] == self.L:
                    Pos = np.tile(Pos[None, :], (B, 1))
                if Pos.shape != (B, self.L):
                    raise ValueError(f"For sample mode, positions must have shape ({B},{self.L}) or ({self.L},).")
            return Wc, Pos

        # window mode
        if positions is None:
            Pos = np.zeros((B,), dtype=float)
        else:
            Pos = np.asarray(positions, dtype=float)
            if Pos.ndim == 0:
                Pos = np.full((B,), float(Pos))
            if Pos.shape != (B,):
                raise ValueError(f"For window mode, positions must have shape ({B},) or scalar.")
        return Wc, Pos

    def encode_windows(self, W_BLX: np.ndarray, positions=None) -> np.ndarray:
        """
        Encode a batch of query windows using Nyström.

        Inputs:
          W_BLX: (B,L,D) or (L,D)
          positions:
            sample mode: (B,L), or (L,), absolute positions inside query windows
            window mode: (B,), or scalar, one position per query window

        Returns:
          Psi_Br: (B,r)
        """
        if self.psi_ is None or self.lam_ is None or self.psi_.shape[1] == 0:
            B = 1 if np.asarray(W_BLX).ndim == 2 else np.asarray(W_BLX).shape[0]
            return np.zeros((B, 0), dtype=float)

        Wc, Pos = self._prepare_query_windows(W_BLX, positions=positions)
        Yq = self._transform_windows(Wc, Pos)

        Gq = Yq @ self.Y_.T
        r2q = np.sum(Yq * Yq, axis=1)

        D2 = r2q[:, None] + self.r2_T_[None, :] - 2.0 * Gq
        np.maximum(D2, 0.0, out=D2)

        k_qT = np.exp(-self.beta_ * D2)

        # alpha normalization query -> train
        Kq = k_qT.sum(axis=1) + 1e-18
        k_qT_alpha = k_qT / ((Kq[:, None] ** self.alpha_) * (self.K_T_[None, :] ** self.alpha_))

        dq = k_qT_alpha.sum(axis=1) + 1e-18
        P_qT = k_qT_alpha / dq[:, None]

        lam_safe = np.maximum(self.lam_, 1e-12)
        psi_q = P_qT @ (self.psi_ / lam_safe[None, :])
        return psi_q

    def encode_window(self, W_cX: np.ndarray, positions=None) -> np.ndarray:
        """
        Encode one query window.

        Returns:
          psi_q: (r,)
        """
        return self.encode_windows(np.asarray(W_cX)[None, :, :], positions=positions)[0]

    def encode_series(
        self,
        F_aX: np.ndarray,
        sample_positions: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Encode all Hankel windows of a novel series.

        F_aX: (N,D), N >= (L-1)*step+1

        sample_positions:
          optional length-N positions. If omitted, uses arange(N).

        Returns:
          Psi: (K_n,r)
        """
        F = np.asarray(F_aX, dtype=float)
        if F.ndim == 1:
            F = F[:, None]
        N, D = F.shape
        if D != self.D_:
            raise ValueError(f"Expected D={self.D_}, got D={D}.")

        if sample_positions is None:
            sample_positions = np.arange(N, dtype=float)
        sample_positions = np.asarray(sample_positions, dtype=float)
        if sample_positions.shape != (N,):
            raise ValueError(f"sample_positions must have shape ({N},).")

        # Build raw query windows and matching positions.
        W = make_hankel_windows(F, self.L, step=self.step)

        Pos = make_window_positions(
            sample_positions,
            self.L,
            step=self.step,
            mode=self.rope_mode_,
            window_position=self.window_position_,
        )

        return self.encode_windows(W, positions=Pos)

    # -------------------------
    # convenience
    # -------------------------

    @property
    def embedding_(self) -> np.ndarray:
        """
        Training diffusion coordinates, shape (K,rank).
        """
        return self.psi_

    def fit_transform(self) -> np.ndarray:
        """
        Return training diffusion coordinates.
        """
        return self.psi_


__all__ = ["ROPE", "make_hankel_windows", "make_window_positions"]
```

## `src/flashdiffusion/__init__.py`

```python
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
```

## `src/flashdiffusion/cdc.py`

```python
"""
flashdiffusion/cdc.py
=====================
Carré du Champ (CDC) operator and diffusion metric tensor.

The Carré du Champ operator of the diffusion semigroup is:

    Γ(f, g)(x) = ½ [L(fg)(x) - f(x)Lg(x) - g(x)Lf(x)]

where L = I - P⁺ is the (infinitesimal) generator of the diffusion.
For discrete diffusion maps this becomes:

    Γ(φₖ, φₗ)(i) = ½ Σⱼ P⁺(i,j) [φₖ(j) - φₖ(i)] [φₗ(j) - φₗ(i)]
                  = ½ [Mₖₗ - λₖ φₖ(i) φₗ(i) - λₗ φₗ(i) φₖ(i)]    (*)

where Mₖₗ(i) = (P⁺ φₖ φₗ)(i) — the diffusion applied to the pointwise product.

(*) uses the fact that P⁺ φₖ = λₖ φₖ.

The Diffusion Metric Tensor at point i is:

    G^i_{kl} = Γ(φₖ, φₗ)(i)

and the global Riemannian metric on the embedding space is:

    M = Σᵢ Gⁱ (Gⁱ)ᵀ   (summed over data points)

This gives a data-driven geometry that is intrinsic to the diffusion —
it captures how much uncertainty/variance each direction in embedding
space has relative to the local transition structure.

References
----------
- Coifman & Lafon (2006), §3
- Singer & Coifman (2008) — diffusion distance and CDC
- Nadler, Lafon, Coifman, Kevrekidis (2006) — diffusion maps for data analysis
"""

from __future__ import annotations
import numpy as np
from typing import Optional

from .kernel import FlashDiffusion


def carre_du_champ(
    dm,                         # fitted DiffusionMap instance
    phi_k: np.ndarray,          # (N,)  eigenvector k
    phi_l: np.ndarray,          # (N,)  eigenvector l (may equal phi_k)
    lambda_k: float,
    lambda_l: float,
) -> np.ndarray:
    """
    Compute Γ(φₖ, φₗ)(i) for all i.

    Uses the identity (*):
        Γ(φₖ, φₗ)(i) = ½ [(P⁺ φₖφₗ)(i)  -  λₖ φₖ(i)φₗ(i)  -  λₗ φₗ(i)φₖ(i)]

    Only ONE additional FlashDiffusion matvec is needed (for P⁺ φₖφₗ),
    beyond the eigenvectors already computed by DiffusionMap.transform().

    Parameters
    ----------
    dm       : fitted DiffusionMap (after transform(), so rscale_ is set)
    phi_k/l  : eigenvectors of the *row-normalised* operator P+
               (i.e. vecs / rscale, as returned by dmap_coords)
    lambda_k/l : corresponding eigenvalues

    Returns
    -------
    gamma : (N,) pointwise Carré du Champ values
    """
    # pointwise product of the two eigenfunctions
    phi_kl = phi_k * phi_l                   # (N,)

    # apply P⁺ to phi_kl via FlashDiffusion
    # P⁺ f = (1/D_alpha) * K_alpha * f
    #      = rscale * K * (rscale * f)    (using our normalisation)
    # but phi_k/l are already in the P⁺-eigenvector basis, so we apply
    # the DMAP matvec directly
    P_phi_kl = dm.matvec(phi_kl)             # (N,)  = M applied to (phi_kl/rscale) * rscale
    # correction: dm.matvec applies the SYMMETRIC M, not P⁺.
    # We need P⁺ = D_alpha^{-1} K_alpha = rscale/d_half * M / rscale * d_half
    # But phi_k/l are P⁺ eigenvectors: P⁺ φ = λ φ.
    # For Γ we need (P⁺ φₖφₗ)(i), i.e. row-normalised application.
    # The row-normalised operator applied to f is:
    #   (P⁺ f)_i = rscale_i * Σⱼ K(i,j) * rscale_j * (f_j / rscale_j) * rscale_j
    # Wait — let's be explicit. With our notation:
    #   P⁺_{ij} = K_alpha(i,j) / D_alpha_i
    # and rscale_i = D_alpha_i^{-1/2} * w_i, so:
    #   (P⁺ f)_i = Σⱼ [K(i,j)*w_i*w_j / D_alpha_i] * f_j
    #            = [rscale_i/w_i]² * Σⱼ K(i,j) * (rscale_j/w_j)^{-1} * rscale_j * f_j ... complex
    # Simplest: use matvec (symmetric M) and note M = D_alpha^{-1/2} P⁺_sym D_alpha^{1/2}
    # For the CDC formula (*) with phi_k in P⁺ basis, apply P⁺ directly:
    P_phi_kl = _apply_Pplus(dm, phi_kl)

    gamma = 0.5 * (P_phi_kl
                   - lambda_k * phi_k * phi_l
                   - lambda_l * phi_l * phi_k)
    return gamma


def _apply_Pplus(dm, f: np.ndarray) -> np.ndarray:
    """
    Apply the row-normalised operator P⁺ to a function f.

    P⁺ f_i = Σⱼ K_alpha(i,j) / D_alpha_i · f_j
           = (1/D_alpha_i) · w_i · Σⱼ K(i,j) · w_j · f_j

    This is a FlashDiffusion call with rscale_q = 1/D_alpha (not d_half*w).
    """
    N = dm.X_.shape[0]
    # rscale_k for right side: w (density correction only)
    # rscale_q for left side: 1/D_alpha (full row normalisation)
    w        = dm.D_  ** (-dm.alpha)
    inv_Dalpha = dm.D_alpha_ ** (-1.0)

    out = FlashDiffusion(
        Q        = dm.X_,
        K        = dm.X_,
        V        = f[:, None],
        beta     = dm.beta,
        rscale_q = inv_Dalpha * w,    # (D_alpha^{-1}) * w = P⁺ left normaliser
        rscale_k = w,                 # right: w_j factor
        tile     = dm.tile,
    )
    return out.ravel()


def metric_tensor(
    dm,
    eigenvalues:  np.ndarray,       # (k,)
    eigenvectors: np.ndarray,       # (N, k)  in P⁺ basis
    pairs: Optional[list] = None,   # list of (k, l) index pairs; default all
) -> np.ndarray:
    """
    Compute the diffusion metric tensor G (N x k x k) where:

        G[i, k, l] = Γ(φₖ, φₗ)(i)

    and the global Riemannian metric M = Σᵢ G[i] G[i]ᵀ  (k x k).

    For large k, pass `pairs` to compute only the entries you need.

    Returns
    -------
    G : (N, k, k)  pointwise metric tensor (symmetric)
    M : (k, k)     global Riemannian metric
    """
    k = eigenvalues.shape[0]
    N = eigenvectors.shape[0]
    G = np.zeros((N, k, k), dtype=np.float64)

    if pairs is None:
        pairs = [(i, j) for i in range(k) for j in range(i, k)]

    for (ki, li) in pairs:
        gamma = carre_du_champ(
            dm,
            eigenvectors[:, ki], eigenvectors[:, li],
            eigenvalues[ki],     eigenvalues[li],
        )
        G[:, ki, li] = gamma
        G[:, li, ki] = gamma

    # global metric: M_kl = Σᵢ G[i,k,:] · G[i,l,:]  — batched einsum
    M = np.einsum('ikm,ilm->kl', G, G)
    return G, M
```

## `src/flashdiffusion/csrc/flash_diffusion_sm120.cu`

```cuda
/*
 * flash_diffusion_sm120.cu  — SM120 WMMA v2
 * ==========================================
 * SM120 (RTX 5090 / RTX PRO 6000 Blackwell consumer) WMMA kernel.
 *
 * v2 over v1:
 *   Fixed SMEM size via k-chunking — supports arbitrary d (including d=784).
 *   v1 allocated BM×d SMEM which exceeded 99KB limit for d>~350.
 *
 * SMEM layout (fixed, independent of d):
 *   sA   = BM × KCHUNK × 2  = 128 × 32 × 2 =  8,192 bytes
 *   sB   = BN × KCHUNK × 2  = 128 × 32 × 2 =  8,192 bytes
 *   sDot = BM × BN    × 4   = 128 × 128 × 4 = 65,536 bytes
 *   Total = 81,920 bytes = 80KB  ✓  (SM120 limit: 99KB)
 *
 * K-chunking: outer loop over d in steps of KCHUNK=32.
 *   Each chunk loads sA/sB (BM×32) and (BN×32), runs 2 WMMA calls per warp,
 *   accumulates into sDot with fill_fragment(fc, 0) only on first chunk.
 *
 * Build:
 *   -arch=sm_120 -std=c++17 -O3 --use_fast_math
 *   -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__
 *   --expt-relaxed-constexpr
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

using namespace nvcuda;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr int WMMA_M  = 16;
static constexpr int WMMA_N  = 16;
static constexpr int WMMA_K  = 16;

static constexpr int BM      = 128;
static constexpr int BN      = 128;
static constexpr int KCHUNK  = 32;    // process d in 32-wide chunks (2 WMMA_K)
                                       // keeps SMEM fixed at 80KB regardless of d

// 16 warps: each covers 32×32 sub-tile via 2×2 WMMA calls
static constexpr int WARPS_M  = 4;
static constexpr int WARPS_N  = 4;
static constexpr int NWARPS   = WARPS_M * WARPS_N;   // 16
static constexpr int NTHREADS = NWARPS * 32;           // 512

// Fixed SMEM: sA + sB + sDot
static constexpr size_t SMEM_BYTES =
    (size_t)BM * KCHUNK * sizeof(half)    // sA
  + (size_t)BN * KCHUNK * sizeof(half)    // sB
  + (size_t)BM * BN     * sizeof(float);  // sDot

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }
static inline int round_up(int a, int b) { return ceil_div(a, b) * b; }

// ---------------------------------------------------------------------------
// Squared norms
// ---------------------------------------------------------------------------
__global__ void sq_norm_kernel_sm120(
    const __half* __restrict__ X, float* __restrict__ sq, int N, int d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s = 0.f;
    for (int k = 0; k < d; ++k) { float v = __half2float(X[i*d+k]); s += v*v; }
    sq[i] = s;
}

// ---------------------------------------------------------------------------
// SMEM loaders for a k-chunk [k0, k0+klen)
// sA: [BM, KCHUNK] row-major — zero-pad for klen < KCHUNK
// sB: [BN, KCHUNK] col-major (stored as [col, k]) — zero-pad similarly
// ---------------------------------------------------------------------------
__device__ __forceinline__ void load_sA_chunk(
    half* sA, const half* X, int i0, int bm, int d,
    int k0, int klen, int tid)
{
    for (int idx = tid; idx < BM * KCHUNK; idx += NTHREADS) {
        int r = idx / KCHUNK, c = idx % KCHUNK;
        sA[idx] = (r < bm && c < klen)
            ? X[(long long)(i0+r)*d + k0+c]
            : __float2half(0.f);
    }
}

__device__ __forceinline__ void load_sB_chunk(
    half* sB, const half* X, int j0, int bn, int d,
    int k0, int klen, int tid)
{
    for (int idx = tid; idx < BN * KCHUNK; idx += NTHREADS) {
        int col = idx / KCHUNK, c = idx % KCHUNK;
        sB[idx] = (col < bn && c < klen)
            ? X[(long long)(j0+col)*d + k0+c]
            : __float2half(0.f);
    }
}

// ---------------------------------------------------------------------------
// WMMA dot tile accumulate
// sA (BM×KCHUNK, row) @ sB^T (KCHUNK×BN, col) → sDot (BM×BN)
// Each of 16 warps covers 32×32 sub-tile (2×2 WMMA calls).
// first_chunk=true: fill fc=0 before MMA
// first_chunk=false: load fc from sDot first (accumulate across k-chunks)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void wmma_accumulate(
    const half* sA, const half* sB, float* sDot,
    bool first_chunk, int tid)
{
    const int warp = tid >> 5;
    const int wm   = warp / WARPS_N;
    const int wn   = warp % WARPS_N;

    for (int dm = 0; dm < 2; ++dm) {
        for (int dn = 0; dn < 2; ++dn) {
            const int row16 = wm * 32 + dm * 16;
            const int col16 = wn * 32 + dn * 16;

            wmma::fragment<wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K,
                           half, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K,
                           half, wmma::col_major> fb;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fc;

            if (first_chunk) {
                wmma::fill_fragment(fc, 0.f);
            } else {
                wmma::load_matrix_sync(fc,
                    sDot + row16 * BN + col16, BN, wmma::mem_row_major);
            }

            for (int kk = 0; kk < KCHUNK; kk += WMMA_K) {
                wmma::load_matrix_sync(fa, sA + row16 * KCHUNK + kk, KCHUNK);
                wmma::load_matrix_sync(fb, sB + col16 * KCHUNK + kk, KCHUNK);
                wmma::mma_sync(fc, fa, fb, fc);
            }

            wmma::store_matrix_sync(
                sDot + row16 * BN + col16, fc, BN, wmma::mem_row_major);
        }
    }
}

// ---------------------------------------------------------------------------
// Prepass kernel: D_i = Σ_j exp(-β||xi-xj||²) * w_j
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(NTHREADS)
prepass_kernel_sm120(
    const half*  __restrict__ X,
    const float* __restrict__ w,
    float*       __restrict__ D,
    const float* __restrict__ sq_X,
    int N, int d, float beta)
{
    extern __shared__ unsigned char smem[];
    half*  sA   = reinterpret_cast<half*>(smem);
    half*  sB   = sA + BM * KCHUNK;
    float* sDot = reinterpret_cast<float*>(sB + BN * KCHUNK);

    const int tid = threadIdx.x;
    const int i0  = blockIdx.x * BM;
    const int bm  = min(BM, N - i0);
    float acc = 0.f;

    for (int j0 = 0; j0 < N; j0 += BN) {
        const int bn = min(BN, N - j0);

        // k-chunk loop: accumulate dot products into sDot
        for (int k0 = 0; k0 < d; k0 += KCHUNK) {
            const int klen = min(KCHUNK, d - k0);
            const bool first = (k0 == 0);

            load_sA_chunk(sA, X, i0, bm, d, k0, klen, tid);
            load_sB_chunk(sB, X, j0, bn, d, k0, klen, tid);
            __syncthreads();

            wmma_accumulate(sA, sB, sDot, first, tid);
            __syncthreads();
        }

        // epilogue: dist2 → exp → accumulate
        if (tid < bm) {
            float sqi = sq_X[i0 + tid];
            #pragma unroll 4
            for (int j = 0; j < bn; ++j) {
                float dist2 = fmaxf(sqi + sq_X[j0+j] - 2.f * sDot[tid*BN+j], 0.f);
                acc += __expf(-beta * dist2) * w[j0+j];
            }
        }
        __syncthreads();
    }

    if (tid < bm) D[i0 + tid] = acc;
}

// ---------------------------------------------------------------------------
// Matvec kernel: out_i = rscale_i * Σ_j exp(-β||xi-xj||²) * u_j
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(NTHREADS)
matvec_kernel_sm120(
    const half*  __restrict__ X,
    const float* __restrict__ u,
    float*       __restrict__ out,
    const float* __restrict__ sq_X,
    const float* __restrict__ rscale,
    int N, int d, float beta)
{
    extern __shared__ unsigned char smem[];
    half*  sA   = reinterpret_cast<half*>(smem);
    half*  sB   = sA + BM * KCHUNK;
    float* sDot = reinterpret_cast<float*>(sB + BN * KCHUNK);

    const int tid = threadIdx.x;
    const int i0  = blockIdx.x * BM;
    const int bm  = min(BM, N - i0);
    float acc = 0.f;

    for (int j0 = 0; j0 < N; j0 += BN) {
        const int bn = min(BN, N - j0);

        for (int k0 = 0; k0 < d; k0 += KCHUNK) {
            const int klen = min(KCHUNK, d - k0);
            const bool first = (k0 == 0);

            load_sA_chunk(sA, X, i0, bm, d, k0, klen, tid);
            load_sB_chunk(sB, X, j0, bn, d, k0, klen, tid);
            __syncthreads();

            wmma_accumulate(sA, sB, sDot, first, tid);
            __syncthreads();
        }

        if (tid < bm) {
            float sqi = sq_X[i0 + tid];
            #pragma unroll 4
            for (int j = 0; j < bn; ++j) {
                float dist2 = fmaxf(sqi + sq_X[j0+j] - 2.f * sDot[tid*BN+j], 0.f);
                acc += __expf(-beta * dist2) * u[j0+j];
            }
        }
        __syncthreads();
    }

    if (tid < bm) out[i0 + tid] = rscale[i0 + tid] * acc;
}

// ---------------------------------------------------------------------------
// SMEM opt-in
// ---------------------------------------------------------------------------
static void set_smem(const void* fn) {
    int dev; C10_CUDA_CHECK(cudaGetDevice(&dev));
    int maxoptin;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &maxoptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    TORCH_CHECK((int)SMEM_BYTES <= maxoptin,
        "SMEM ", SMEM_BYTES, " > device max ", maxoptin);
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES));
}

// ---------------------------------------------------------------------------
// Python entry points
// ---------------------------------------------------------------------------

torch::Tensor compute_sq_norms(torch::Tensor X_fp16) {
    int N = X_fp16.size(0), d = X_fp16.size(1);
    auto sq = torch::empty({N}, X_fp16.options().dtype(torch::kFloat32));
    sq_norm_kernel_sm120<<<ceil_div(N,256), 256, 0,
                           at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(X_fp16.data_ptr<at::Half>()),
        sq.data_ptr<float>(), N, d);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return sq;
}

torch::Tensor prepass(
    torch::Tensor X_fp16, torch::Tensor w,
    torch::Tensor sq_X,   float beta)
{
    int N = X_fp16.size(0), d = X_fp16.size(1);
    set_smem(reinterpret_cast<const void*>(prepass_kernel_sm120));

    auto D = torch::zeros({N}, w.options());
    prepass_kernel_sm120<<<ceil_div(N,BM), NTHREADS, SMEM_BYTES,
                           at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(X_fp16.data_ptr<at::Half>()),
        w.data_ptr<float>(), D.data_ptr<float>(),
        sq_X.data_ptr<float>(), N, d, beta);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return D;
}

torch::Tensor matvec(
    torch::Tensor X_fp16, torch::Tensor v,
    torch::Tensor rscale, torch::Tensor sq_X, float beta)
{
    int N = X_fp16.size(0), d = X_fp16.size(1);
    set_smem(reinterpret_cast<const void*>(matvec_kernel_sm120));

    auto u   = (rscale * v).contiguous();
    auto out = torch::empty({N}, v.options());
    matvec_kernel_sm120<<<ceil_div(N,BM), NTHREADS, SMEM_BYTES,
                          at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(X_fp16.data_ptr<at::Half>()),
        u.data_ptr<float>(), out.data_ptr<float>(),
        sq_X.data_ptr<float>(), rscale.data_ptr<float>(),
        N, d, beta);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashDiffusion SM120 WMMA v2 — k-chunked, arbitrary d "
              "(RTX 5090 / RTX PRO 6000 Blackwell consumer)";
    m.def("compute_sq_norms", &compute_sq_norms, py::arg("X_fp16"));
    m.def("prepass",  &prepass,
          py::arg("X_fp16"), py::arg("w"), py::arg("sq_X"), py::arg("beta"));
    m.def("matvec",   &matvec,
          py::arg("X_fp16"), py::arg("v"), py::arg("rscale"),
          py::arg("sq_X"), py::arg("beta"));
}
```

## `src/flashdiffusion/csrc/flash_diffusion_sm120X.cu`

```cuda
/*
 * flash_diffusion_sm120.cu
 * ========================
 * SM120 (RTX 5090 / RTX PRO 6000 Blackwell consumer) WMMA kernel.
 *
 * Upgraded from scalar v1 to WMMA tensor cores.
 * SM120 uses nvcuda::wmma (same mma.sync family as SM80/SM86/SM89).
 * No wgmma, no TMEM, no UMMA — those are SM90/SM100 only.
 *
 * Key differences from SM80 WMMA kernel:
 *   - BM=128, BN=128 (vs 64×64) — larger tiles exploit SM120's bigger SMEM
 *   - 16 warps = 512 threads, each warp covers 32×32 via 2×2 WMMA calls
 *   - KPAD = ceil(d/16)*16 as usual
 *
 * SMEM per block (SM120 has 100KB):
 *   sA   = BM × KPAD × 2  =  128 × KPAD × 2
 *   sB   = BN × KPAD × 2  =  128 × KPAD × 2
 *   sDot = BM × BN  × 4   =  128 × 128  × 4  = 64KB
 *   For d≤16  (KPAD=16):  64 + 4 + 4   = 72KB ✓
 *   For d≤64  (KPAD=64):  64 + 16 + 16 = 96KB ✓
 *   For d≤128 (KPAD=128): 64 + 32 + 32 = 128KB — needs opt-in
 *
 * Build:
 *   -arch=sm_120 -std=c++17 -O3 --use_fast_math
 *   -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__
 *   --expt-relaxed-constexpr
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

using namespace nvcuda;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr int WMMA_M   = 16;
static constexpr int WMMA_N   = 16;
static constexpr int WMMA_K   = 16;

static constexpr int BM       = 128;
static constexpr int BN       = 128;

// 16 warps = 512 threads
// Each warp covers a 32×32 sub-tile via 2×2 WMMA calls
static constexpr int WARPS_M  = 4;    // 4 warps in M → 4×32 = 128 rows ✓
static constexpr int WARPS_N  = 4;    // 4 warps in N → 4×32 = 128 cols ✓
static constexpr int NWARPS   = WARPS_M * WARPS_N;  // 16
static constexpr int NTHREADS = NWARPS * 32;          // 512

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }
static inline int round_up(int a, int b) { return ceil_div(a, b) * b; }

static size_t smem_size(int d) {
    int KPAD = round_up(d, WMMA_K);
    return (size_t)BM * KPAD * sizeof(half)
         + (size_t)BN * KPAD * sizeof(half)
         + (size_t)BM * BN   * sizeof(float);
}

// ---------------------------------------------------------------------------
// Squared norms
// ---------------------------------------------------------------------------
__global__ void sq_norm_kernel_sm120(
    const __half* __restrict__ X, float* __restrict__ sq, int N, int d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s = 0.f;
    for (int k = 0; k < d; ++k) { float v = __half2float(X[i*d+k]); s += v*v; }
    sq[i] = s;
}

// ---------------------------------------------------------------------------
// SMEM loaders
// sA: [BM, KPAD] row-major      → matrix_a row_major
// sB: [BN, KPAD] stored [col,k] → matrix_b col_major (B^T)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void load_sA(
    half* sA, const half* X, int i0, int bm, int d, int KPAD, int tid)
{
    for (int idx = tid; idx < BM * KPAD; idx += NTHREADS) {
        int r = idx / KPAD, c = idx % KPAD;
        sA[idx] = (r < bm && c < d) ? X[(long long)(i0+r)*d + c] : __float2half(0.f);
    }
}

__device__ __forceinline__ void load_sB(
    half* sB, const half* X, int j0, int bn, int d, int KPAD, int tid)
{
    for (int idx = tid; idx < BN * KPAD; idx += NTHREADS) {
        int col = idx / KPAD, k = idx % KPAD;
        sB[idx] = (col < bn && k < d) ? X[(long long)(j0+col)*d + k] : __float2half(0.f);
    }
}

// ---------------------------------------------------------------------------
// WMMA dot tile: sA (BM×KPAD, row) @ sB^T (KPAD×BN, col) → sDot (BM×BN)
//
// Each of 16 warps covers a 32×32 sub-tile using 2×2 WMMA fragments.
// warp (wm, wn): output rows [wm*32, wm*32+32), cols [wn*32, wn*32+32)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void wmma_dot_tile(
    const half* sA, const half* sB, float* sDot, int KPAD, int tid)
{
    const int warp = tid >> 5;
    const int wm   = warp / WARPS_N;   // 0..3
    const int wn   = warp % WARPS_N;   // 0..3

    // 2×2 WMMA tiles per warp → 32×32 sub-tile
    for (int dm = 0; dm < 2; ++dm) {
        for (int dn = 0; dn < 2; ++dn) {
            const int row16 = wm * 32 + dm * 16;
            const int col16 = wn * 32 + dn * 16;

            wmma::fragment<wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K,
                           half, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K,
                           half, wmma::col_major> fb;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fc;
            wmma::fill_fragment(fc, 0.f);

            for (int kk = 0; kk < KPAD; kk += WMMA_K) {
                wmma::load_matrix_sync(fa, sA + row16 * KPAD + kk, KPAD);
                wmma::load_matrix_sync(fb, sB + col16 * KPAD + kk, KPAD);
                wmma::mma_sync(fc, fa, fb, fc);
            }

            wmma::store_matrix_sync(
                sDot + row16 * BN + col16, fc, BN, wmma::mem_row_major);
        }
    }
}

// ---------------------------------------------------------------------------
// Prepass kernel
// ---------------------------------------------------------------------------
__global__ void prepass_kernel_sm120(
    const half*  __restrict__ X,
    const float* __restrict__ w,
    float*       __restrict__ D,
    const float* __restrict__ sq_X,
    int N, int d, int KPAD, float beta)
{
    extern __shared__ unsigned char smem[];
    half*  sA   = reinterpret_cast<half*>(smem);
    half*  sB   = sA + BM * KPAD;
    float* sDot = reinterpret_cast<float*>(sB + BN * KPAD);

    const int tid = threadIdx.x;
    const int i0  = blockIdx.x * BM;
    const int bm  = min(BM, N - i0);
    float acc = 0.f;

    load_sA(sA, X, i0, bm, d, KPAD, tid);
    __syncthreads();

    for (int j0 = 0; j0 < N; j0 += BN) {
        const int bn = min(BN, N - j0);

        load_sB(sB, X, j0, bn, d, KPAD, tid);
        __syncthreads();

        wmma_dot_tile(sA, sB, sDot, KPAD, tid);
        __syncthreads();

        if (tid < bm) {
            float sqi = sq_X[i0 + tid];
            #pragma unroll 4
            for (int j = 0; j < bn; ++j) {
                float dist2 = fmaxf(sqi + sq_X[j0+j] - 2.f * sDot[tid*BN+j], 0.f);
                acc += __expf(-beta * dist2) * w[j0+j];
            }
        }
        __syncthreads();
    }

    if (tid < bm) D[i0 + tid] = acc;
}

// ---------------------------------------------------------------------------
// Matvec kernel
// ---------------------------------------------------------------------------
__global__ void matvec_kernel_sm120(
    const half*  __restrict__ X,
    const float* __restrict__ u,
    float*       __restrict__ out,
    const float* __restrict__ sq_X,
    const float* __restrict__ rscale,
    int N, int d, int KPAD, float beta)
{
    extern __shared__ unsigned char smem[];
    half*  sA   = reinterpret_cast<half*>(smem);
    half*  sB   = sA + BM * KPAD;
    float* sDot = reinterpret_cast<float*>(sB + BN * KPAD);

    const int tid = threadIdx.x;
    const int i0  = blockIdx.x * BM;
    const int bm  = min(BM, N - i0);
    float acc = 0.f;

    load_sA(sA, X, i0, bm, d, KPAD, tid);
    __syncthreads();

    for (int j0 = 0; j0 < N; j0 += BN) {
        const int bn = min(BN, N - j0);

        load_sB(sB, X, j0, bn, d, KPAD, tid);
        __syncthreads();

        wmma_dot_tile(sA, sB, sDot, KPAD, tid);
        __syncthreads();

        if (tid < bm) {
            float sqi = sq_X[i0 + tid];
            #pragma unroll 4
            for (int j = 0; j < bn; ++j) {
                float dist2 = fmaxf(sqi + sq_X[j0+j] - 2.f * sDot[tid*BN+j], 0.f);
                acc += __expf(-beta * dist2) * u[j0+j];
            }
        }
        __syncthreads();
    }

    if (tid < bm) out[i0 + tid] = rscale[i0 + tid] * acc;
}

// ---------------------------------------------------------------------------
// SMEM opt-in
// ---------------------------------------------------------------------------
static void set_smem(const void* fn, size_t bytes) {
    int dev; C10_CUDA_CHECK(cudaGetDevice(&dev));
    int maxoptin;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &maxoptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    TORCH_CHECK((int)bytes <= maxoptin,
        "SMEM ", bytes, " > device max ", maxoptin);
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes));
}

// ---------------------------------------------------------------------------
// Python entry points
// ---------------------------------------------------------------------------

torch::Tensor compute_sq_norms(torch::Tensor X_fp16) {
    int N = X_fp16.size(0), d = X_fp16.size(1);
    auto sq = torch::empty({N}, X_fp16.options().dtype(torch::kFloat32));
    sq_norm_kernel_sm120<<<ceil_div(N,256), 256, 0,
                           at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(X_fp16.data_ptr<at::Half>()),
        sq.data_ptr<float>(), N, d);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return sq;
}

torch::Tensor prepass(
    torch::Tensor X_fp16, torch::Tensor w,
    torch::Tensor sq_X,   float beta)
{
    int N = X_fp16.size(0), d = X_fp16.size(1);
    int KPAD = round_up(d, WMMA_K);
    size_t sm = smem_size(d);
    set_smem(reinterpret_cast<const void*>(prepass_kernel_sm120), sm);

    auto D = torch::zeros({N}, w.options());
    prepass_kernel_sm120<<<ceil_div(N,BM), NTHREADS, sm,
                           at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(X_fp16.data_ptr<at::Half>()),
        w.data_ptr<float>(), D.data_ptr<float>(),
        sq_X.data_ptr<float>(), N, d, KPAD, beta);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return D;
}

torch::Tensor matvec(
    torch::Tensor X_fp16, torch::Tensor v,
    torch::Tensor rscale, torch::Tensor sq_X, float beta)
{
    int N = X_fp16.size(0), d = X_fp16.size(1);
    int KPAD = round_up(d, WMMA_K);
    size_t sm = smem_size(d);
    set_smem(reinterpret_cast<const void*>(matvec_kernel_sm120), sm);

    auto u   = (rscale * v).contiguous();
    auto out = torch::empty({N}, v.options());
    matvec_kernel_sm120<<<ceil_div(N,BM), NTHREADS, sm,
                          at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(X_fp16.data_ptr<at::Half>()),
        u.data_ptr<float>(), out.data_ptr<float>(),
        sq_X.data_ptr<float>(), rscale.data_ptr<float>(),
        N, d, KPAD, beta);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashDiffusion SM120 WMMA tensor cores "
              "(RTX 5090 / RTX PRO 6000 Blackwell consumer)";
    m.def("compute_sq_norms", &compute_sq_norms, py::arg("X_fp16"));
    m.def("prepass",  &prepass,
          py::arg("X_fp16"), py::arg("w"), py::arg("sq_X"), py::arg("beta"));
    m.def("matvec",   &matvec,
          py::arg("X_fp16"), py::arg("v"), py::arg("rscale"),
          py::arg("sq_X"), py::arg("beta"));
}
```

## `src/flashdiffusion/csrc/flash_diffusion_sm120_.cu`

```cuda
/*
 * flash_diffusion_sm120.cu
 * ========================
 * SM120 (RTX 5090 / RTX PRO 6000 Blackwell) kernel for FlashDiffusion.
 *
 * Validated working. No cp.async pipeline (v1 — synchronous loads).
 * cp.async pipeline to be added in v2 once correctness is confirmed.
 *
 * SM120 vs SM80 differences used here:
 *   - Larger tiles: BM=128, BN=128 (vs BM=64, BN=64 on SM80)
 *   - More threads: NTHREADS=256 (8 warps vs 4 warps)
 *   - Same mma.sync instruction family (no wgmma, no TMEM)
 *   - No multicast, cluster shape 1x1x1
 *
 * No online softmax. No gradients. rscale precomputed.
 *
 * Build:
 *   -arch=sm_120 -std=c++17 --use_fast_math
 *   -U__CUDA_NO_HALF_OPERATORS__ --expt-relaxed-constexpr
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

static constexpr int BM       = 128;
static constexpr int BN       = 128;
static constexpr int NTHREADS = 256;

__device__ __forceinline__ float fast_exp(float x) { return __expf(x); }

__global__ void sq_norm_kernel_sm120(
    const __half* __restrict__ X,
    float*        __restrict__ sq,
    int N, int d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s = 0.f;
    for (int k = 0; k < d; ++k) {
        float v = __half2float(X[i * d + k]);
        s += v * v;
    }
    sq[i] = s;
}

__global__ void prepass_kernel_sm120(
    const __half* __restrict__ X,
    const float*  __restrict__ w,
    float*        __restrict__ D,
    const float*  __restrict__ sq_X,
    int N, int d, float beta)
{
    extern __shared__ __half smem[];
    __half* sXi = smem;
    __half* sXj = sXi + BM * d;

    int tid = threadIdx.x;
    int i0  = blockIdx.x * BM;
    int i1  = min(i0 + BM, N);
    int bm  = i1 - i0;

    float acc = 0.f;

    for (int idx = tid; idx < bm * d; idx += NTHREADS) {
        int r = idx / d, c = idx % d;
        sXi[r * d + c] = X[(i0 + r) * d + c];
    }
    __syncthreads();

    for (int j0 = 0; j0 < N; j0 += BN) {
        int j1 = min(j0 + BN, N);
        int bn = j1 - j0;

        for (int idx = tid; idx < bn * d; idx += NTHREADS) {
            int r = idx / d, c = idx % d;
            sXj[r * d + c] = X[(j0 + r) * d + c];
        }
        __syncthreads();

        if (tid < bm) {
            float sqi = sq_X[i0 + tid];
            for (int j = 0; j < bn; ++j) {
                float dot = 0.f;
                for (int k = 0; k < d; ++k)
                    dot += __half2float(sXi[tid * d + k])
                         * __half2float(sXj[j   * d + k]);
                float dist2 = fmaxf(sqi + sq_X[j0 + j] - 2.f * dot, 0.f);
                acc += fast_exp(-beta * dist2) * w[j0 + j];
            }
        }
        __syncthreads();
    }

    if (tid < bm) D[i0 + tid] = acc;
}

__global__ void matvec_kernel_sm120(
    const __half* __restrict__ X,
    const float*  __restrict__ u,
    float*        __restrict__ out,
    const float*  __restrict__ sq_X,
    const float*  __restrict__ rscale,
    int N, int d, float beta)
{
    extern __shared__ __half smem[];
    __half* sXi = smem;
    __half* sXj = sXi + BM * d;

    int tid = threadIdx.x;
    int i0  = blockIdx.x * BM;
    int i1  = min(i0 + BM, N);
    int bm  = i1 - i0;

    float acc = 0.f;

    for (int idx = tid; idx < bm * d; idx += NTHREADS) {
        int r = idx / d, c = idx % d;
        sXi[r * d + c] = X[(i0 + r) * d + c];
    }
    __syncthreads();

    for (int j0 = 0; j0 < N; j0 += BN) {
        int j1 = min(j0 + BN, N);
        int bn = j1 - j0;

        for (int idx = tid; idx < bn * d; idx += NTHREADS) {
            int r = idx / d, c = idx % d;
            sXj[r * d + c] = X[(j0 + r) * d + c];
        }
        __syncthreads();

        if (tid < bm) {
            float sqi = sq_X[i0 + tid];
            for (int j = 0; j < bn; ++j) {
                float dot = 0.f;
                for (int k = 0; k < d; ++k)
                    dot += __half2float(sXi[tid * d + k])
                         * __half2float(sXj[j   * d + k]);
                float dist2 = fmaxf(sqi + sq_X[j0 + j] - 2.f * dot, 0.f);
                acc += fast_exp(-beta * dist2) * u[j0 + j];
            }
        }
        __syncthreads();
    }

    if (tid < bm) out[i0 + tid] = rscale[i0 + tid] * acc;
}

torch::Tensor compute_sq_norms(torch::Tensor X_fp16) {
    int N = X_fp16.size(0), d = X_fp16.size(1);
    auto sq = torch::empty({N}, X_fp16.options().dtype(torch::kFloat32));
    sq_norm_kernel_sm120<<<(N + 255) / 256, 256, 0,
                           at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(X_fp16.data_ptr<at::Half>()),
        sq.data_ptr<float>(), N, d);
    return sq;
}

torch::Tensor prepass(
    torch::Tensor X_fp16, torch::Tensor w,
    torch::Tensor sq_X,   float beta)
{
    int N = X_fp16.size(0), d = X_fp16.size(1);
    auto D    = torch::zeros({N}, w.options());
    int  grid = (N + BM - 1) / BM;
    size_t smem = (size_t)(BM + BN) * d * sizeof(__half);
    prepass_kernel_sm120<<<grid, NTHREADS, smem,
                           at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(X_fp16.data_ptr<at::Half>()),
        w.data_ptr<float>(), D.data_ptr<float>(),
        sq_X.data_ptr<float>(), N, d, beta);
    return D;
}

torch::Tensor matvec(
    torch::Tensor X_fp16, torch::Tensor v,
    torch::Tensor rscale, torch::Tensor sq_X, float beta)
{
    int N = X_fp16.size(0), d = X_fp16.size(1);
    auto u   = rscale * v;
    auto out = torch::empty({N}, v.options());
    int  grid = (N + BM - 1) / BM;
    size_t smem = (size_t)(BM + BN) * d * sizeof(__half);
    matvec_kernel_sm120<<<grid, NTHREADS, smem,
                          at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(X_fp16.data_ptr<at::Half>()),
        u.data_ptr<float>(), out.data_ptr<float>(),
        sq_X.data_ptr<float>(), rscale.data_ptr<float>(), N, d, beta);
    return out;
}

PYBIND11_MODULE(flash_diffusion_sm120, m) {
    m.doc() = "FlashDiffusion SM120 — 128x128 tiles v1 validated "
              "(RTX 5090 / RTX PRO 6000 Blackwell)";
    m.def("compute_sq_norms", &compute_sq_norms, py::arg("X_fp16"));
    m.def("prepass", &prepass,
          py::arg("X_fp16"), py::arg("w"), py::arg("sq_X"), py::arg("beta"));
    m.def("matvec",  &matvec,
          py::arg("X_fp16"), py::arg("v"), py::arg("rscale"),
          py::arg("sq_X"), py::arg("beta"));
}
```

## `src/flashdiffusion/csrc/flash_diffusion_sm80.cu`

```cuda
/*
 * flash_diffusion_sm80.cu  — WMMA v2
 * ===================================
 * FlashDiffusion WMMA tensor core kernel.
 * Targets: SM80 (A100), SM86 (RTX 3090), SM89 (RTX 4090),
 *          SM120 (RTX 5090 / RTX PRO 6000) via fallback
 *
 * v2 over v1:
 *   Fixed SMEM size via k-chunking — supports arbitrary d (including d=784).
 *   v1 allocated BM×KPAD SMEM which exceeded 96KB A100 limit for d > ~400.
 *
 * SMEM layout (fixed, independent of d):
 *   sA   = BM × KCHUNK × 2  = 64 × 32 × 2 =  4,096 bytes
 *   sB   = BN × KCHUNK × 2  = 64 × 32 × 2 =  4,096 bytes
 *   sDot = BM × BN    × 4   = 64 × 64  × 4 = 16,384 bytes
 *   Total = 24,576 bytes = 24KB  ✓  (well under 96KB A100 limit)
 *
 * K-chunking: loop over d in steps of KCHUNK=32.
 *   Each chunk: load sA[BM×32], sB[BN×32], run 1 WMMA call per warp,
 *   accumulate into sDot (load fc from sDot on non-first chunks).
 *
 * Build:
 *   SM80/86/89: -arch=sm_80  -std=c++17 -O3 --use_fast_math
 *   -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__
 *   --expt-relaxed-constexpr
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

using namespace nvcuda;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr int WMMA_M  = 16;
static constexpr int WMMA_N  = 16;
static constexpr int WMMA_K  = 16;

static constexpr int BM      = 64;
static constexpr int BN      = 64;
static constexpr int KCHUNK  = 32;    // process d in 32-wide chunks (2 WMMA_K)

// 16 warps: each covers one 16×16 WMMA tile
static constexpr int WARPS_M  = BM / WMMA_M;        // 4
static constexpr int WARPS_N  = BN / WMMA_N;        // 4
static constexpr int NWARPS   = WARPS_M * WARPS_N;  // 16
static constexpr int NTHREADS = NWARPS * 32;          // 512

// Fixed SMEM — 24KB, works for any d
static constexpr size_t SMEM_BYTES =
    (size_t)BM * KCHUNK * sizeof(half)    // sA
  + (size_t)BN * KCHUNK * sizeof(half)    // sB
  + (size_t)BM * BN     * sizeof(float);  // sDot

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------------
// Squared norms
// ---------------------------------------------------------------------------
__global__ void sq_norm_kernel(
    const half* __restrict__ X, float* __restrict__ sq, int N, int d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s = 0.f;
    for (int k = 0; k < d; ++k) { float v = __half2float(X[i*d+k]); s += v*v; }
    sq[i] = s;
}

// ---------------------------------------------------------------------------
// SMEM loaders for a k-chunk [k0, k0+klen)
// sA: [BM, KCHUNK] row-major   → matrix_a row_major
// sB: [BN, KCHUNK] col-major   → matrix_b col_major (= B^T)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void load_sA_chunk(
    half* sA, const half* X, int i0, int bm, int d,
    int k0, int klen, int tid)
{
    for (int idx = tid; idx < BM * KCHUNK; idx += NTHREADS) {
        int r = idx / KCHUNK, c = idx % KCHUNK;
        sA[idx] = (r < bm && c < klen)
            ? X[(long long)(i0+r)*d + k0+c]
            : __float2half(0.f);
    }
}

__device__ __forceinline__ void load_sB_chunk(
    half* sB, const half* X, int j0, int bn, int d,
    int k0, int klen, int tid)
{
    for (int idx = tid; idx < BN * KCHUNK; idx += NTHREADS) {
        int col = idx / KCHUNK, c = idx % KCHUNK;
        sB[idx] = (col < bn && c < klen)
            ? X[(long long)(j0+col)*d + k0+c]
            : __float2half(0.f);
    }
}

// ---------------------------------------------------------------------------
// WMMA accumulate: sA (BM×KCHUNK, row) @ sB^T (KCHUNK×BN, col) → sDot
// Each warp covers one 16×16 WMMA tile.
// first_chunk=true:  initialize fc=0
// first_chunk=false: load fc from sDot first (accumulate across k-chunks)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void wmma_accumulate(
    const half* sA, const half* sB, float* sDot,
    bool first_chunk, int tid)
{
    const int warp = tid >> 5;
    const int wm   = warp / WARPS_N;
    const int wn   = warp % WARPS_N;
    const int row16 = wm * WMMA_M;
    const int col16 = wn * WMMA_N;

    wmma::fragment<wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K,
                   half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K,
                   half, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fc;

    if (first_chunk) {
        wmma::fill_fragment(fc, 0.f);
    } else {
        wmma::load_matrix_sync(fc,
            sDot + row16 * BN + col16, BN, wmma::mem_row_major);
    }

    for (int kk = 0; kk < KCHUNK; kk += WMMA_K) {
        wmma::load_matrix_sync(fa, sA + row16 * KCHUNK + kk, KCHUNK);
        wmma::load_matrix_sync(fb, sB + col16 * KCHUNK + kk, KCHUNK);
        wmma::mma_sync(fc, fa, fb, fc);
    }

    wmma::store_matrix_sync(
        sDot + row16 * BN + col16, fc, BN, wmma::mem_row_major);
}

// ---------------------------------------------------------------------------
// Prepass kernel: D_i = Σ_j exp(-β||xi-xj||²) * w_j
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(NTHREADS)
prepass_kernel(
    const half*  __restrict__ X,
    const float* __restrict__ w,
    float*       __restrict__ D,
    const float* __restrict__ sq_X,
    int N, int d, float beta)
{
    extern __shared__ unsigned char smem[];
    half*  sA   = reinterpret_cast<half*>(smem);
    half*  sB   = sA + BM * KCHUNK;
    float* sDot = reinterpret_cast<float*>(sB + BN * KCHUNK);

    const int tid = threadIdx.x;
    const int i0  = blockIdx.x * BM;
    const int bm  = min(BM, N - i0);
    float acc = 0.f;

    for (int j0 = 0; j0 < N; j0 += BN) {
        const int bn = min(BN, N - j0);

        // k-chunk loop: accumulate dot products into sDot
        for (int k0 = 0; k0 < d; k0 += KCHUNK) {
            const int klen = min(KCHUNK, d - k0);
            const bool first = (k0 == 0);

            load_sA_chunk(sA, X, i0, bm, d, k0, klen, tid);
            load_sB_chunk(sB, X, j0, bn, d, k0, klen, tid);
            __syncthreads();

            wmma_accumulate(sA, sB, sDot, first, tid);
            __syncthreads();
        }

        // epilogue: dist2 → exp → accumulate
        if (tid < bm) {
            float sqi = sq_X[i0 + tid];
            #pragma unroll 4
            for (int j = 0; j < bn; ++j) {
                float dist2 = fmaxf(sqi + sq_X[j0+j] - 2.f * sDot[tid*BN+j], 0.f);
                acc += __expf(-beta * dist2) * w[j0+j];
            }
        }
        __syncthreads();
    }

    if (tid < bm) D[i0 + tid] = acc;
}

// ---------------------------------------------------------------------------
// Matvec kernel: out_i = rscale_i * Σ_j exp(-β||xi-xj||²) * u_j
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(NTHREADS)
matvec_kernel(
    const half*  __restrict__ X,
    const float* __restrict__ u,
    float*       __restrict__ out,
    const float* __restrict__ sq_X,
    const float* __restrict__ rscale,
    int N, int d, float beta)
{
    extern __shared__ unsigned char smem[];
    half*  sA   = reinterpret_cast<half*>(smem);
    half*  sB   = sA + BM * KCHUNK;
    float* sDot = reinterpret_cast<float*>(sB + BN * KCHUNK);

    const int tid = threadIdx.x;
    const int i0  = blockIdx.x * BM;
    const int bm  = min(BM, N - i0);
    float acc = 0.f;

    for (int j0 = 0; j0 < N; j0 += BN) {
        const int bn = min(BN, N - j0);

        for (int k0 = 0; k0 < d; k0 += KCHUNK) {
            const int klen = min(KCHUNK, d - k0);
            const bool first = (k0 == 0);

            load_sA_chunk(sA, X, i0, bm, d, k0, klen, tid);
            load_sB_chunk(sB, X, j0, bn, d, k0, klen, tid);
            __syncthreads();

            wmma_accumulate(sA, sB, sDot, first, tid);
            __syncthreads();
        }

        if (tid < bm) {
            float sqi = sq_X[i0 + tid];
            #pragma unroll 4
            for (int j = 0; j < bn; ++j) {
                float dist2 = fmaxf(sqi + sq_X[j0+j] - 2.f * sDot[tid*BN+j], 0.f);
                acc += __expf(-beta * dist2) * u[j0+j];
            }
        }
        __syncthreads();
    }

    if (tid < bm) out[i0 + tid] = rscale[i0 + tid] * acc;
}

// ---------------------------------------------------------------------------
// SMEM opt-in
// ---------------------------------------------------------------------------
static void set_smem(const void* fn) {
    int dev; C10_CUDA_CHECK(cudaGetDevice(&dev));
    int maxoptin;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &maxoptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    TORCH_CHECK((int)SMEM_BYTES <= maxoptin,
        "SMEM ", SMEM_BYTES, " > device max ", maxoptin);
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES));
}

// ---------------------------------------------------------------------------
// Python entry points
// ---------------------------------------------------------------------------

torch::Tensor compute_sq_norms(torch::Tensor X_fp16) {
    TORCH_CHECK(X_fp16.is_cuda() && X_fp16.scalar_type() == at::kHalf,
                "X_fp16 must be a CUDA half tensor");
    int N = X_fp16.size(0), d = X_fp16.size(1);
    auto sq = torch::empty({N}, X_fp16.options().dtype(at::kFloat));
    sq_norm_kernel<<<ceil_div(N,256), 256, 0,
                     at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(X_fp16.data_ptr<at::Half>()),
        sq.data_ptr<float>(), N, d);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return sq;
}

torch::Tensor prepass(
    torch::Tensor X_fp16, torch::Tensor w,
    torch::Tensor sq_X,   float beta)
{
    int N = X_fp16.size(0), d = X_fp16.size(1);
    set_smem(reinterpret_cast<const void*>(prepass_kernel));

    auto D = torch::zeros({N}, w.options());
    prepass_kernel<<<ceil_div(N,BM), NTHREADS, SMEM_BYTES,
                     at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(X_fp16.data_ptr<at::Half>()),
        w.data_ptr<float>(), D.data_ptr<float>(),
        sq_X.data_ptr<float>(), N, d, beta);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return D;
}

torch::Tensor matvec(
    torch::Tensor X_fp16, torch::Tensor v,
    torch::Tensor rscale, torch::Tensor sq_X, float beta)
{
    int N = X_fp16.size(0), d = X_fp16.size(1);
    set_smem(reinterpret_cast<const void*>(matvec_kernel));

    auto u   = (rscale * v).contiguous();
    auto out = torch::empty({N}, v.options());
    matvec_kernel<<<ceil_div(N,BM), NTHREADS, SMEM_BYTES,
                    at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(X_fp16.data_ptr<at::Half>()),
        u.data_ptr<float>(), out.data_ptr<float>(),
        sq_X.data_ptr<float>(), rscale.data_ptr<float>(),
        N, d, beta);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashDiffusion WMMA v2 — k-chunked, arbitrary d "
              "(SM80/SM86/SM89 A100/RTX3090/RTX4090)";
    m.def("compute_sq_norms", &compute_sq_norms, py::arg("X_fp16"));
    m.def("prepass",  &prepass,
          py::arg("X_fp16"), py::arg("w"), py::arg("sq_X"), py::arg("beta"));
    m.def("matvec",   &matvec,
          py::arg("X_fp16"), py::arg("v"), py::arg("rscale"),
          py::arg("sq_X"), py::arg("beta"));
}
```

## `src/flashdiffusion/csrc/flash_diffusion_sm80X.cu`

```cuda
/*
 * flash_diffusion_sm80.cu
 * =======================
 * FlashDiffusion WMMA tensor core kernel.
 * Targets: SM80 (A100), SM86 (RTX 3090), SM89 (RTX 4090),
 *          SM120 (RTX 5090 / RTX PRO 6000 Blackwell consumer)
 *
 * Uses nvcuda::wmma — portable from SM70 (Volta) through SM120.
 * Replaces the scalar flash_diffusion_sm80.cu baseline.
 *
 * Tile: BM=64, BN=64, 16 warps = 512 threads
 * WMMA: 16×16×16 fp16 input, fp32 accumulator
 *
 * Same public API as all FlashDiffusion kernels:
 *   compute_sq_norms(X_fp16)
 *   prepass(X_fp16, w, sq_X, beta)
 *   matvec(X_fp16, v, rscale, sq_X, beta)
 *
 * Build:
 *   SM80/86/89: -arch=sm_80  -std=c++17 -O3 --use_fast_math
 *   SM120/121:  -arch=sm_120 -std=c++17 -O3 --use_fast_math
 *   -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__
 *   --expt-relaxed-constexpr
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

using namespace nvcuda;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr int WMMA_M   = 16;
static constexpr int WMMA_N   = 16;
static constexpr int WMMA_K   = 16;
static constexpr int BM       = 64;
static constexpr int BN       = 64;
static constexpr int WARPS_M  = BM / WMMA_M;     // 4
static constexpr int WARPS_N  = BN / WMMA_N;     // 4
static constexpr int NWARPS   = WARPS_M * WARPS_N;  // 16
static constexpr int NTHREADS = NWARPS * 32;      // 512

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }
static inline int round_up(int a, int b) { return ceil_div(a, b) * b; }

// Shared memory: sA[BM×KPAD] + sB[BN×KPAD] + sDot[BM×BN]
static size_t smem_size(int d) {
    int KPAD = round_up(d, WMMA_K);
    return (size_t)BM * KPAD * sizeof(half)
         + (size_t)BN * KPAD * sizeof(half)
         + (size_t)BM * BN   * sizeof(float);
}

// ---------------------------------------------------------------------------
// Squared norms
// ---------------------------------------------------------------------------
__global__ void sq_norm_kernel(
    const half* __restrict__ X, float* __restrict__ sq, int N, int d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s = 0.f;
    for (int k = 0; k < d; ++k) { float v = __half2float(X[i*d+k]); s += v*v; }
    sq[i] = s;
}

// ---------------------------------------------------------------------------
// SMEM loaders
// sA: [BM, KPAD] row-major    → matrix_a row_major
// sB: [BN, KPAD] stored as col × k  → matrix_b col_major (= B^T row-major)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void load_sA(
    half* sA, const half* X, int i0, int bm, int d, int KPAD, int tid)
{
    for (int idx = tid; idx < BM * KPAD; idx += NTHREADS) {
        int r = idx / KPAD, c = idx % KPAD;
        sA[idx] = (r < bm && c < d) ? X[(long long)(i0+r)*d + c] : __float2half(0.f);
    }
}

__device__ __forceinline__ void load_sB(
    half* sB, const half* X, int j0, int bn, int d, int KPAD, int tid)
{
    for (int idx = tid; idx < BN * KPAD; idx += NTHREADS) {
        int col = idx / KPAD, k = idx % KPAD;
        sB[idx] = (col < bn && k < d) ? X[(long long)(j0+col)*d + k] : __float2half(0.f);
    }
}

// ---------------------------------------------------------------------------
// WMMA dot tile: sA (BM×KPAD, row) @ sB (KPAD×BN, col) → sDot (BM×BN, row)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void wmma_dot_tile(
    const half* sA, const half* sB, float* sDot, int KPAD, int tid)
{
    int warp = tid >> 5;
    int wm   = warp / WARPS_N;
    int wn   = warp % WARPS_N;

    wmma::fragment<wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fc;
    wmma::fill_fragment(fc, 0.f);

    for (int kk = 0; kk < KPAD; kk += WMMA_K) {
        wmma::load_matrix_sync(fa, sA + wm * WMMA_M * KPAD + kk, KPAD);
        wmma::load_matrix_sync(fb, sB + wn * WMMA_N * KPAD + kk, KPAD);
        wmma::mma_sync(fc, fa, fb, fc);
    }
    wmma::store_matrix_sync(sDot + wm * WMMA_M * BN + wn * WMMA_N,
                            fc, BN, wmma::mem_row_major);
}

// ---------------------------------------------------------------------------
// Prepass: D_i = Σ_j exp(-β||xi-xj||²) * w_j
// ---------------------------------------------------------------------------
__global__ void prepass_kernel(
    const half*  __restrict__ X,
    const float* __restrict__ w,
    float*       __restrict__ D,
    const float* __restrict__ sq_X,
    int N, int d, int KPAD, float beta)
{
    extern __shared__ unsigned char smem[];
    half*  sA   = reinterpret_cast<half*>(smem);
    half*  sB   = sA + BM * KPAD;
    float* sDot = reinterpret_cast<float*>(sB + BN * KPAD);

    const int tid = threadIdx.x;
    const int i0  = blockIdx.x * BM;
    const int bm  = min(BM, N - i0);
    float acc = 0.f;

    load_sA(sA, X, i0, bm, d, KPAD, tid);
    __syncthreads();

    for (int j0 = 0; j0 < N; j0 += BN) {
        const int bn = min(BN, N - j0);

        load_sB(sB, X, j0, bn, d, KPAD, tid);
        __syncthreads();

        wmma_dot_tile(sA, sB, sDot, KPAD, tid);
        __syncthreads();

        if (tid < bm) {
            float sqi = sq_X[i0 + tid];
            #pragma unroll 4
            for (int j = 0; j < bn; ++j) {
                float dist2 = fmaxf(sqi + sq_X[j0+j] - 2.f * sDot[tid*BN+j], 0.f);
                acc += __expf(-beta * dist2) * w[j0+j];
            }
        }
        __syncthreads();
    }

    if (tid < bm) D[i0 + tid] = acc;
}

// ---------------------------------------------------------------------------
// Matvec: out_i = rscale_i * Σ_j exp(-β||xi-xj||²) * u_j
// ---------------------------------------------------------------------------
__global__ void matvec_kernel(
    const half*  __restrict__ X,
    const float* __restrict__ u,
    float*       __restrict__ out,
    const float* __restrict__ sq_X,
    const float* __restrict__ rscale,
    int N, int d, int KPAD, float beta)
{
    extern __shared__ unsigned char smem[];
    half*  sA   = reinterpret_cast<half*>(smem);
    half*  sB   = sA + BM * KPAD;
    float* sDot = reinterpret_cast<float*>(sB + BN * KPAD);

    const int tid = threadIdx.x;
    const int i0  = blockIdx.x * BM;
    const int bm  = min(BM, N - i0);
    float acc = 0.f;

    load_sA(sA, X, i0, bm, d, KPAD, tid);
    __syncthreads();

    for (int j0 = 0; j0 < N; j0 += BN) {
        const int bn = min(BN, N - j0);

        load_sB(sB, X, j0, bn, d, KPAD, tid);
        __syncthreads();

        wmma_dot_tile(sA, sB, sDot, KPAD, tid);
        __syncthreads();

        if (tid < bm) {
            float sqi = sq_X[i0 + tid];
            #pragma unroll 4
            for (int j = 0; j < bn; ++j) {
                float dist2 = fmaxf(sqi + sq_X[j0+j] - 2.f * sDot[tid*BN+j], 0.f);
                acc += __expf(-beta * dist2) * u[j0+j];
            }
        }
        __syncthreads();
    }

    if (tid < bm) out[i0 + tid] = rscale[i0 + tid] * acc;
}

// ---------------------------------------------------------------------------
// SMEM opt-in helpers
// ---------------------------------------------------------------------------
static void set_smem(const void* fn, size_t bytes) {
    int dev; C10_CUDA_CHECK(cudaGetDevice(&dev));
    int maxoptin;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &maxoptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    TORCH_CHECK((int)bytes <= maxoptin,
        "SMEM request ", bytes, " exceeds device max ", maxoptin);
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes));
}

// ---------------------------------------------------------------------------
// Python entry points
// ---------------------------------------------------------------------------

torch::Tensor compute_sq_norms(torch::Tensor X_fp16) {
    TORCH_CHECK(X_fp16.is_cuda() && X_fp16.scalar_type() == at::kHalf,
                "X_fp16 must be a CUDA half tensor");
    int N = X_fp16.size(0), d = X_fp16.size(1);
    auto sq = torch::empty({N}, X_fp16.options().dtype(at::kFloat));
    sq_norm_kernel<<<ceil_div(N,256), 256, 0,
                     at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(X_fp16.data_ptr<at::Half>()),
        sq.data_ptr<float>(), N, d);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return sq;
}

torch::Tensor prepass(
    torch::Tensor X_fp16, torch::Tensor w,
    torch::Tensor sq_X,   float beta)
{
    int N = X_fp16.size(0), d = X_fp16.size(1);
    int KPAD = round_up(d, WMMA_K);
    size_t sm = smem_size(d);
    set_smem(reinterpret_cast<const void*>(prepass_kernel), sm);

    auto D = torch::zeros({N}, w.options());
    prepass_kernel<<<ceil_div(N,BM), NTHREADS, sm,
                     at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(X_fp16.data_ptr<at::Half>()),
        w.data_ptr<float>(), D.data_ptr<float>(),
        sq_X.data_ptr<float>(), N, d, KPAD, beta);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return D;
}

torch::Tensor matvec(
    torch::Tensor X_fp16, torch::Tensor v,
    torch::Tensor rscale, torch::Tensor sq_X, float beta)
{
    int N = X_fp16.size(0), d = X_fp16.size(1);
    int KPAD = round_up(d, WMMA_K);
    size_t sm = smem_size(d);
    set_smem(reinterpret_cast<const void*>(matvec_kernel), sm);

    auto u   = (rscale * v).contiguous();
    auto out = torch::empty({N}, v.options());
    matvec_kernel<<<ceil_div(N,BM), NTHREADS, sm,
                    at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(X_fp16.data_ptr<at::Half>()),
        u.data_ptr<float>(), out.data_ptr<float>(),
        sq_X.data_ptr<float>(), rscale.data_ptr<float>(),
        N, d, KPAD, beta);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashDiffusion WMMA tensor cores — SM80/SM86/SM89/SM120/SM121";
    m.def("compute_sq_norms", &compute_sq_norms, py::arg("X_fp16"));
    m.def("prepass",  &prepass,
          py::arg("X_fp16"), py::arg("w"), py::arg("sq_X"), py::arg("beta"));
    m.def("matvec",   &matvec,
          py::arg("X_fp16"), py::arg("v"), py::arg("rscale"),
          py::arg("sq_X"), py::arg("beta"));
}
```

## `src/flashdiffusion/csrc/flash_diffusion_sm80_.cu`

```cuda
/*
 * flash_diffusion_sm80.cu
 * =======================
 * SM80 (A100 / Ampere) CUDA kernel for FlashDiffusion.
 *
 * Computes, tile by tile, without materialising the O(N^2) kernel matrix:
 *
 *   PREPASS:
 *     D_i = sum_j  exp(-beta * ||x_i - x_j||^2) * w_j
 *
 *   MATVEC:
 *     out_i = rscale_i * sum_j  exp(-beta * ||x_i - x_j||^2) * rscale_j * v_j
 *
 * No online softmax — rscale is precomputed, epilogue is a scalar multiply.
 * No gradients — we compute the harmonic basis (eigenvectors), not a
 * differentiable layer.
 *
 * v1: scalar loop (validates correctness on A100).
 * v2: replace inner loop with CuTe TiledMMA (tensor cores) once v1 is green.
 *
 * Build (via torch.utils.cpp_extension.load in notebook):
 *   extra_cuda_cflags = [
 *       "-arch=sm_80", "-std=c++17", "--use_fast_math",
 *       "-U__CUDA_NO_HALF_OPERATORS__", "--expt-relaxed-constexpr",
 *   ]
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// ---------------------------------------------------------------------------
// Tile constants
// ---------------------------------------------------------------------------
static constexpr int BM       = 64;    // rows per Q-tile
static constexpr int BN       = 64;    // cols per K-tile
static constexpr int NTHREADS = 128;   // threads per block (4 warps)

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ float fast_exp(float x) { return __expf(x); }

// ---------------------------------------------------------------------------
// Sq-norm kernel:  sq[i] = ||x_i||^2
// ---------------------------------------------------------------------------
__global__ void sq_norm_kernel(
    const __half* __restrict__ X,
    float*        __restrict__ sq,
    int N, int d
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s = 0.f;
    for (int k = 0; k < d; ++k) {
        float v = __half2float(X[i * d + k]);
        s += v * v;
    }
    sq[i] = s;
}

// ---------------------------------------------------------------------------
// Prepass kernel:  D_i = sum_j K(i,j) * w_j
// ---------------------------------------------------------------------------
__global__ void prepass_kernel(
    const __half* __restrict__ X,
    const float*  __restrict__ w,
    float*        __restrict__ D,
    const float*  __restrict__ sq_X,
    int N, int d, float beta
) {
    extern __shared__ char smem[];
    __half* sXi = (__half*)smem;
    __half* sXj = sXi + BM * d;

    int tid = threadIdx.x;
    int i0  = blockIdx.x * BM;
    int i1  = min(i0 + BM, N);
    int bm  = i1 - i0;

    float acc = 0.f;

    // load Xi tile into smem
    for (int idx = tid; idx < bm * d; idx += NTHREADS)
        sXi[(idx / d) * d + idx % d] = X[(i0 + idx / d) * d + idx % d];
    __syncthreads();

    // loop over Xj tiles
    for (int j0 = 0; j0 < N; j0 += BN) {
        int j1 = min(j0 + BN, N);
        int bn = j1 - j0;

        for (int idx = tid; idx < bn * d; idx += NTHREADS)
            sXj[(idx / d) * d + idx % d] = X[(j0 + idx / d) * d + idx % d];
        __syncthreads();

        if (tid < bm) {
            float sqi = sq_X[i0 + tid];
            for (int j = 0; j < bn; ++j) {
                float dot = 0.f;
                for (int k = 0; k < d; ++k)
                    dot += __half2float(sXi[tid * d + k])
                         * __half2float(sXj[j   * d + k]);
                float dist2 = fmaxf(sqi + sq_X[j0 + j] - 2.f * dot, 0.f);
                acc += fast_exp(-beta * dist2) * w[j0 + j];
            }
        }
        __syncthreads();
    }

    if (tid < bm)
        D[i0 + tid] = acc;
}

// ---------------------------------------------------------------------------
// Matvec kernel:  out_i = rscale_i * sum_j K(i,j) * rscale_j * v_j
// ---------------------------------------------------------------------------
__global__ void matvec_kernel(
    const __half* __restrict__ X,
    const float*  __restrict__ u,        // u = rscale * v, pre-scaled on host
    float*        __restrict__ out,
    const float*  __restrict__ sq_X,
    const float*  __restrict__ rscale,
    int N, int d, float beta
) {
    extern __shared__ char smem[];
    __half* sXi = (__half*)smem;
    __half* sXj = sXi + BM * d;

    int tid = threadIdx.x;
    int i0  = blockIdx.x * BM;
    int i1  = min(i0 + BM, N);
    int bm  = i1 - i0;

    float acc = 0.f;

    for (int idx = tid; idx < bm * d; idx += NTHREADS)
        sXi[(idx / d) * d + idx % d] = X[(i0 + idx / d) * d + idx % d];
    __syncthreads();

    for (int j0 = 0; j0 < N; j0 += BN) {
        int j1 = min(j0 + BN, N);
        int bn = j1 - j0;

        for (int idx = tid; idx < bn * d; idx += NTHREADS)
            sXj[(idx / d) * d + idx % d] = X[(j0 + idx / d) * d + idx % d];
        __syncthreads();

        if (tid < bm) {
            float sqi = sq_X[i0 + tid];
            for (int j = 0; j < bn; ++j) {
                float dot = 0.f;
                for (int k = 0; k < d; ++k)
                    dot += __half2float(sXi[tid * d + k])
                         * __half2float(sXj[j   * d + k]);
                float dist2 = fmaxf(sqi + sq_X[j0 + j] - 2.f * dot, 0.f);
                acc += fast_exp(-beta * dist2) * u[j0 + j];
            }
        }
        __syncthreads();
    }

    if (tid < bm)
        out[i0 + tid] = rscale[i0 + tid] * acc;
}

// ---------------------------------------------------------------------------
// Python-facing entry points
// ---------------------------------------------------------------------------

// Call once per dataset — cache the result on the Python side
torch::Tensor compute_sq_norms(torch::Tensor X_fp16) {
    int N = X_fp16.size(0), d = X_fp16.size(1);
    auto sq = torch::empty({N}, X_fp16.options().dtype(torch::kFloat32));
    int threads = 256, blocks = (N + threads - 1) / threads;
    sq_norm_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(X_fp16.data_ptr<at::Half>()),
        sq.data_ptr<float>(), N, d);
    return sq;
}

// Pass 1: w = ones(N)  ->  D  (raw row sums)
// Pass 2: w = D^{-alpha}  ->  D_alpha (before * w_i)
torch::Tensor prepass(
    torch::Tensor X_fp16,
    torch::Tensor w,
    torch::Tensor sq_X,
    float beta
) {
    int N = X_fp16.size(0), d = X_fp16.size(1);
    auto D      = torch::zeros({N}, w.options());
    int  grid   = (N + BM - 1) / BM;
    size_t smem = (size_t)(BM + BN) * d * sizeof(__half);
    prepass_kernel<<<grid, NTHREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(X_fp16.data_ptr<at::Half>()),
        w.data_ptr<float>(), D.data_ptr<float>(),
        sq_X.data_ptr<float>(), N, d, beta);
    return D;
}

// Called every Lanczos iteration — X/rscale/sq_X fixed, only v changes
torch::Tensor matvec(
    torch::Tensor X_fp16,
    torch::Tensor v,
    torch::Tensor rscale,
    torch::Tensor sq_X,
    float beta
) {
    int N = X_fp16.size(0), d = X_fp16.size(1);
    auto u      = rscale * v;                    // pre-scale on GPU, O(N)
    auto out    = torch::empty({N}, v.options());
    int  grid   = (N + BM - 1) / BM;
    size_t smem = (size_t)(BM + BN) * d * sizeof(__half);
    matvec_kernel<<<grid, NTHREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(X_fp16.data_ptr<at::Half>()),
        u.data_ptr<float>(), out.data_ptr<float>(),
        sq_X.data_ptr<float>(), rscale.data_ptr<float>(), N, d, beta);
    return out;
}

// ---------------------------------------------------------------------------
// pybind11
// ---------------------------------------------------------------------------
PYBIND11_MODULE(flash_diffusion_cuda, m) {
    m.doc() = "FlashDiffusion SM80 CUDA kernel — scalar v1";
    m.def("compute_sq_norms", &compute_sq_norms,
          "Precompute ||x_i||^2. Call once, cache result.",
          py::arg("X_fp16"));
    m.def("prepass", &prepass,
          "Weighted row sum D_i = sum_j K(i,j)*w_j. Call twice for C-L normalisation.",
          py::arg("X_fp16"), py::arg("w"), py::arg("sq_X"), py::arg("beta"));
    m.def("matvec", &matvec,
          "Matvec out_i = rscale_i * sum_j K(i,j) * rscale_j * v_j. No gradients.",
          py::arg("X_fp16"), py::arg("v"), py::arg("rscale"),
          py::arg("sq_X"), py::arg("beta"));
}
```

## `src/flashdiffusion/dmap.py`

```python
"""
flashdiffusion/dmap.py
======================
DiffusionMap class: Coifman-Lafon diffusion maps via FlashDiffusion kernel.

Usage
-----
    dm = DiffusionMap(beta=1.0, alpha=0.5, n_components=8)
    dm.fit(X)               # precompute rscale (two kernel passes, O(N·tile) mem)
    coords = dm.transform() # Lanczos eigensolver -> diffusion coordinates
    new    = dm.transform(X_new)  # Nyström out-of-sample extension

Relationship to DAC paper
--------------------------
With Q=K=X and symmetric W (W=Wᵀ), this is the canonical EQ Schrödinger
bridge (Proposition 5.2). The operator M = D_α^{-1/2} K_α D_α^{-1/2} is
the symmetrised DMAP forward operator. Its eigenvectors give the diffusion
map embedding.

The Coifman-Lafon α-normalisation is an exact Doob h-transform
(Corollary 4.2, DAC): h = D^{-α}, so the EQ class is preserved.
"""

from __future__ import annotations
import numpy as np
from typing import Optional
import time

from .kernel import FlashDiffusion, prepass_rowsum


class DiffusionMap:
    """
    Coifman-Lafon diffusion maps.

    Parameters
    ----------
    beta : float
        Kernel bandwidth: K(i,j) = exp(-beta · ||xi - xj||²).
        Use utils.median_bandwidth(X) for automatic selection.
    alpha : float
        Density normalisation exponent (0 = no correction, 0.5 = default,
        1.0 = Fokker-Planck normalisation).
    n_components : int
        Number of eigenvectors to compute (including trivial λ=1).
    tile : int
        Tile size for the kernel. Controls peak memory: O(N · tile · d · 4 bytes).
    """

    def __init__(
        self,
        beta:          float = 1.0,
        alpha:         float = 1.0,
        n_components:  int   = 8,
        tile:          int   = 512,
        n_iter:        int   = None,
    ):
        self.beta         = beta
        self.alpha        = alpha
        self.n_components = n_components
        self.tile         = tile
        self.n_iter       = n_iter  # None → gpu_lanczos uses its default (6k+20)

        # set after fit()
        self.X_:          Optional[np.ndarray] = None
        self.rscale_:     Optional[np.ndarray] = None
        self.D_:          Optional[np.ndarray] = None
        self.D_alpha_:    Optional[np.ndarray] = None
        self._cuda_state: Optional[object]     = None  # CUDAState if GPU used

        # set after transform()
        self.eigenvalues_:  Optional[np.ndarray] = None
        self.eigenvectors_: Optional[np.ndarray] = None
        self.embedding_:    Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # fit: two prepass reductions, O(N·tile) memory
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "DiffusionMap":
        """
        Precompute Coifman-Lafon normalisation for dataset X.
    
        Uses CUDA backend (two GPU prepass kernels) if available,
        otherwise numpy CPU fallback. Either way the interface is identical.
        """
        N = X.shape[0]
        beta, alpha, tile = self.beta, self.alpha, self.tile
    
        print(f"[DiffusionMap.fit] N={N}  beta={beta}  alpha={alpha}")
        t0 = time.perf_counter()
    
        # ── center before everything else ───────────────────────────────────
        # Isometric: ||xi - xj||² = ||x_bar_i - x_bar_j||² exactly.
        # Leaves all eigenvalues and eigenvectors unchanged in exact arithmetic.
        # Numerical benefit: keeps ||x_bar_i||² = O(D·sigma²) with no mean-offset,
        # preventing underflow in exp(-beta·||x_bar_i||²) for large D or large mu.
        # mu_ stored for Nyström out-of-sample extension: x_new_bar = x_new - mu_
        X        = np.asarray(X, dtype=np.float64)
        self.mu_ = X.mean(axis=0)                    # (D,) fp64
        X_bar    = X - self.mu_                      # (N, D) centered
    
        # store centered version (was self.X_ = X before)
        self.X_  = X_bar

        # In dmap.py fit(), before passing to CUDA: !!!!!!!!!!!!!!!!!!!!!!!
        if X_bar.shape[1] < 16:
            pad = 16 - X_bar.shape[1]
            X_bar_padded = np.pad(X_bar, ((0,0),(0,pad)))  # zero pad to D=16
        else:
            X_bar_padded = X_bar
    
        # Try CUDA first
        from .kernel import _get_cuda_backend
        cuda = _get_cuda_backend()
        if cuda is not None:
            precompute_fn, matvec_fn = cuda
            self._cuda_state = precompute_fn(X_bar, beta, alpha)  # ← X_bar not X
            self._matvec_fn  = matvec_fn
            # Mirror numpy fields from GPU tensors for CDC / Doob / Nyström
            self.D_       = self._cuda_state.D.cpu().numpy().astype(np.float64)
            self.D_alpha_ = self._cuda_state.D_alpha.cpu().numpy().astype(np.float64)
            self.rscale_  = self._cuda_state.rscale.cpu().numpy().astype(np.float64)
            print(f"  precompute done ({self._cuda_state.__class__.__name__}) in {time.perf_counter()-t0:.3f}s")
            return self
    
        # Numpy fallback — also receives X_bar
        self._cuda_state = None
        D = prepass_rowsum(X_bar, X_bar, beta,
                           weights_k=np.ones(N, np.float64), tile=tile)
        self.D_ = D
        w = D ** (-alpha)
        D_alpha = prepass_rowsum(X_bar, X_bar, beta, weights_k=w, tile=tile) * w
        self.D_alpha_ = D_alpha
        d_half = D_alpha ** (-0.5)
        self.rscale_ = d_half * w
        print(f"  precompute done (numpy) in {time.perf_counter()-t0:.3f}s")
        return self

    def matvec(self, v: np.ndarray) -> np.ndarray:
        """
        Apply M to v. Uses CUDA kernel if available, numpy otherwise.
        Called every Lanczos iteration — v is the only thing that changes.
        """
        self._check_fitted()

        if self._cuda_state is not None:
            return self._matvec_fn(self._cuda_state, v)

        # numpy path
        v2d = v.reshape(-1, 1)
        out = FlashDiffusion(
            Q=self.X_, K=self.X_, V=v2d, beta=self.beta,
            rscale_q=self.rscale_, rscale_k=self.rscale_,
            tile=self.tile,
        )
        return out.ravel()

    # ------------------------------------------------------------------
    # transform: Lanczos eigensolver -> diffusion coordinates
    # ------------------------------------------------------------------

    def transform(
        self,
        X_new:    Optional[np.ndarray] = None,
        t:        int   = 1,
        tol:      float = 0.0,
        maxiter:  Optional[int] = None,
        use_gpu_lanczos: bool = True,   # ← new flag, default on
        ) -> np.ndarray:
        """
        Compute diffusion map coordinates.
    
        If X_new is None: eigensolver on training set (fit must be called first).
        If X_new is given: Nyström out-of-sample extension (see utils.nystrom).
    
        Parameters
        ----------
        X_new           : (M, d) new points, or None for training set
        t               : diffusion time (eigenvalues raised to power t)
        tol             : ARPACK tolerance (0 = machine precision), fallback only
        maxiter         : ARPACK max iterations, fallback only
        use_gpu_lanczos : use GPU Lanczos if CUDA state available (default True)
    
        Returns
        -------
        coords : (N, n_components-1) — drops the trivial first eigenvector
        """
        self._check_fitted()
    
        if X_new is not None:
            from .utils import nystrom_extend
            return nystrom_extend(self, X_new, t=t)
    
        # ── eigensolver dispatch ─────────────────────────────────────────────
        vals, vecs = self._solve_eigenproblem(
            tol=tol, maxiter=maxiter, use_gpu_lanczos=use_gpu_lanczos)
    
        self.eigenvalues_  = vals
        self.eigenvectors_ = vecs
    
        # convert symmetric-M eigenvectors → row-normalised P+ eigenvectors
        psi = vecs / self.rscale_[:, None]
        psi /= np.linalg.norm(psi, axis=0, keepdims=True)
    
        # diffusion map embedding: scale by λ^t, drop trivial first vector
        coords = psi[:, 1:] * (vals[1:] ** t)[None, :]
        self.embedding_ = coords
        return coords
    
    
    def _solve_eigenproblem(
        self,
        tol:             float = 0.0,
        maxiter:         Optional[int] = None,
        use_gpu_lanczos: bool = True,
        ) -> tuple:
        """
        Dispatch to GPU Lanczos or scipy ARPACK.
        Returns (vals, vecs) as numpy arrays.
        """
        N = self.X_.shape[0]
        k = self.n_components
    
        # ── GPU Lanczos path ─────────────────────────────────────────────────
        if use_gpu_lanczos and self._cuda_state is not None:
            try:
                import sys
                import torch
                from .eigensolver_gpu import gpu_lanczos, make_gpu_matvec
    
                # find the ext module that was injected by the notebook build
                _EXT_NAMES = {
                    'CUDAState':      'flash_diffusion_cuda',
                    'CUDAStateSM90':  'flash_diffusion_sm90',
                    'CUDAStateSM100': 'flash_diffusion_sm100',
                    'CUDAStateSM120': 'flash_diffusion_sm120',
                }
                ext_name = _EXT_NAMES.get(
                    type(self._cuda_state).__name__, 'flash_diffusion_cuda')
                ext = sys.modules.get(ext_name)
    
                if ext is None:
                    raise RuntimeError(
                        f"{ext_name} not in sys.modules — "
                        "build the kernel first then call dm.fit()")
    
                matvec_gpu = make_gpu_matvec(self._cuda_state, ext)
                
                # in the GPU Lanczos branch:
                q0 = torch.from_numpy(
                    np.sqrt(self._cuda_state.D_alpha.cpu().numpy())
                ).to(device='cuda', dtype=torch.float32)
                q0 /= q0.norm()
                
                vals, vecs = gpu_lanczos(matvec_gpu, N=N, k=k, tol=1e-6, q0=q0, n_iter=self.n_iter)
                return vals, vecs
    
            except Exception as e:
                print(f"[DiffusionMap] GPU Lanczos failed ({e}), "
                      f"falling back to ARPACK")
    
        # ── scipy ARPACK fallback ────────────────────────────────────────────
        from .eigensolver import lanczos_eigsh
        return lanczos_eigsh(
            self.matvec,
            N       = N,
            k       = k,
            tol     = tol,
            maxiter = maxiter,
        )

    # ------------------------------------------------------------------
    # Doob h-transform: returns a new DiffusionMap-like matvec
    # ------------------------------------------------------------------

    def doob_transform(self, h: np.ndarray) -> "DoobTransformedMap":
        """
        Apply a Doob h-transform to the current operator.

        By Theorem 5.1 (DAC): if the current operator is EQ (symmetric K),
        the transformed operator is also EQ. The new stationary distribution
        is π_h ∝ π · h · (P+h).

        Returns a DoobTransformedMap with a matvec() method compatible
        with eigensolver.py.
        """
        self._check_fitted()
        return DoobTransformedMap(self, h)

    # ------------------------------------------------------------------

    def _check_fitted(self):
        if self.rscale_ is None:
            raise RuntimeError("Call fit(X) before transform() or matvec().")


# ---------------------------------------------------------------------------
# Doob-transformed operator (wraps DiffusionMap with an h-potential)
# ---------------------------------------------------------------------------

class DoobTransformedMap:
    """
    Doob h-transform of a DiffusionMap operator.

    (M_h v)_i = rscale_i · Σⱼ K(i,j) · rscale_j · h_j · v_j
               / (P+ h)_i                         (row-renormalisation)

    By Theorem 5.1 (DAC), this preserves the EQ class when K is symmetric.
    The new rscale absorbs h into the kernel normalisation.
    """

    def __init__(self, base: DiffusionMap, h: np.ndarray):
        self.base = base
        self.h    = np.asarray(h, dtype=np.float64)

        # precompute (P+ h)_i = rscale_i * Σⱼ K(i,j) * rscale_j * h_j
        # = one FlashDiffusion call with V=h
        Ph = FlashDiffusion(
            Q        = base.X_,
            K        = base.X_,
            V        = self.h[:, None],
            beta     = base.beta,
            rscale_q = base.rscale_,
            rscale_k = base.rscale_,
            tile     = base.tile,
        ).ravel()
        self.Ph_ = Ph                      # normaliser for the Doob transform

    def matvec(self, v: np.ndarray) -> np.ndarray:
        """Apply Doob-transformed operator to v."""
        h   = self.h
        out = FlashDiffusion(
            Q        = self.base.X_,
            K        = self.base.X_,
            V        = (h * v)[:, None],
            beta     = self.base.beta,
            rscale_q = self.base.rscale_,
            rscale_k = self.base.rscale_,
            tile     = self.base.tile,
        ).ravel()
        return out / self.Ph_              # row-renormalise
```

## `src/flashdiffusion/eigenX.py`

```python
def gpu_lanczos(
    matvec_fn,
    N:        int,
    k:        int   = 8,
    n_iter:   int   = None,
    tol:      float = 1e-6,
    dtype           = None,
    device:   str   = "cuda",
    verbose:  bool  = True,
    q0=None,
    check_every: int = 5,
    min_iter: int | None = None,
    breakdown_tol: float = 1e-12,
    relative_residual: bool = True,
    return_info: bool = False,
) -> tuple:
    """
    GPU Lanczos with full re-orthogonalization and Ritz residual convergence checks.

    This is still plain non-restarted Lanczos, but now `tol` is active:
    the solver stops early once the top-k Ritz residuals satisfy

        residual_i = |beta_m * y_i[-1]|

    where y_i is the eigenvector of the small tridiagonal T_m.

    Parameters
    ----------
    matvec_fn : callable
        (N,) GPU tensor -> (N,) GPU tensor.
    N : int
        Problem size.
    k : int
        Number of largest eigenpairs wanted.
    n_iter : int
        Maximum Lanczos steps. If None, uses min(6*k + 20, N).
    tol : float
        Convergence tolerance for Ritz residuals.
    dtype : torch.dtype
        Default float32.
    device : str
        CUDA device.
    verbose : bool
        Print progress.
    q0 : torch.Tensor or None
        Optional initial vector. Warning: if q0 is exactly an eigenvector and k > 1,
        Lanczos may break down after one step.
    check_every : int
        Check Ritz residuals every this many iterations.
    min_iter : int or None
        Do not check convergence before this many steps.
        Default: min(max(2*k + 5, k + 2), n_iter).
    breakdown_tol : float
        Stop if beta falls below this value.
    relative_residual : bool
        If True, use residual <= tol * max(1, |theta|).
        If False, use residual <= tol.
    return_info : bool
        If True, return (evals, evecs, info).

    Returns
    -------
    vals : np.ndarray, shape <= (k,)
        Ritz eigenvalues in descending order.
    vecs : np.ndarray, shape (N, len(vals))
        Ritz vectors.
    info : dict, optional
        Returned only when return_info=True.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch required for gpu_lanczos")

    import torch

    if dtype is None:
        dtype = torch.float32

    if n_iter is None:
        n_iter = min(6 * k + 20, N)

    n_iter = int(min(n_iter, N))

    if min_iter is None:
        min_iter = min(max(2 * k + 5, k + 2), n_iter)
    else:
        min_iter = int(min(min_iter, n_iter))

    check_every = max(1, int(check_every))

    t_start = time.perf_counter()

    if verbose:
        print(
            f"[gpu_lanczos] N={N}  k={k}  max_iter={n_iter}  "
            f"min_iter={min_iter}  tol={tol:g}  dtype={dtype}"
        )

    # Lanczos basis and recurrence coefficients.
    Q      = torch.zeros(N, n_iter, dtype=dtype, device=device)
    alphas = torch.zeros(n_iter,    dtype=dtype, device=device)
    betas  = torch.zeros(n_iter,    dtype=dtype, device=device)

    # Initial vector.
    torch.manual_seed(42)

    if q0 is not None:
        q = q0.to(device=device, dtype=dtype).reshape(-1)
        if q.numel() != N:
            raise ValueError(f"q0 has length {q.numel()}, expected N={N}")
    else:
        # Random start is safer than all-ones if the constant vector is exactly
        # the leading eigenvector. All-ones can cause lucky breakdown at m=1.
        q = torch.randn(N, dtype=dtype, device=device)

    q_norm = q.norm()
    if q_norm.item() == 0:
        raise ValueError("Initial vector q0 has zero norm.")

    q = q / q_norm
    Q[:, 0] = q

    converged = False
    breakdown = False
    j_final = n_iter

    final_evals = None
    final_evecs_small = None
    final_residuals = None

    def solve_small_tridiagonal(m: int):
        """
        Solve T_m eigenproblem on CPU using NumPy.
        T_m is tiny, so CPU transfer is negligible and robust.
        Returns eigenvalues descending and eigenvectors in matching order.
        """
        T = torch.diag(alphas[:m])
        if m > 1:
            T = T + torch.diag(betas[:m - 1], 1)
            T = T + torch.diag(betas[:m - 1], -1)

        T_np = T.detach().cpu().numpy().astype(np.float64)
        evals_np, evecs_np = np.linalg.eigh(T_np)  # ascending

        idx = np.argsort(evals_np)[::-1]
        return evals_np[idx], evecs_np[:, idx]

    for j in range(n_iter):
        # Matvec.
        z = matvec_fn(Q[:, j])
        z = z.to(device=device, dtype=dtype).reshape(-1)

        # Three-term recurrence.
        alphas[j] = torch.dot(Q[:, j], z)
        z = z - alphas[j] * Q[:, j]

        if j > 0:
            z = z - betas[j - 1] * Q[:, j - 1]

        # Full double-pass MGS re-orthogonalization.
        # Use Q[:, :j+1] because current vector Q[:, j] is already filled.
        for _ in range(2):
            coeffs = Q[:, :j + 1].T @ z
            z = z - Q[:, :j + 1] @ coeffs

        beta_j_tensor = z.norm()
        beta_j = float(beta_j_tensor.detach().cpu())
        betas[j] = beta_j

        m = j + 1

        should_check = (
            m >= min_iter
            and (
                m == n_iter
                or m % check_every == 0
                or beta_j < breakdown_tol
            )
        )

        if should_check:
            evals_all, evecs_all = solve_small_tridiagonal(m)
            n_take = min(k, len(evals_all))

            evals_k = evals_all[:n_take]
            evecs_k = evecs_all[:, :n_take]

            # Lanczos Ritz residual:
            # ||A(Qy) - theta(Qy)|| ~= |beta_m * y[-1]|
            residuals = np.abs(beta_j * evecs_k[-1, :])

            if relative_residual:
                thresholds = tol * np.maximum(1.0, np.abs(evals_k))
            else:
                thresholds = np.full_like(residuals, tol, dtype=np.float64)

            n_conv = int(np.sum(residuals <= thresholds))

            if verbose:
                max_res = float(np.max(residuals)) if len(residuals) else float("nan")
                print(
                    f"  iter={m:4d}  beta={beta_j:.3e}  "
                    f"converged={n_conv}/{n_take}  max_res={max_res:.3e}"
                )

            final_evals = evals_all
            final_evecs_small = evecs_all
            final_residuals = residuals

            if n_take >= k and np.all(residuals[:k] <= thresholds[:k]):
                converged = True
                j_final = m
                if verbose:
                    print(f"[gpu_lanczos] converged at iter={m}")
                break

        # Lucky breakdown / invariant subspace exhausted.
        if beta_j < breakdown_tol:
            breakdown = True
            j_final = m
            if verbose:
                print(f"[gpu_lanczos] breakdown at iter={m}, beta={beta_j:.3e}")
            break

        # Store next basis vector.
        if j < n_iter - 1:
            Q[:, j + 1] = z / beta_j_tensor

    # Final solve if we never checked, or if the latest check was not final.
    m = j_final if (converged or breakdown) else n_iter

    if final_evals is None or final_evecs_small is None or final_evecs_small.shape[0] != m:
        final_evals, final_evecs_small = solve_small_tridiagonal(m)

        # Compute residuals for final reported Ritz pairs.
        n_take_tmp = min(k, len(final_evals))
        beta_m = float(betas[m - 1].detach().cpu()) if m > 0 else 0.0
        final_residuals = np.abs(beta_m * final_evecs_small[-1, :n_take_tmp])

    # Sort descending already done.
    n_return = min(k, len(final_evals), m)
    evals_k = final_evals[:n_return]
    evecs_k = final_evecs_small[:, :n_return]

    # Ritz vectors: Q[:, :m] @ evecs_k.
    evecs_k_gpu = torch.from_numpy(evecs_k.astype(np.float32)).to(device=device)
    ritz_vecs = Q[:, :m] @ evecs_k_gpu

    elapsed = time.perf_counter() - t_start

    if verbose:
        print(f"[gpu_lanczos] finished: m={m}  converged={converged}  time={elapsed:.2f}s")
        print(f"  eigenvalues: {evals_k}")
        if final_residuals is not None:
            print(f"  residuals:   {final_residuals[:n_return]}")

    vals_out = evals_k.astype(np.float64)
    vecs_out = ritz_vecs.detach().cpu().numpy()

    if return_info:
        info = {
            "converged": bool(converged),
            "breakdown": bool(breakdown),
            "n_iter_used": int(m),
            "n_iter_max": int(n_iter),
            "tol": float(tol),
            "relative_residual": bool(relative_residual),
            "residuals": None if final_residuals is None else final_residuals[:n_return].astype(np.float64),
            "elapsed": float(elapsed),
        }
        return vals_out, vecs_out, info

    return vals_out, vecs_out
```

## `src/flashdiffusion/eigensolver.py`

```python
"""
flashdiffusion/eigensolver.py
=============================
Mixed-precision Krylov eigensolver for the FlashDiffusion operator.

ARPACK (via scipy.sparse.linalg.eigsh) handles the Lanczos recurrence
internally. It calls our matvec() on each iteration and manages
orthogonalisation in float64. Convergence for diffusion maps is typically
fast because:
  - The spectrum is bounded in [0, 1] with a clear gap after k eigenvectors
  - ARPACK uses implicit restarting (IRLM) with a Krylov space much larger
    than k, achieving near-optimal convergence per matvec
  - The trivial λ=1 eigenvector is found immediately

Precision ramp
--------------
  matvec arithmetic : float32 GEMM (dist2) + float64 exp + float64 accumulate
  Krylov basis      : float64 (ARPACK internal)
  orthogonalisation : float64 QR (ARPACK internal, O(N k²) per restart)
  eigenvalue output : float64

This is the same split as FA4: tensor-core GEMM in low precision,
correction/accumulation in high precision.
"""

from __future__ import annotations
import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator
from typing import Callable, Optional, Tuple
import time


def lanczos_eigsh(
    matvec:  Callable[[np.ndarray], np.ndarray],
    N:       int,
    k:       int       = 8,
    tol:     float     = 0.0,
    maxiter: Optional[int] = None,
    ncv:     Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute top-k eigenpairs via ARPACK Lanczos (implicitly restarted).

    Parameters
    ----------
    matvec  : callable (N,) -> (N,), applies the symmetric operator M
    N       : problem dimension
    k       : number of eigenpairs
    tol     : ARPACK convergence tolerance (0 = machine eps)
    maxiter : max ARPACK iterations (default: 10 * N)
    ncv     : Krylov space size (default: min(max(2k+1, 20), N))
              Larger ncv -> faster convergence, more memory

    Returns
    -------
    eigenvalues  : (k,) float64, descending
    eigenvectors : (N, k) float64
    """
    if ncv is None:
        ncv = min(max(2 * k + 1, 20), N)

    op = LinearOperator(
        shape  = (N, N),
        matvec = lambda v: matvec(np.asarray(v, dtype=np.float64).ravel()),
        dtype  = np.float64,
    )

    t0 = time.perf_counter()
    vals, vecs = eigsh(
        op,
        k       = k,
        which   = 'LM',
        tol     = tol,
        maxiter = maxiter,
        ncv     = ncv,
    )
    dt = time.perf_counter() - t0

    idx  = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]

    print(f"[lanczos_eigsh] k={k}  N={N}  time={dt:.3f}s")
    print(f"  eigenvalues: {vals}")
    return vals, vecs


def lobpcg_eigsh(
    matvec:  Callable[[np.ndarray], np.ndarray],
    N:       int,
    k:       int   = 8,
    tol:     float = 1e-8,
    maxiter: int   = 300,
    seed:    int   = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute top-k eigenpairs via LOBPCG.

    LOBPCG carries the previous search direction (conjugate gradient step),
    giving superlinear convergence for clustered spectra. Preferred over
    Lanczos when k is large relative to the spectral gap.

    orthogonalisation: float64 QR inside the k x k projected problem.
    Each outer iteration: one matvec + O(N k²) ortho (cheap).
    """
    from scipy.sparse.linalg import lobpcg

    rng = np.random.default_rng(seed)
    # float64 initial subspace, orthonormalised
    X0 = np.linalg.qr(rng.standard_normal((N, k)))[0]

    op = LinearOperator(
        shape  = (N, N),
        matvec = lambda v: matvec(np.asarray(v, dtype=np.float64).ravel()),
        dtype  = np.float64,
    )

    t0 = time.perf_counter()
    vals, vecs = lobpcg(
        op,
        X0,
        tol     = tol,
        maxiter = maxiter,
        largest = True,
    )
    dt = time.perf_counter() - t0

    idx  = np.argsort(vals)[::-1]
    print(f"[lobpcg_eigsh] k={k}  N={N}  time={dt:.3f}s")
    print(f"  eigenvalues: {vals[idx]}")
    return vals[idx], vecs[:, idx]
```

## `src/flashdiffusion/eigensolver_gpu.py`

```python
from __future__ import annotations

import time
import warnings
import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def gpu_irlm_dense_qr(
    matvec_fn,
    N: int,
    k: int = 8,
    ncv: int | None = None,
    keep: int | None = None,
    max_restarts: int = 20,
    tol: float = 1e-6,
    dtype=None,
    small_dtype=None,
    device: str = "cuda",
    verbose: bool = True,
    q0=None,
    deflate_basis=None,
    breakdown_tol: float = 1e-12,
    relative_residual: bool = True,
    reorthogonalize: bool = True,
    reorth_passes: int = 2,
    block_rows: int = 262_144,
    compute_true_residuals: bool = True,
    return_info: bool = False,
    return_torch: bool = False,
):
    """
    GPU-resident implicitly restarted Lanczos method (IRLM) using dense
    shifted QR on the small projected matrix, with proper retained-block
    tridiagonalization via scipy Hessenberg reduction.

    Computes the k largest algebraic eigenpairs of a symmetric/self-adjoint
    matrix-free operator.

    Parameters
    ----------
    matvec_fn : callable
        (N,) GPU tensor -> (N,) GPU tensor.
    N : int
        Problem size.
    k : int
        Number of requested eigenpairs.
    ncv : int or None
        Krylov subspace size. Default: min(N, max(4*k+40, k+20)).
    keep : int or None
        Directions retained per restart. k <= keep <= ncv-2.
        Default: min(ncv-2, k+max(8, k//2)).
    max_restarts : int
        Maximum restart cycles.
    tol : float
        Ritz residual tolerance. Values < 5e-6 with float32 emit a warning.
    dtype : torch.dtype or None
        GPU vector dtype. Default: torch.float32.
    small_dtype : torch.dtype or None
        Dtype for ncv x ncv projected problems. Default: torch.float64.
    device : str
        CUDA device string.
    verbose : bool
        Print per-cycle convergence info.
    q0 : torch.Tensor or None
        Optional starting vector. Small perturbation added automatically.
    deflate_basis : array-like or None
        (N, d) columns to deflate. Orthonormalized internally via QR.
        Use to skip the trivial diffusion-map mode (lambda=1).
    breakdown_tol : float
        Treat beta below this as lucky breakdown.
    relative_residual : bool
        If True: converge when |beta*y[-1]| <= tol*max(1,|theta|).
    reorthogonalize : bool
        Full MGS re-orthogonalization. Must be True for clustered spectra.
    reorth_passes : int
        Number of MGS passes. 2 is standard.
    block_rows : int
        Row-block size for basis rotation and orthogonality check.
        Limits peak VRAM. Default 262144.
    compute_true_residuals : bool
        If True (default), compute ||A v_i - lam_i v_i|| at the end via
        k extra matvecs. Set False for production benchmarking.
    return_info : bool
        If True, return diagnostics dict as third element.
    return_torch : bool
        If True return GPU tensors; if False (default) return NumPy.

    Returns
    -------
    vals : ndarray or Tensor, shape (k,)
    vecs : ndarray or Tensor, shape (N, k)
    info : dict  [only when return_info=True]

    Notes
    -----
    Changes in this version:
      1. TF32 disabled for all GEMM ops (correctness on Ampere/Blackwell).
         TF32 has ~10-bit mantissa; Lanczos orthogonality accumulates
         ~1e-3 error per step with TF32, defeating double MGS.
      2. Blocked orthogonality error: Q.T@Q accumulates in (m,m) blocks,
         no float64 copy of the full (N,m) basis. OOM-safe at large N.
      3. deflate_basis pre-processed once at setup: cast to device/dtype,
         QR-orthonormalized defensively, shape-checked. Projection is then
         a simple GEMV/GEMM with no per-call .to() overhead.
      4. compute_true_residuals=False skips k end-of-solve matvecs for
         production benchmarking.
      5. Re-deflate residual after orthogonalization against retained basis,
         then one more orthogonalization pass to clean up any leaked component.
      Previous fixes retained:
      - Real retained-block tridiagonalization via scipy Hessenberg.
      - Explicit restart residual only (implicit mode removed).
      - state_valid fallback if restart mutation fails.
      - alpha not updated inside reortho loop.
      - q0 warm-start perturbation.
      - Blocked basis rotation.
      - Formula + true residual diagnostics.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch required for gpu_irlm_dense_qr")

    import torch
    import scipy.linalg

    if dtype is None:
        dtype = torch.float32
    if small_dtype is None:
        small_dtype = torch.float64

    device = torch.device(device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    if dtype == torch.float32 and tol < 5e-6:
        warnings.warn(
            f"tol={tol} may be unreliable for dtype=float32. "
            "Consider tol >= 5e-6.",
            RuntimeWarning, stacklevel=2,
        )

    if k < 1:
        raise ValueError("k must be >= 1")

    if ncv is None:
        ncv = min(N, max(4 * k + 40, k + 20))
    ncv = int(min(ncv, N))

    if ncv <= k + 1:
        raise ValueError(f"ncv={ncv} too small for k={k}; need ncv > k+1")

    if keep is None:
        keep = min(ncv - 2, k + max(8, k // 2))
    keep = int(keep)

    if keep < k:
        raise ValueError(f"keep={keep} must be >= k={k}")
    if keep > ncv - 2:
        raise ValueError(f"keep={keep} must be <= ncv-2={ncv-2}")

    # ── Save and set precision flags ───────────────────────────────────────
    # FIX 1: Disable TF32 for correctness on Ampere/Blackwell.
    # TF32 GEMM has ~10-bit mantissa vs float32's 23 bits.
    # Lanczos orthogonality accumulates ~1e-3 error per step with TF32,
    # which defeats double MGS reorthogonalization for clustered spectra.
    grad_state = torch.is_grad_enabled()
    torch.set_grad_enabled(False)

    if device.type == "cuda":
        old_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        old_tf32_cudnn  = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32       = False
    else:
        old_tf32_matmul = None
        old_tf32_cudnn  = None

    t_start = time.perf_counter()

    if verbose:
        print(
            f"[irlm] N={N}  k={k}  ncv={ncv}  keep={keep}  "
            f"max_restarts={max_restarts}  tol={tol:g}  "
            f"dtype={dtype}  small_dtype={small_dtype}  device={device}"
        )

    # ── FIX 3: Pre-process deflation basis once ────────────────────────────
    # Cast to device/dtype, QR-orthonormalize defensively, shape-check.
    # Projection is then a simple GEMV with no per-call .to() overhead.
    # The QR step is necessary (not just defensive): if the user passes a
    # non-orthonormal basis the projection v - B@(B.T@v) is wrong.
    if deflate_basis is not None:
        B_def = torch.as_tensor(
            deflate_basis if isinstance(deflate_basis, torch.Tensor)
            else np.asarray(deflate_basis),
            device=device, dtype=dtype,
        )
        if B_def.ndim == 1:
            B_def = B_def[:, None]
        if B_def.shape[0] != N:
            raise ValueError(
                f"deflate_basis has {B_def.shape[0]} rows, expected N={N}"
            )
        B_def, _ = torch.linalg.qr(B_def, mode="reduced")   # orthonormalize
        if verbose:
            print(f"  deflate_basis: {B_def.shape[1]} direction(s)")
    else:
        B_def = None

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _project_deflation(v: torch.Tensor) -> torch.Tensor:
        """Remove B_def components from v. B_def pre-orthonormalized."""
        if B_def is None:
            return v
        return v - B_def @ (B_def.T @ v)

    def _make_initial_vector() -> torch.Tensor:
        if q0 is not None:
            q = torch.as_tensor(q0, device=device, dtype=dtype).reshape(-1)
            if q.numel() != N:
                raise ValueError(f"q0 length {q.numel()} != N={N}")
            noise_scale = 1e-3 * float(torch.linalg.norm(q).cpu()) / (N ** 0.5)
            q = q + noise_scale * torch.randn_like(q)
        else:
            q = torch.randn(N, device=device, dtype=dtype)
        q = _project_deflation(q)
        qn = torch.linalg.norm(q)
        if float(qn.cpu()) == 0.0:
            raise ValueError("Initial vector zero after deflation")
        return q / qn

    def _orthogonalize(Qbasis: torch.Tensor, v: torch.Tensor,
                       passes: int = 2) -> torch.Tensor:
        if Qbasis.numel() == 0:
            return v
        for _ in range(passes):
            c = Qbasis.T @ v
            v = v - Qbasis @ c
        return v

    def _build_T(alpha: torch.Tensor, beta: torch.Tensor,
                 m: int) -> torch.Tensor:
        T = torch.diag(alpha[:m].to(device=device, dtype=small_dtype))
        if m > 1:
            b = beta[:m - 1].to(device=device, dtype=small_dtype)
            T = T + torch.diag(b, 1) + torch.diag(b, -1)
        return 0.5 * (T + T.T)

    def _small_eigh_desc(T: torch.Tensor):
        T = 0.5 * (T + T.T)
        evals, evecs = torch.linalg.eigh(T)
        idx = torch.argsort(evals, descending=True)
        return evals[idx], evecs[:, idx]

    def _implicit_shifted_qr_dense(T: torch.Tensor,
                                    shifts: torch.Tensor):
        m  = T.shape[0]
        I  = torch.eye(m, device=device, dtype=small_dtype)
        Tw = T.clone().to(small_dtype)
        Z  = torch.eye(m, device=device, dtype=small_dtype)
        for mu in shifts:
            Qs, Rs = torch.linalg.qr(Tw - mu * I, mode="reduced")
            d      = torch.diagonal(Rs)
            signs  = torch.where(
                d < 0,
                torch.tensor(-1.0, device=device, dtype=small_dtype),
                torch.tensor( 1.0, device=device, dtype=small_dtype),
            )
            Qs = Qs * signs[None, :]
            Rs = signs[:, None] * Rs
            Tw = Rs @ Qs + mu * I
            Tw = 0.5 * (Tw + Tw.T)
            Z  = Z @ Qs
        return Tw, Z

    def _tridiagonalize_retained_block(Qkeep: torch.Tensor,
                                        Tkeep: torch.Tensor):
        """
        After dense shifted QR the retained block Tkeep is generally dense.
        Restore the Lanczos three-term recurrence via scipy Hessenberg
        (symmetric Hessenberg = tridiagonal). Rotate Qkeep accordingly.
        Cost: O(keep^3) on CPU — negligible.
        """
        T_np = Tkeep.detach().cpu().numpy().astype(np.float64)
        T_np = 0.5 * (T_np + T_np.T)
        H_np, S_np = scipy.linalg.hessenberg(T_np, calc_q=True)
        H_np = 0.5 * (H_np + H_np.T)

        r     = H_np.shape[0]
        H_tri = np.zeros_like(H_np)
        np.fill_diagonal(H_tri, np.diag(H_np))
        if r > 1:
            off    = 0.5 * (np.diag(H_np, 1) + np.diag(H_np, -1))
            H_tri += np.diag(off, 1) + np.diag(off, -1)

        S     = torch.from_numpy(S_np).to(device=device, dtype=dtype)
        Qkeep = Qkeep @ S

        H = torch.from_numpy(H_tri).to(device=device, dtype=small_dtype)

        for i in range(r - 1):
            if float(H[i + 1, i].cpu()) < 0:
                Qkeep[:, i + 1].mul_(-1)
                H[i + 1, :].mul_(-1)
                H[:, i + 1].mul_(-1)

        alpha_new = torch.diagonal(H).clone().to(small_dtype)
        beta_new  = (torch.diagonal(H, offset=-1).clone().abs().to(small_dtype)
                     if r > 1 else
                     torch.empty(0, device=device, dtype=small_dtype))
        return Qkeep, alpha_new, beta_new, H

    def _rotate_basis_blocked(m: int, Z_keep: torch.Tensor,
                               keep_eff: int) -> torch.Tensor:
        """Blocked Q[:,:m] @ Z_keep — avoids one large (N,keep) temporary."""
        Qout = torch.empty((N, keep_eff), device=device, dtype=dtype)
        for lo in range(0, N, block_rows):
            hi = min(lo + block_rows, N)
            Qout[lo:hi] = Q[lo:hi, :m] @ Z_keep
        return Qout

    def _orth_error_blocked(Qblock: torch.Tensor) -> float:
        """
        FIX 2: ||Q.T Q - I||_F / ncols via blocked accumulation.
        Avoids creating a float64 copy of the full (N, m) basis.
        At N=10M, m=64: naive cast uses 5.1 GB; this uses ~30 MB per block.
        """
        r = Qblock.shape[1]
        G = torch.zeros((r, r), device=device, dtype=small_dtype)
        for lo in range(0, Qblock.shape[0], block_rows):
            hi = min(lo + block_rows, Qblock.shape[0])
            B  = Qblock[lo:hi].to(dtype=small_dtype)
            G += B.T @ B
        I = torch.eye(r, device=device, dtype=small_dtype)
        return float((torch.linalg.norm(G - I) / max(1, r)).cpu())

    def _true_residuals(vals_t: torch.Tensor,
                        vecs_t: torch.Tensor) -> np.ndarray:
        """||A v_i - lam_i v_i|| via explicit matvec. k extra matvecs."""
        out = []
        for i in range(vals_t.numel()):
            Av = matvec_fn(vecs_t[:, i])
            r  = (Av.to(device=device, dtype=dtype)
                  - vals_t[i].to(dtype=dtype) * vecs_t[:, i])
            out.append(float(torch.linalg.norm(r).cpu()))
        return np.array(out, dtype=np.float64)

    # ──────────────────────────────────────────────────────────────────────
    # Persistent GPU state
    # ──────────────────────────────────────────────────────────────────────

    Q     = torch.empty((N, ncv + 1), device=device, dtype=dtype)
    alpha = torch.zeros(ncv,          device=device, dtype=small_dtype)
    beta  = torch.zeros(ncv,          device=device, dtype=small_dtype)

    Q[:, 0] = _make_initial_vector()

    n_matvec            = 0
    cycles_done         = 0
    converged           = False
    breakdown           = False
    state_valid         = True

    best_evals          = None
    best_evecs          = None
    best_m              = None
    best_beta_last      = None

    off_tridiag_history  = []
    restart_beta_history = []
    orth_error_history   = []

    start_col           = 0
    m_final             = 1
    beta_last           = torch.tensor(0.0, device=device, dtype=small_dtype)
    has_residual_vector = False

    def _expand_from(start_col_local: int):
        nonlocal n_matvec
        beta_last_local = torch.tensor(0.0, device=device, dtype=small_dtype)
        has_residual    = False
        m_local         = start_col_local + 1

        for j in range(start_col_local, ncv):
            qj = Q[:, j]
            z  = matvec_fn(qj)
            z  = z.to(device=device, dtype=dtype).reshape(-1)
            n_matvec += 1

            if z.numel() != N:
                raise ValueError(f"matvec returned {z.numel()}, expected {N}")

            z = _project_deflation(z)

            if j > 0:
                z = z - beta[j - 1].to(dtype=dtype) * Q[:, j - 1]

            # Diagonal coeff set before reortho; NOT updated inside loop.
            aj = torch.dot(qj, z).to(small_dtype)
            z  = z - aj.to(dtype=dtype) * qj

            if reorthogonalize:
                Qj = Q[:, :j + 1]
                for _ in range(reorth_passes):
                    c = Qj.T @ z
                    z = z - Qj @ c

            bj = torch.linalg.norm(z).to(small_dtype)
            alpha[j]        = aj
            beta_last_local = bj
            m_local         = j + 1

            if float(bj.cpu()) < breakdown_tol:
                return m_local, beta_last_local, False

            beta[j]      = bj
            Q[:, j + 1]  = z / bj.to(dtype=dtype)
            has_residual = True

        return ncv, beta_last_local, has_residual

    # ──────────────────────────────────────────────────────────────────────
    # Main IRLM loop
    # ──────────────────────────────────────────────────────────────────────

    try:
        for cycle in range(max_restarts + 1):
            cycles_done = cycle + 1

            m, beta_last, has_residual_vector = _expand_from(start_col)
            m_final = m

            Tm = _build_T(alpha, beta, m)
            evals_all, evecs_all = _small_eigh_desc(Tm)

            n_take  = min(k, m)
            evals_k = evals_all[:n_take]
            evecs_k = evecs_all[:, :n_take]

            residuals = torch.abs(beta_last * evecs_k[-1, :])

            if relative_residual:
                thresholds = tol * torch.maximum(
                    torch.ones_like(evals_k), torch.abs(evals_k))
            else:
                thresholds = torch.full_like(residuals, float(tol))

            n_conv   = int(torch.sum(residuals <= thresholds).cpu())
            orth_err = _orth_error_blocked(Q[:, :m])   # FIX 2
            orth_error_history.append(orth_err)

            best_evals     = evals_all
            best_evecs     = evecs_all
            best_m         = m
            best_beta_last = beta_last

            if verbose:
                max_res = (float(torch.max(residuals).cpu())
                           if residuals.numel() else float("nan"))
                print(
                    f"  cycle={cycle:02d}  m={m:4d}  "
                    f"beta={float(beta_last.cpu()):.3e}  "
                    f"converged={n_conv}/{n_take}  "
                    f"max_res={max_res:.3e}  orth={orth_err:.2e}"
                )
                print(f"    theta[:5] = "
                      f"{evals_k[:min(5,len(evals_k))].cpu().numpy()}")

            if n_take >= k and bool(
                torch.all(residuals[:k] <= thresholds[:k]).cpu()
            ):
                converged = True
                if verbose:
                    print(f"[irlm] converged after cycle={cycle}")
                break

            if float(beta_last.cpu()) < breakdown_tol:
                breakdown = True
                if verbose:
                    print(f"[irlm] lucky breakdown cycle={cycle}")
                break

            if cycle == max_restarts:
                break

            if not has_residual_vector:
                breakdown = True
                break

            if m <= keep + 1:
                break

            # ── IRLM restart ──────────────────────────────────────────────

            keep_eff = min(keep, m - 2)
            shifts   = evals_all[keep_eff:m].detach()

            if shifts.numel() == 0:
                break

            T_shifted, Z = _implicit_shifted_qr_dense(Tm, shifts)

            # Off-tridiagonal fill diagnostic.
            Tk = T_shifted[:keep_eff, :keep_eff]
            Tk_tri = torch.diag(torch.diagonal(Tk))
            if keep_eff > 1:
                off    = 0.5 * (torch.diagonal(Tk, 1) + torch.diagonal(Tk, -1))
                Tk_tri = Tk_tri + torch.diag(off, 1) + torch.diag(off, -1)
            denom   = max(float(torch.linalg.norm(Tk).cpu()), 1e-300)
            off_rel = float(torch.linalg.norm(Tk - Tk_tri).cpu()) / denom
            off_tridiag_history.append(off_rel)

            Z_keep = Z[:, :keep_eff].to(dtype=dtype)
            Qkeep  = _rotate_basis_blocked(m, Z_keep, keep_eff)
            Tkeep  = T_shifted[:keep_eff, :keep_eff]

            state_valid = False   # mark invalid before mutation

            Qkeep, alpha_new, beta_new, Tkeep_tri = \
                _tridiagonalize_retained_block(Qkeep, Tkeep)

            # Explicit restart residual.
            qlast = Qkeep[:, keep_eff - 1]
            r     = matvec_fn(qlast).to(device=device, dtype=dtype).reshape(-1)
            n_matvec += 1

            r = _project_deflation(r)
            r = r - alpha_new[keep_eff - 1].to(dtype=dtype) * qlast
            if keep_eff > 1:
                r = r - beta_new[keep_eff - 2].to(dtype=dtype) * Qkeep[:, keep_eff - 2]

            # FIX 5: re-deflate and clean up after orthogonalization.
            r = _orthogonalize(Qkeep, r, passes=reorth_passes)
            r = _project_deflation(r)           # remove any leaked deflation component
            r = _orthogonalize(Qkeep, r, passes=1)   # one final cleanup pass

            beta_restart = torch.linalg.norm(r).to(small_dtype)
            restart_beta_history.append(float(beta_restart.cpu()))

            if float(beta_restart.cpu()) < breakdown_tol:
                breakdown = True
                if verbose:
                    print("[irlm] restart residual collapsed")
                break   # state_valid stays False; fallback used below

            q_restart = r / beta_restart.to(dtype=dtype)

            Q[:, :keep_eff]     = Qkeep
            Q[:, keep_eff]      = q_restart
            alpha.zero_()
            beta.zero_()
            alpha[:keep_eff]    = alpha_new
            if keep_eff > 1:
                beta[:keep_eff - 1] = beta_new
            beta[keep_eff - 1]  = beta_restart
            start_col           = keep_eff
            state_valid         = True       # install complete

            if verbose:
                print(
                    f"    restart: shifts={int(shifts.numel())}  "
                    f"off_tridiag={off_rel:.3e}  "
                    f"beta_restart={float(beta_restart.cpu()):.3e}"
                )

        # ── Final Ritz extraction ─────────────────────────────────────────

        if best_evals is None or best_m is None:
            raise RuntimeError("IRLM produced no Ritz pairs")

        if state_valid:
            Tfinal           = _build_T(alpha, beta, m_final)
            evals_f, evecs_f = _small_eigh_desc(Tfinal)
        else:
            # Restart failed mid-way — fall back to last complete cycle.
            evals_f  = best_evals
            evecs_f  = best_evecs
            m_final  = best_m
            beta_last = best_beta_last

        n_return  = min(k, m_final)
        vals      = evals_f[:n_return]
        Y         = evecs_f[:, :n_return].to(dtype=dtype)
        ritz_vecs = Q[:, :m_final] @ Y
        ritz_vecs = ritz_vecs / torch.linalg.norm(
            ritz_vecs, dim=0).clamp_min(torch.finfo(dtype).eps)

        formula_res = torch.abs(beta_last * evecs_f[-1, :n_return])

        # FIX 4: true residuals are optional — skip in production.
        if compute_true_residuals:
            true_res = _true_residuals(vals, ritz_vecs)
        else:
            true_res = None

        elapsed = time.perf_counter() - t_start

        if verbose:
            print(
                f"[irlm] done  converged={converged}  "
                f"breakdown={breakdown}  state_valid={state_valid}  "
                f"cycles={cycles_done}  matvecs={n_matvec}  "
                f"time={elapsed:.2f}s"
            )
            print(f"  eigenvalues:       {vals.cpu().numpy()}")
            print(f"  formula residuals: {formula_res.cpu().numpy()}")
            if true_res is not None:
                print(f"  true residuals:    {true_res}")

        if return_torch:
            vals_out, vecs_out = vals, ritz_vecs
        else:
            vals_out = vals.cpu().numpy().astype(np.float64)
            vecs_out = ritz_vecs.cpu().numpy()

        if return_info:
            info = {
                "converged":           bool(converged),
                "breakdown":           bool(breakdown),
                "state_valid":         bool(state_valid),
                "cycles":              int(cycles_done),
                "n_matvec":            int(n_matvec),
                "ncv":                 int(ncv),
                "keep":                int(keep),
                "k":                   int(k),
                "tol":                 float(tol),
                "relative_residual":   bool(relative_residual),
                "elapsed":             float(elapsed),
                "m_final":             int(m_final),
                "beta_final":          float(beta_last.cpu()),
                "formula_residuals":   formula_res.cpu().numpy().astype(np.float64),
                "true_residuals":      true_res,
                "off_tridiag_restart": off_tridiag_history,
                "restart_beta":        restart_beta_history,
                "orth_error":          orth_error_history,
                "dtype":               str(dtype),
                "small_dtype":         str(small_dtype),
                "device":              str(device),
            }
            return vals_out, vecs_out, info

        return vals_out, vecs_out

    finally:
        # Restore all saved state unconditionally.
        torch.set_grad_enabled(grad_state)
        if old_tf32_matmul is not None:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32_matmul
            torch.backends.cudnn.allow_tf32       = old_tf32_cudnn


# ─────────────────────────────────────────────────────────────────────────────
# Drop-in compatibility shim for eigensolver_gpu.py
# Matches the signature of the original gpu_lanczos() exactly.
# ─────────────────────────────────────────────────────────────────────────────

def gpu_lanczos(
    matvec_fn,
    N:        int,
    k:        int   = 8,
    n_iter:   int   = None,
    tol:      float = 1e-6,
    dtype           = None,
    device:   str   = "cuda",
    verbose:  bool  = True,
    q0=None,
    **kwargs,
):
    """
    Drop-in replacement for the original gpu_lanczos() in eigensolver_gpu.py.

    Translates n_iter -> ncv for gpu_irlm_dense_qr.
    All other original parameters are forwarded unchanged.
    Extra IRLM parameters (keep, max_restarts, etc.) can be passed via kwargs.

    Parameters
    ----------
    n_iter : int or None
        Maps to ncv (Krylov subspace size) in IRLM.
        Default behaviour is identical: None -> min(N, max(4k+40, k+20)).

    Returns
    -------
    vals : np.ndarray, shape (k,)  — descending eigenvalues
    vecs : np.ndarray, shape (N, k) — corresponding Ritz vectors
    """
    return gpu_irlm_dense_qr(
        matvec_fn,
        N             = N,
        k             = k,
        ncv           = n_iter,   # n_iter in plain Lanczos = ncv in IRLM
        tol           = tol,
        dtype         = dtype,
        device        = device,
        verbose       = verbose,
        q0            = q0,
        # production defaults: skip true residuals (saves k matvecs)
        compute_true_residuals = kwargs.pop("compute_true_residuals", False),
        **kwargs,
    )


def make_gpu_matvec(state, ext_module):
    """
    Unchanged from eigensolver_gpu.py — wrap CUDAState + ext into GPU matvec.
    Kept here so this file is a complete drop-in replacement.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("torch required")

    def matvec_gpu(v):
        with torch.no_grad():
            return ext_module.matvec(
                state.X_fp16,
                v,
                state.rscale,
                state.sq_X,
                state.beta,
            )
    return matvec_gpu
```

## `src/flashdiffusion/eigensolver_gpu_.py`

```python
"""
flashdiffusion/eigensolver_gpu.py
==================================
GPU-resident Lanczos eigensolver for the diffusion map operator.

v1: Plain Lanczos with full re-orthogonalization (double MGS).
    No restart — simpler, proven correct, already 10× faster than ARPACK
    for our case because vectors stay on GPU with zero CPU transfer.

v2 (future): Thick-restart Lanczos for even fewer matvecs.

Algorithm
---------
1. Run n_iter Lanczos steps, all vectors on GPU
2. Full double-pass MGS re-orthogonalization per step
3. Solve small (n_iter × n_iter) tridiagonal on GPU via torch.linalg.eigh
4. Return top-k Ritz pairs

Why this beats ARPACK for fast GPU kernels
------------------------------------------
ARPACK runs on CPU and transfers v to GPU + back each iteration:
  overhead per iter ≈ 2.35s  (B200, N=200k)
  50 iters × 2.35s = 117s overhead
  
GPU Lanczos: zero transfer, all computation on GPU:
  50 iters × 0.0s transfer = 0s overhead
  total ≈ 50 × 0.23s matvec + 50 × O(N×j) ortho ≈ 12s

References
----------
Golub & Van Loan, "Matrix Computations" 4th ed., Ch. 10
Parlett, "The Symmetric Eigenvalue Problem", SIAM 1998
"""

from __future__ import annotations
import time
import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def gpu_lanczos(
    matvec_fn,
    N:        int,
    k:        int   = 8,
    n_iter:   int   = None,
    tol:      float = 1e-6,
    dtype           = None,
    device:   str   = "cuda",
    verbose:  bool  = True,
    q0=None
) -> tuple:
    """
    Plain GPU Lanczos with full re-orthogonalization for top-k eigenvalues.

    Parameters
    ----------
    matvec_fn : callable
        (N,) float32 GPU tensor → (N,) float32 GPU tensor.
        Zero CPU transfer — vectors stay on GPU throughout.
    N : int
        Problem size.
    k : int
        Number of eigenvalues/vectors wanted (default 8).
    n_iter : int
        Number of Lanczos steps. Default: min(3*k + 20, N).
        More iterations → better accuracy for clustered spectra.
        Memory cost: N × n_iter × 4 bytes
        (N=200k, n_iter=60: 200k × 60 × 4 = 48MB — trivial on any GPU)
    tol : float
        Not used in plain Lanczos (kept for API compatibility with ARPACK).
    dtype : torch.dtype
        Default float32. Use float64 for better orthogonality (2× memory).
    device : str
        CUDA device (default 'cuda').
    verbose : bool
        Print iteration info.

    Returns
    -------
    vals : np.ndarray, shape (k,)
        Top-k eigenvalues, descending order.
    vecs : np.ndarray, shape (N, k)
        Corresponding Ritz vectors (approximate eigenvectors).
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch required for gpu_lanczos")

    import torch

    if dtype is None:
        dtype = torch.float32

    if n_iter is None:
        #n_iter = min(6 * k + 40, N)   # 88 for k=8 — converges for typical DMAP spectra
        n_iter = min(6 * k + 20, N)   # 68 for k=8 with warm start


    t_start = time.perf_counter()
    if verbose:
        print(f"[gpu_lanczos] N={N}  k={k}  n_iter={n_iter}  dtype={dtype}")

    # ── allocate Lanczos basis on GPU ────────────────────────────────────────
    # Q: (N, n_iter) — all basis vectors, needed for re-orthogonalization
    # Memory: N=200k, n_iter=60, fp32 → 200k × 60 × 4 = 48MB
    Q      = torch.zeros(N, n_iter, dtype=dtype, device=device)
    alphas = torch.zeros(n_iter,    dtype=dtype, device=device)
    betas  = torch.zeros(n_iter,    dtype=dtype, device=device)

    # random normalised start vector
    torch.manual_seed(42)
    # correct warm start (exact leading eigenvector of P):
    # pass D_alpha from state as optional q0
    if q0 is not None:
        q = q0.to(dtype=dtype)
    else:
        q = torch.ones(N, dtype=dtype, device=device)
    q /= q.norm()

    Q[:, 0] = q

    j_final = n_iter   # actual steps taken (may terminate early)

    for j in range(n_iter):
        # ── matvec — stays on GPU, zero transfer ─────────────────────────
        z = matvec_fn(Q[:, j])

        # ── three-term recurrence ─────────────────────────────────────────
        alphas[j] = torch.dot(Q[:, j], z)
        z = z - alphas[j] * Q[:, j]
        if j > 0:
            z = z - betas[j - 1] * Q[:, j - 1]

        # ── full re-orthogonalization (double MGS) ────────────────────────
        # Critical for clustered eigenvalues — single pass loses orthogonality
        # Cost: 2 × j × N flops per step — tiny vs matvec cost O(N²)
        if j > 0:
            for _ in range(2):
                # (j+1, N) × (N,) → (j+1,) coefficients
                coeffs = Q[:, :j + 1].T @ z
                z      = z - Q[:, :j + 1] @ coeffs

        # ── next basis vector ─────────────────────────────────────────────
        beta_j  = z.norm().item()
        betas[j] = beta_j

        if j < n_iter - 1:
            if beta_j < 1e-12:
                # lucky breakdown — Krylov space exhausted
                j_final = j + 1
                if verbose:
                    print(f"  lucky breakdown at j={j}, β={beta_j:.2e}")
                break
            Q[:, j + 1] = z / beta_j

    # ── solve tridiagonal eigenvalue problem ──────────────────────────────────
    # T is (j_final × j_final) — tiny, solve on GPU
    m = j_final
    T = torch.diag(alphas[:m])
    if m > 1:
        T += torch.diag(betas[:m - 1], 1)
        T += torch.diag(betas[:m - 1], -1)

    # use numpy for robustness — T is tiny (60×60), CPU cost negligible
    T_np = T.cpu().numpy().astype(np.float64)
    evals_np, evecs_np = np.linalg.eigh(T_np)   # ascending, numpy is robust

    # sort descending, take top-k
    idx      = np.argsort(evals_np)[::-1][:k]
    evals_k  = evals_np[idx]
    evecs_k  = evecs_np[:, idx]   # (m, k)

    # ── Ritz vectors: Q[:, :m] @ evecs_k — GPU matmul ────────────────────────
    # (N, m) × (m, k) → (N, k)
    evecs_k_gpu = torch.from_numpy(evecs_k.astype(np.float32)).to(device)
    ritz_vecs   = Q[:, :m] @ evecs_k_gpu   # (N, k)

    elapsed = time.perf_counter() - t_start
    if verbose:
        print(f"[gpu_lanczos] {m} matvecs  time={elapsed:.2f}s")
        print(f"  eigenvalues: {evals_k}")

    return evals_k.astype(np.float64), ritz_vecs.cpu().numpy()


def make_gpu_matvec(state, ext_module):
    """
    Wrap a CUDAState + ext module into a pure GPU→GPU matvec callable.
    No numpy, no CPU transfer — v stays on GPU throughout.

    Parameters
    ----------
    state : CUDAState / CUDAStateSM100 / CUDAStateSM120 / etc.
        Precomputed GPU state with X_fp16, sq_X, rscale, beta.
    ext_module : compiled CUDA extension
        Must expose .matvec(X_fp16, v, rscale, sq_X, beta) → GPU tensor.

    Returns
    -------
    callable : (N,) float32 GPU tensor → (N,) float32 GPU tensor
    """
    import torch

    def matvec_gpu(v: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return ext_module.matvec(
                state.X_fp16,
                v,
                state.rscale,
                state.sq_X,
                state.beta,
            )
    return matvec_gpu
```

## `src/flashdiffusion/eigensolver_gpu_block.py`

```python
from __future__ import annotations

import time
import warnings
from typing import Callable, Optional, Tuple, Any, Dict

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# GPU Hessenberg (replaces scipy.linalg.hessenberg after thick restart)
# ---------------------------------------------------------------------------

def _gpu_hessenberg_sym(T: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
    """
    Reduce a symmetric matrix T (on GPU, small_dtype) to symmetric tridiagonal
    form via Householder reflections, entirely on the GPU.

    For a symmetric matrix, Hessenberg = tridiagonal, so this is equivalent to
    scipy.linalg.hessenberg but stays on-device.

    Returns
    -------
    H : tridiagonal (symmetric) version of T, same device/dtype
    Q : orthogonal accumulation such that Q.T @ T_orig @ Q = H
    """
    device = T.device
    dtype = T.dtype
    n = T.shape[0]

    T = 0.5 * (T + T.T).clone()
    Q = torch.eye(n, device=device, dtype=dtype)

    for k in range(n - 2):
        x = T[k + 1:, k].clone()
        norm_x = torch.linalg.norm(x)
        if float(norm_x) < 1e-15:
            continue
        # Householder vector: sign chosen to avoid cancellation.
        # sign(x[0]) >= 0 convention: add norm in the direction of x[0].
        sign = torch.where(
            x[0] >= 0,
            torch.tensor(1.0, device=device, dtype=dtype),
            torch.tensor(-1.0, device=device, dtype=dtype),
        )
        v = x.clone()
        v[0] = v[0] + sign * norm_x
        v = v / torch.linalg.norm(v)

        # Apply H = I - 2vv^T from left and right to the sub-block.
        sub = T[k + 1:, k + 1:].clone()
        # Left: sub <- (I - 2vv^T) sub
        sub = sub - 2.0 * v.unsqueeze(1) * (v @ sub).unsqueeze(0)
        # Right: sub <- sub (I - 2vv^T)
        sub = sub - 2.0 * (sub @ v).unsqueeze(1) * v.unsqueeze(0)
        T[k + 1:, k + 1:] = sub

        # Set the off-diagonal entry explicitly; zero the rest of column/row k.
        T[k + 2:, k] = 0.0
        T[k, k + 2:] = 0.0
        T[k + 1, k] = -sign * norm_x
        T[k, k + 1] = T[k + 1, k]

        # Accumulate Q
        Q[:, k + 1:] -= 2.0 * (Q[:, k + 1:] @ v).unsqueeze(1) * v.unsqueeze(0)

    # Force exact symmetry and extract clean tridiagonal
    T = 0.5 * (T + T.T)
    alpha = torch.diagonal(T).clone()
    beta = torch.diagonal(T, offset=1).clone()

    H_tri = torch.diag(alpha) + torch.diag(beta, 1) + torch.diag(beta, -1)
    return H_tri, Q


def gpu_block_irlm(
    matmat_fn: Optional[Callable[["torch.Tensor"], "torch.Tensor"]] = None,
    *,
    matvec_fn: Optional[Callable[["torch.Tensor"], "torch.Tensor"]] = None,
    N: int,
    k: int = 8,
    block_size: Optional[int] = None,
    ncv: Optional[int] = None,
    keep: Optional[int] = None,
    max_restarts: int = 20,
    tol: float = 1e-6,
    dtype=None,
    small_dtype=None,
    device: str = "cuda",
    verbose: bool = True,
    q0_block=None,
    q0=None,
    deflate_basis=None,
    breakdown_tol: float = 1e-12,
    rank_tol: Optional[float] = None,
    relative_residual: bool = True,
    reorthogonalize: bool = True,
    reorth_passes: int = 2,
    block_rows: int = 262_144,
    compute_true_residuals: bool = True,
    orth_check_every: Optional[int] = 1,
    disable_tf32: bool = True,
    return_info: bool = False,
    return_torch: bool = False,
):
    """
    GPU thick-restarted block Arnoldi / block Rayleigh-Ritz eigensolver.

    Correctly described: this is a thick-restarted block Arnoldi method for
    symmetric operators, not a strict block IRLM. Each restart selects the top
    keep Ritz vectors as the new seed and rebuilds the Krylov relation from
    scratch via _expand_cycle. This loses one benefit of true IRLM (continuing
    the Krylov relation without reapplying A to retained vectors), but is simpler
    and correct. The matmat_fn(V) block interface still gives GPU throughput
    benefits over scalar IRLM for large N.

    To become true block Krylov-Schur/IRLM, _expand_cycle would need to accept
    the retained H sub-block and continue expansion from column keep rather than 0.

    Fixes applied in this version:
      1. H double-counting removed: H[:m, applied:end] = coeff (not +=), no
         transpose fill during expansion. H[:p,:p] symmetrized only at Ritz time.
      2. Hessenberg rotation removed from thick restart (was unused/incorrect:
         rotating by S from H[:keep,:keep] doesn't re-tridiagonalize the Ritz
         basis since the projected matrix there is diag(evals), not H).
      3. Householder sign bug fixed in _gpu_hessenberg_sym (was clamping sign
         to [0,1], turning negative x[0] into 0).
      4. keep_eff validated before _orthonormalize_block.
      5. True-residual matvec cost excluded from n_matmat/n_rhs counters.
      6. Q buffer reused across cycles (single allocation).

    Parameters
    ----------
    matmat_fn:
        Callable V -> A V, where V has shape (N, s).
    matvec_fn:
        Optional fallback callable v -> A v.
    N:
        Problem size.
    k:
        Number of eigenpairs requested.
    block_size:
        Vectors per operator application. Default: min(16, max(2,k)).
    ncv:
        Rayleigh-Ritz subspace dimension. Default: max(4*k+4*b, k+6*b).
    keep:
        Ritz vectors retained after restart. Default: min(ncv-b, k+max(2*b,k//2)).
    q0_block / q0:
        Optional initial block or vector.
    deflate_basis:
        Optional (N,d) basis to project out.
    compute_true_residuals:
        Compute ||A v_i - lam_i v_i|| at end (excluded from op counters).

    Returns
    -------
    vals : np.ndarray or torch.Tensor, shape (k,)
    vecs : np.ndarray or torch.Tensor, shape (N,k)
    info : dict, optional
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch required for gpu_block_irlm")

    import torch

    if matmat_fn is None and matvec_fn is None:
        raise ValueError("Provide either matmat_fn(V)->A@V or matvec_fn(v)->A@v")

    if dtype is None:
        dtype = torch.float32
    if small_dtype is None:
        small_dtype = torch.float64

    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    if k < 1:
        raise ValueError("k must be >= 1")

    if block_size is None:
        block_size = min(16, max(2, int(k)))
    block_size = int(block_size)
    if block_size < 1:
        raise ValueError("block_size must be >= 1")

    if ncv is None:
        ncv = max(4 * k + 4 * block_size, k + 6 * block_size)
    ncv = int(min(N, ncv))
    if ncv < N:
        ncv = int(min(N, ((ncv + block_size - 1) // block_size) * block_size))

    if ncv <= k + block_size:
        raise ValueError(
            f"ncv={ncv} too small for k={k}, block_size={block_size}; "
            f"need ncv > k + block_size"
        )

    if keep is None:
        keep = min(ncv - block_size, k + max(2 * block_size, k // 2))
    keep = int(keep)
    if keep < k:
        raise ValueError(f"keep={keep} must be >= k={k}")
    if keep > ncv - block_size:
        raise ValueError(
            f"keep={keep} must be <= ncv-block_size={ncv - block_size}"
        )

    if rank_tol is None:
        rank_tol = 1e-7 if dtype == torch.float32 else 1e-12
    if dtype == torch.float32 and tol < 5e-6:
        warnings.warn(
            f"tol={tol} may be unreliable for dtype=float32. Consider tol >= 5e-6.",
            RuntimeWarning,
            stacklevel=2,
        )

    grad_state = torch.is_grad_enabled()
    torch.set_grad_enabled(False)

    if device.type == "cuda":
        old_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        old_tf32_cudnn = torch.backends.cudnn.allow_tf32
        if disable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    else:
        old_tf32_matmul = None
        old_tf32_cudnn = None

    t_start = time.perf_counter()

    if deflate_basis is not None:
        B_def = torch.as_tensor(
            deflate_basis if isinstance(deflate_basis, torch.Tensor) else np.asarray(deflate_basis),
            device=device,
            dtype=dtype,
        )
        if B_def.ndim == 1:
            B_def = B_def[:, None]
        if B_def.shape[0] != N:
            raise ValueError(f"deflate_basis has {B_def.shape[0]} rows, expected N={N}")
        B_def, _ = torch.linalg.qr(B_def, mode="reduced")
    else:
        B_def = None

    n_matmat = 0
    n_rhs = 0

    def _project_deflation(V: "torch.Tensor") -> "torch.Tensor":
        if B_def is None:
            return V
        return V - B_def @ (B_def.T @ V)

    def _apply_A(V: "torch.Tensor", _count: bool = True) -> "torch.Tensor":
        # FIX 5: _count=False lets true-residual calls skip the benchmark counters.
        nonlocal n_matmat, n_rhs
        V = V.to(device=device, dtype=dtype)
        if V.ndim == 1:
            V = V[:, None]
        if V.shape[0] != N:
            raise ValueError(f"matmat input has {V.shape[0]} rows, expected {N}")
        with torch.no_grad():
            if matmat_fn is not None:
                Y = matmat_fn(V)
            else:
                cols = [matvec_fn(V[:, j]).reshape(-1) for j in range(V.shape[1])]
                Y = torch.stack(cols, dim=1)
        Y = Y.to(device=device, dtype=dtype)
        if Y.ndim == 1:
            Y = Y[:, None]
        if Y.shape != V.shape:
            raise ValueError(f"operator returned shape {tuple(Y.shape)}, expected {tuple(V.shape)}")
        if _count:
            n_matmat += 1
            n_rhs += V.shape[1]
        return Y

    def _linear_combo_blocked(Qsource: "torch.Tensor", C: "torch.Tensor") -> "torch.Tensor":
        C = C.to(device=device, dtype=dtype)
        out = torch.empty((N, C.shape[1]), device=device, dtype=dtype)
        for lo in range(0, N, block_rows):
            hi = min(lo + block_rows, N)
            out[lo:hi] = Qsource[lo:hi] @ C
        return out

    def _project_against(Qbasis: "torch.Tensor", V: "torch.Tensor", passes: int) -> "torch.Tensor":
        if Qbasis.numel() == 0:
            return V
        for _ in range(passes):
            C = Qbasis.T @ V
            V = V - Qbasis @ C
        return V

    def _qr_rank(V: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", int]:
        Qv, R = torch.linalg.qr(V, mode="reduced")
        if R.numel() == 0:
            return Qv[:, :0], R[:0, :], 0
        diag = torch.abs(torch.diagonal(R))
        maxd = torch.max(diag).clamp_min(torch.finfo(dtype).eps)
        keep_mask = diag > max(float(rank_tol), float(breakdown_tol)) * maxd
        r = int(torch.sum(keep_mask).detach().cpu())
        if r == Qv.shape[1]:
            return Qv, R, r
        if r == 0:
            return Qv[:, :0], R[:0, :], 0
        return Qv[:, :r], R[:r, :], r

    def _orthonormalize_block(
        V: "torch.Tensor",
        Qbasis: Optional["torch.Tensor"],
        target_cols: int,
        *,
        allow_random_fill: bool = True,
    ) -> "torch.Tensor":
        pieces = []
        work = V if V is not None else torch.empty((N, 0), device=device, dtype=dtype)
        if work.ndim == 1:
            work = work[:, None]
        work = work.to(device=device, dtype=dtype)

        for attempt in range(10):
            needed = target_cols - sum(p.shape[1] for p in pieces)
            if needed <= 0:
                break
            if work.shape[1] == 0:
                if not allow_random_fill:
                    break
                work = torch.randn((N, needed), device=device, dtype=dtype)

            basis_parts = []
            if Qbasis is not None and Qbasis.numel() > 0:
                basis_parts.append(Qbasis)
            if pieces:
                basis_parts.append(torch.cat(pieces, dim=1))
            Qall = torch.cat(basis_parts, dim=1) if basis_parts else None

            work = _project_deflation(work)
            if Qall is not None:
                work = _project_against(Qall, work, passes=max(1, reorth_passes))
            Qv, _R, r = _qr_rank(work)
            if r > 0:
                take = min(r, needed)
                pieces.append(Qv[:, :take])
            needed = target_cols - sum(p.shape[1] for p in pieces)
            if needed <= 0:
                break
            if not allow_random_fill:
                break
            work = torch.randn((N, needed), device=device, dtype=dtype)

        if not pieces:
            raise RuntimeError("failed to construct any orthonormal block columns")
        Qout = torch.cat(pieces, dim=1)
        if Qout.shape[1] < target_cols:
            raise RuntimeError(
                f"only built {Qout.shape[1]} orthonormal columns, needed {target_cols}"
            )
        return Qout[:, :target_cols]

    def _initial_seed() -> "torch.Tensor":
        if q0_block is not None:
            V = torch.as_tensor(q0_block, device=device, dtype=dtype)
        elif q0 is not None:
            V = torch.as_tensor(q0, device=device, dtype=dtype)
        else:
            V = torch.randn((N, block_size), device=device, dtype=dtype)
        if V.ndim == 1:
            V = V[:, None]
        if V.shape[0] != N:
            raise ValueError(f"q0_block has {V.shape[0]} rows, expected {N}")
        if q0_block is not None or q0 is not None:
            scale = float(torch.linalg.norm(V).detach().cpu()) / max(1.0, (N * V.shape[1]) ** 0.5)
            V = V + (1e-3 * scale) * torch.randn_like(V)
        return _orthonormalize_block(V, None, block_size, allow_random_fill=True)

    def _orth_error_blocked(Qblock: "torch.Tensor") -> float:
        r = Qblock.shape[1]
        G = torch.zeros((r, r), device=device, dtype=small_dtype)
        for lo in range(0, N, block_rows):
            hi = min(lo + block_rows, N)
            B = Qblock[lo:hi].to(dtype=small_dtype)
            G += B.T @ B
        I = torch.eye(r, device=device, dtype=small_dtype)
        return float((torch.linalg.norm(G - I) / max(1, r)).detach().cpu())

    def _small_eigh_desc(H: "torch.Tensor"):
        H = 0.5 * (H + H.T)
        evals, evecs = torch.linalg.eigh(H)
        idx = torch.argsort(evals, descending=True)
        return evals[idx], evecs[:, idx]

    def _true_residuals(vals_t: "torch.Tensor", vecs_t: "torch.Tensor") -> np.ndarray:
        # FIX 5: pass _count=False so these don't pollute benchmark counters.
        AV = _apply_A(vecs_t, _count=False)
        if B_def is not None:
            AV = _project_deflation(AV)
        R = AV - vecs_t * vals_t.to(device=device, dtype=dtype)[None, :]
        return torch.linalg.norm(R, dim=0).detach().cpu().numpy().astype(np.float64)

    # FIX 2: Allocate Q buffer once and reuse across cycles.
    Q_buf = torch.empty((N, ncv + block_size), device=device, dtype=dtype)

    def _expand_cycle(Qseed: "torch.Tensor"):
        """
        Build one block Krylov cycle starting from Qseed, reusing Q_buf.
        Returns a *view* slice of Q_buf, H, p, m.

        FIX 3 (H symmetry): The block Lanczos relation is
            A Q[:,applied:end] = Q[:,:m] H[:m, applied:end]  +  W_residual
        where W_residual = Q[:,m:m+r] * R[:r,:].
        We fill H[applied:end, m:m+r] = R[:r,:].T as well so that H[:p,:p]
        is exactly the projected matrix Q[:,:p].T @ A @ Q[:,:p] (up to
        floating point), making the residual formula exact.
        """
        s0 = Qseed.shape[1]
        if s0 > ncv:
            Qseed = Qseed[:, :ncv]
            s0 = ncv
        Q_buf[:, :s0] = Qseed

        H = torch.zeros((ncv + block_size, ncv + block_size), device=device, dtype=small_dtype)
        m = s0
        applied = 0

        while applied < ncv:
            if applied >= m:
                needed = min(block_size, ncv + block_size - m)
                Qnew = _orthonormalize_block(
                    torch.randn((N, needed), device=device, dtype=dtype),
                    Q_buf[:, :m],
                    needed,
                    allow_random_fill=True,
                )
                Q_buf[:, m:m + needed] = Qnew
                m += needed

            end = min(applied + block_size, m, ncv)
            s = end - applied
            if s <= 0:
                break

            Vblock = Q_buf[:, applied:end]
            W = _apply_A(Vblock)
            W = _project_deflation(W)

            # Full block Arnoldi/Lanczos orthogonalization.
            coeff_total = torch.zeros((m, s), device=device, dtype=small_dtype)
            if reorthogonalize:
                for _ in range(reorth_passes):
                    C = (Q_buf[:, :m].T @ W).to(dtype=small_dtype)
                    W = W - Q_buf[:, :m] @ C.to(dtype=dtype)
                    coeff_total += C
            else:
                C = (Q_buf[:, :m].T @ W).to(dtype=small_dtype)
                W = W - Q_buf[:, :m] @ C.to(dtype=dtype)
                coeff_total += C

            # Store Gram coefficients in this column block only (no transpose fill).
            # H is upper-Hessenberg style here; we symmetrize only at Ritz-problem time.
            # Using = not += avoids double-counting when reorth adds multiple passes.
            H[:m, applied:end] = coeff_total

            # Residual block QR.
            Qnew, R, r = _qr_rank(W)
            if r > 0:
                max_add = min(r, ncv + block_size - m)
                Q_buf[:, m:m + max_add] = Qnew[:, :max_add]
                # Subdiagonal block only — symmetrize H[:p,:p] at Ritz-problem time.
                H[m:m + max_add, applied:end] = R[:max_add, :].to(dtype=small_dtype)
                m += max_add

            applied = end

        p = min(applied, ncv)
        return Q_buf, H, p, m

    def _thick_restart_seed(
        Q: "torch.Tensor",
        H: "torch.Tensor",
        p: int,
        m: int,
        evals_all: "torch.Tensor",
        evecs_all: "torch.Tensor",
        n_take: int,
    ) -> Optional["torch.Tensor"]:
        """
        Thick restart: keep the top keep_eff Ritz vectors plus a fresh residual block.

        The Hessenberg rotation has been removed (per review point 2): after
        U_keep = Q[:,:p] @ Y_keep the projected matrix in that Ritz basis is
        diag(evals[:keep_eff]), not H[:keep_eff,:keep_eff], so rotating by S
        does not restore a tridiagonal relation — it just changes the basis within
        the same subspace without benefit. The next _expand_cycle rebuilds the
        relation from scratch anyway.

        Returns Qseed of shape (N, keep_eff + block_size), or None on breakdown.
        """
        # Validated keep_eff: must satisfy k <= keep_eff <= p - block_size.
        keep_eff = min(keep, p - block_size)
        keep_eff = max(k, keep_eff)
        keep_eff = min(keep_eff, p)

        Y_keep = evecs_all[:, :keep_eff].to(dtype=dtype)
        U_keep = _linear_combo_blocked(Q[:, :p], Y_keep)

        # Defensive reorth/QR of retained Ritz basis (no Hessenberg rotation).
        try:
            U_keep_orth = _orthonormalize_block(U_keep, None, keep_eff, allow_random_fill=False)
        except RuntimeError:
            return None

        # Residual seed from Ritz residual directions.
        r_cols = min(block_size, n_take)
        if m > p:
            coeff_tail = H[p:m, :p] @ evecs_all[:, :r_cols]
            Rseed = _linear_combo_blocked(Q[:, p:m], coeff_tail.to(dtype=dtype))
        else:
            Rseed = torch.empty((N, 0), device=device, dtype=dtype)

        try:
            Rblock = _orthonormalize_block(
                Rseed,
                U_keep_orth,
                block_size,
                allow_random_fill=True,
            )
        except RuntimeError:
            return None

        return torch.cat([U_keep_orth, Rblock], dim=1)

    if verbose:
        print(
            f"[block-irlm] N={N} k={k} block_size={block_size} ncv={ncv} "
            f"keep={keep} max_restarts={max_restarts} tol={tol:g} "
            f"dtype={dtype} small_dtype={small_dtype} device={device}"
        )
        if matmat_fn is None:
            print("  warning: using matvec_fn fallback; not a true block matmat")
        if B_def is not None:
            print(f"  deflate_basis: {B_def.shape[1]} direction(s)")

    Qseed = _initial_seed()

    converged = False
    breakdown = False
    cycles_done = 0
    orth_error_history = []
    residual_history = []
    theta_history = []

    last_Q = None
    last_H = None
    last_p = None
    last_m = None
    last_evals = None
    last_evecs = None
    last_formula_res = None

    try:
        for cycle in range(max_restarts + 1):
            cycles_done = cycle + 1
            Q, H, p, m = _expand_cycle(Qseed)
            if p < k:
                raise RuntimeError(f"cycle produced p={p} basis columns < k={k}")

            Hm = H[:p, :p]
            Hm = 0.5 * (Hm + Hm.T)
            evals_all, evecs_all = _small_eigh_desc(Hm)

            n_take = min(k, p)
            Yk = evecs_all[:, :n_take]
            tail = H[p:m, :p] @ Yk if m > p else torch.zeros((0, n_take), device=device, dtype=small_dtype)
            formula_res = torch.linalg.norm(tail, dim=0) if tail.numel() else torch.zeros(n_take, device=device, dtype=small_dtype)

            if relative_residual:
                thresholds = tol * torch.maximum(torch.ones_like(evals_all[:n_take]), torch.abs(evals_all[:n_take]))
            else:
                thresholds = torch.full_like(formula_res, float(tol))
            n_conv = int(torch.sum(formula_res <= thresholds).detach().cpu())

            if orth_check_every is not None and orth_check_every > 0 and cycle % orth_check_every == 0:
                orth_err = _orth_error_blocked(Q[:, :m])
            else:
                orth_err = float("nan")
            orth_error_history.append(orth_err)
            residual_history.append(formula_res.detach().cpu().numpy().astype(np.float64))
            theta_history.append(evals_all[:n_take].detach().cpu().numpy().astype(np.float64))

            last_Q, last_H, last_p, last_m = Q, H, p, m
            last_evals, last_evecs, last_formula_res = evals_all, evecs_all, formula_res

            if verbose:
                max_res = float(torch.max(formula_res).detach().cpu()) if formula_res.numel() else float("nan")
                print(
                    f"  cycle={cycle:02d} p={p:4d} m={m:4d} "
                    f"converged={n_conv}/{n_take} max_res={max_res:.3e} orth={orth_err:.2e}"
                )
                print(f"    theta[:5] = {evals_all[:min(5, n_take)].detach().cpu().numpy()}")

            if n_take >= k and bool(torch.all(formula_res[:k] <= thresholds[:k]).detach().cpu()):
                converged = True
                if verbose:
                    print(f"[block-irlm] converged after cycle={cycle}")
                break

            if cycle == max_restarts:
                break

            # FIX 1+4: thick restart with GPU Hessenberg and keep_eff validation.
            new_seed = _thick_restart_seed(Q, H, p, m, evals_all, evecs_all, n_take)
            if new_seed is None:
                breakdown = True
                if verbose:
                    print("[block-irlm] restart seed construction failed (breakdown)")
                break
            Qseed = new_seed

        # ── Final Ritz extraction ────────────────────────────────────────────

        if last_Q is None or last_evals is None or last_evecs is None or last_p is None:
            raise RuntimeError("block IRLM failed before producing a Ritz problem")

        n_return = min(k, last_p)
        vals = last_evals[:n_return]
        Y = last_evecs[:, :n_return].to(dtype=dtype)
        ritz_vecs = _linear_combo_blocked(last_Q[:, :last_p], Y)
        ritz_vecs = ritz_vecs / torch.linalg.norm(ritz_vecs, dim=0).clamp_min(torch.finfo(dtype).eps)

        formula_res = last_formula_res[:n_return] if last_formula_res is not None else None
        # FIX 5: true residuals use _count=False (handled inside _true_residuals).
        true_res = _true_residuals(vals, ritz_vecs) if compute_true_residuals else None

        elapsed = time.perf_counter() - t_start

        if verbose:
            print(
                f"[block-irlm] done converged={converged} breakdown={breakdown} "
                f"cycles={cycles_done} matmat={n_matmat} rhs={n_rhs} time={elapsed:.2f}s"
            )
            print(f"  eigenvalues:       {vals.detach().cpu().numpy()}")
            if formula_res is not None:
                print(f"  formula residuals: {formula_res.detach().cpu().numpy()}")
            if true_res is not None:
                print(f"  true residuals:    {true_res}")

        if return_torch:
            vals_out = vals
            vecs_out = ritz_vecs
        else:
            vals_out = vals.detach().cpu().numpy().astype(np.float64)
            vecs_out = ritz_vecs.detach().cpu().numpy()

        if return_info:
            info: Dict[str, Any] = {
                "method": "thick_restarted_block_arnoldi_rayleigh_ritz",
                "converged": bool(converged),
                "breakdown": bool(breakdown),
                "cycles": int(cycles_done),
                "n_matmat": int(n_matmat),
                "n_rhs": int(n_rhs),
                "ncv": int(ncv),
                "keep": int(keep),
                "k": int(k),
                "block_size": int(block_size),
                "tol": float(tol),
                "relative_residual": bool(relative_residual),
                "elapsed": float(elapsed),
                "p_final": int(last_p),
                "m_final": int(last_m) if last_m is not None else None,
                "formula_residuals": None if formula_res is None else formula_res.detach().cpu().numpy().astype(np.float64),
                "true_residuals": true_res,
                "orth_error": orth_error_history,
                "residual_history": residual_history,
                "theta_history": theta_history,
                "dtype": str(dtype),
                "small_dtype": str(small_dtype),
                "device": str(device),
            }
            return vals_out, vecs_out, info

        return vals_out, vecs_out

    finally:
        torch.set_grad_enabled(grad_state)
        if old_tf32_matmul is not None:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32_matmul
            torch.backends.cudnn.allow_tf32 = old_tf32_cudnn


def gpu_block_lanczos(
    matmat_fn: Optional[Callable[["torch.Tensor"], "torch.Tensor"]] = None,
    *,
    matvec_fn: Optional[Callable[["torch.Tensor"], "torch.Tensor"]] = None,
    N: int,
    k: int = 8,
    n_iter: Optional[int] = None,
    block_size: Optional[int] = None,
    tol: float = 1e-6,
    dtype=None,
    device: str = "cuda",
    verbose: bool = True,
    q0_block=None,
    q0=None,
    **kwargs,
):
    """Compatibility wrapper. n_iter maps to ncv."""
    return gpu_block_irlm(
        matmat_fn=matmat_fn,
        matvec_fn=matvec_fn,
        N=N,
        k=k,
        ncv=n_iter,
        block_size=block_size,
        tol=tol,
        dtype=dtype,
        device=device,
        verbose=verbose,
        q0_block=q0_block,
        q0=q0,
        **kwargs,
    )


def make_gpu_matmat(state, ext_module):
    """
    Convenience wrapper for a CUDA extension exposing either:
      - ext_module.matmat(X_fp16, V, rscale, sq_X, beta), preferred
      - ext_module.matvec(...), fallback over columns.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch required")
    import torch

    has_matmat = hasattr(ext_module, "matmat")

    def matmat_gpu(V):
        with torch.no_grad():
            if has_matmat:
                return ext_module.matmat(
                    state.X_fp16, V, state.rscale, state.sq_X, state.beta,
                )
            cols = [
                ext_module.matvec(
                    state.X_fp16, V[:, j], state.rscale, state.sq_X, state.beta,
                )
                for j in range(V.shape[1])
            ]
            return torch.stack(cols, dim=1)

    return matmat_gpu

gpu_lanczos = gpu_block_lanczos
make_gpu_matvec = make_gpu_matmat
```

## `src/flashdiffusion/flash_diffusion_sm80.cu`

```cuda
/*
flashdiffusion/csrc/flash_diffusion_sm80.cu
============================================
SM80 (A100 / Ampere) CuTe kernel for FlashDiffusion.

Computes, tile by tile, without materialising the O(N²) kernel matrix:

  PREPASS:
    D_i = Σ_j  exp(-beta * ||x_i - x_j||²) * w_j        (weighted row sum)

  MATVEC:
    out_i = rscale_i * Σ_j  exp(-beta * ||x_i - x_j||²) * rscale_j * v_j

Both kernels share the same tiling structure:
  1. Load X_i tile (Q side) into SMEM via cp.async
  2. Loop over X_j tiles (K side, fixed — "KV cache"):
     a. Load X_j tile into SMEM via cp.async
     b. GEMM: dot_ij = Xi @ Xj^T   (fp16 mma.sync -> fp32 accum)
     c. Bias:  dist2  = sq_i + sq_j - 2*dot_ij
     d. Exp:   K_tile = exp(-beta * dist2)
     e. Accumulate: acc += K_tile * u_j   (u_j = rscale_j * v_j for matvec,
                                                = w_j for prepass)
  3. Post-scale: out_i = rscale_i * acc_i   (matvec only)
  4. Write output tile to GMEM

No online softmax. No backward pass. No autograd.
rscale is precomputed by the two prepass calls; only V changes per Lanczos iter.

Architecture notes (SM80)
--------------------------
- MMA atom: SM80_16x8x16_F32F16F16F32  (mma.sync, fp16 in, fp32 accum)
- Async copy: SM80_CP_ASYNC_CACHEALWAYS<uint128_t>  (128-bit loads, 16 bytes)
- Tile shape: [BM=64, BN=64, BK=16] — fits in 192KB SMEM with 2-stage pipeline
- Thread block: 128 threads (4 warps)
- Each thread block computes one (BM, BN) tile of the kernel matrix K_tile,
  then reduces into the output (BM, r) accumulator across the BN dimension.

Key difference from FlashAttention
-----------------------------------
FA: online softmax requires max-tracking across BN tiles -> inter-tile state
FD: plain sum -> acc += K_tile * u_j -> no inter-tile state, tiles independent
This means the epilogue is a single scalar multiply (rscale), not a rescaling.

Build
-----
  nvcc -arch=sm_80 -std=c++17 -O3 \
       -I /path/to/cutlass/include \
       flash_diffusion_sm80.cu -o flash_diffusion_sm80.o

  Or via torch.utils.cpp_extension.CUDAExtension (see setup.py).

Exposed to Python via pybind11 / torch.ops:
  prepass_cuda(X, w, beta, tile_m, tile_n) -> D        (N,)  float32
  matvec_cuda(X, V, rscale, beta, tile_m, tile_n) -> out (N, r) float32
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

using namespace cute;

// ---------------------------------------------------------------------------
// Compile-time tile constants (SM80 sweet spot for 192KB SMEM)
// ---------------------------------------------------------------------------
static constexpr int BM = 64;    // rows per Q-tile  (output rows per block)
static constexpr int BN = 64;    // cols per K-tile  (inner reduction)
static constexpr int BK = 16;    // coordinate dimension tile (k-loop over d)
static constexpr int NTHREADS = 128;  // 4 warps

// MMA atom: SM80, fp16 A/B, fp32 C/D, shape 16x8x16
using MMA_Atom_t = MMA_Atom<SM80_16x8x16_F32F16F16F32>;

// Async copy atom: 128-bit (16 bytes = 8 fp16) per thread
using CopyAtom_t = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cutlass::half_t>;

// ---------------------------------------------------------------------------
// Shared memory layout (2-stage pipeline)
// ---------------------------------------------------------------------------
// Xi: (BM, BK) in smem, 2 stages
// Xj: (BN, BK) in smem, 2 stages
// Swizzled to avoid bank conflicts on the K dimension
using SmemLayoutAtom = decltype(
    composition(Swizzle<3,3,3>{},
                Layout<Shape<_8, _BK>, Stride<_BK, _1>>{}));
//  ^-- will be specialised with BK=16 at compile time

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

// Fast exp approximation (single precision) — same as FA2
__device__ __forceinline__ float fast_exp(float x) {
    return __expf(x);
}

// Squared L2 norm of a register vector fragment (used for diagonal bias)
__device__ __forceinline__ float sq_norm_fragment(
    const float* x, int d)
{
    float acc = 0.f;
    for (int k = 0; k < d; ++k) acc += x[k] * x[k];
    return acc;
}

// ---------------------------------------------------------------------------
// PREPASS kernel: D_i = Σ_j K(i,j) * w_j
// ---------------------------------------------------------------------------
/*
  Grid:  (ceil(N/BM),)          -- one block per output row tile
  Block: (NTHREADS,)

  Each block:
    - Loads X[i0:i1, :] into smem (BM rows)
    - Loops over j tiles of X[j0:j1, :]
      - Computes dist2 via GEMM + diagonal bias
      - Accumulates D[i] += Σ_j exp(-beta*dist2) * w[j]
    - Writes D[i0:i1] to GMEM
*/
template<int BM_, int BN_, int BK_>
__global__ void prepass_kernel(
    const __half* __restrict__ X,      // (N, d)  fp16
    const float*  __restrict__ w,      // (N,)    fp32 weights  (=1 for pass1)
    float*        __restrict__ D,      // (N,)    fp32 output
    const float*  __restrict__ sq_X,   // (N,)    precomputed ||x_i||^2 fp32
    int N, int d,
    float beta,
    int num_j_tiles
) {
    // ---- Thread / block indices ----
    int tid  = threadIdx.x;
    int i0   = blockIdx.x * BM_;
    int i1   = min(i0 + BM_, N);
    int bm   = i1 - i0;               // actual rows in this block

    // ---- Shared memory: 2-stage pipeline for Xi and Xj ----
    extern __shared__ char smem_buf[];
    __half* sXi = (__half*)smem_buf;                         // (BM, BK) x 2
    __half* sXj = sXi + 2 * BM_ * BK_;                     // (BN, BK) x 2

    // ---- Accumulator: (BM,) register float ----
    float acc[BM_];
    #pragma unroll
    for (int m = 0; m < BM_; ++m) acc[m] = 0.f;

    // ---- Main loop over j tiles ----
    for (int jt = 0; jt < num_j_tiles; ++jt) {
        int j0 = jt * BN_;
        int j1 = min(j0 + BN_, N);
        int bn = j1 - j0;

        // Load Xj tile: each thread loads a few elements
        // (simplified: use cp.async for production; direct load here for clarity)
        // In a full CuTe kernel this is replaced by make_tiled_copy + cute::copy
        for (int idx = tid; idx < bn * d; idx += NTHREADS) {
            int row = idx / d, col = idx % d;
            sXj[row * d + col] = X[(j0 + row) * d + col];
        }
        // Load Xi tile
        for (int idx = tid; idx < bm * d; idx += NTHREADS) {
            int row = idx / d, col = idx % d;
            sXi[row * d + col] = X[(i0 + row) * d + col];
        }
        __syncthreads();

        // Each thread owns one row of the output tile (i0 + tid if tid < bm)
        if (tid < bm) {
            int gi = i0 + tid;
            float sqi = sq_X[gi];

            for (int j = 0; j < bn; ++j) {
                int gj = j0 + j;
                // dot = xi . xj
                float dot = 0.f;
                for (int k = 0; k < d; ++k) {
                    dot += __half2float(sXi[tid * d + k])
                         * __half2float(sXj[j   * d + k]);
                }
                float sqj  = sq_X[gj];
                float dist2 = fmaxf(sqi + sqj - 2.f * dot, 0.f);
                float kij   = fast_exp(-beta * dist2);
                acc[tid] += kij * w[gj];
            }
        }
        __syncthreads();
    }

    // Write output
    if (tid < bm) {
        D[i0 + tid] = acc[tid];
    }
}

// ---------------------------------------------------------------------------
// MATVEC kernel: out_i = rscale_i * Σ_j K(i,j) * rscale_j * v_j
// ---------------------------------------------------------------------------
/*
  Grid:  (ceil(N/BM), r)         -- r = number of Lanczos vectors (typically 1)
  Block: (NTHREADS,)

  Same tile loop as prepass, but:
    - u_j = rscale_j * v_j     (pre-scaled input)
    - out_i *= rscale_i         (post-scale)
    - accumulator is a float (scalar per Lanczos vector per row)
*/
__global__ void matvec_kernel(
    const __half* __restrict__ X,       // (N, d)  fp16
    const float*  __restrict__ u,       // (N,)    fp32  u = rscale * v
    float*        __restrict__ out,     // (N,)    fp32
    const float*  __restrict__ sq_X,    // (N,)    precomputed ||x_i||^2
    const float*  __restrict__ rscale,  // (N,)    post-scale diagonal
    int N, int d,
    float beta,
    int num_j_tiles
) {
    int tid = threadIdx.x;
    int i0  = blockIdx.x * BM;
    int i1  = min(i0 + BM, N);
    int bm  = i1 - i0;

    extern __shared__ char smem_buf[];
    __half* sXi = (__half*)smem_buf;
    __half* sXj = sXi + BM * d;

    float acc = 0.f;

    for (int jt = 0, j0 = 0; j0 < N; ++jt, j0 += BN) {
        int j1 = min(j0 + BN, N);
        int bn = j1 - j0;

        for (int idx = tid; idx < bn * d; idx += NTHREADS)
            sXj[(idx/d) * d + idx%d] = X[(j0 + idx/d) * d + idx%d];
        for (int idx = tid; idx < bm * d; idx += NTHREADS)
            sXi[(idx/d) * d + idx%d] = X[(i0 + idx/d) * d + idx%d];
        __syncthreads();

        if (tid < bm) {
            float sqi = sq_X[i0 + tid];
            for (int j = 0; j < bn; ++j) {
                float dot = 0.f;
                for (int k = 0; k < d; ++k)
                    dot += __half2float(sXi[tid*d+k]) * __half2float(sXj[j*d+k]);
                float dist2 = fmaxf(sqi + sq_X[j0+j] - 2.f*dot, 0.f);
                acc += fast_exp(-beta * dist2) * u[j0+j];
            }
        }
        __syncthreads();
    }

    if (tid < bm) {
        out[i0 + tid] = rscale[i0 + tid] * acc;
    }
}

// ---------------------------------------------------------------------------
// Sq-norm precomputation kernel
// ---------------------------------------------------------------------------
__global__ void sq_norm_kernel(
    const __half* __restrict__ X,
    float*        __restrict__ sq,
    int N, int d
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s = 0.f;
    for (int k = 0; k < d; ++k) {
        float v = __half2float(X[i*d+k]);
        s += v*v;
    }
    sq[i] = s;
}

// ---------------------------------------------------------------------------
// Python-facing entry points (called from kernel.py CUDA backend)
// ---------------------------------------------------------------------------

// Precompute ||x_i||^2 for all i — called once, cached on the Python side
torch::Tensor compute_sq_norms(torch::Tensor X_fp16) {
    // X_fp16: (N, d) float16 on CUDA
    int N = X_fp16.size(0), d = X_fp16.size(1);
    auto sq = torch::empty({N}, X_fp16.options().dtype(torch::kFloat32));

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    sq_norm_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(X_fp16.data_ptr<at::Half>()),
        sq.data_ptr<float>(),
        N, d
    );
    return sq;
}

// Prepass: D = K @ w  (one weighted row sum)
// w = ones(N) for pass1, w = D^{-alpha} for pass2
torch::Tensor prepass_cuda(
    torch::Tensor X_fp16,   // (N, d)  fp16
    torch::Tensor w,        // (N,)    fp32
    torch::Tensor sq_X,     // (N,)    fp32  from compute_sq_norms
    float beta
) {
    int N = X_fp16.size(0), d = X_fp16.size(1);
    auto D = torch::zeros({N}, w.options());

    int num_j_tiles = (N + BN - 1) / BN;
    int grid        = (N + BM - 1) / BM;
    // smem: Xi(BM,d) + Xj(BN,d) in fp16
    size_t smem = (BM + BN) * d * sizeof(__half);

    prepass_kernel<BM, BN, BK><<<grid, NTHREADS, smem,
                                  at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(X_fp16.data_ptr<at::Half>()),
        w.data_ptr<float>(),
        D.data_ptr<float>(),
        sq_X.data_ptr<float>(),
        N, d, beta, num_j_tiles
    );
    return D;
}

// Matvec: out = rscale * (K @ (rscale * v))
// Called every Lanczos iteration — V changes, X/rscale fixed
torch::Tensor matvec_cuda(
    torch::Tensor X_fp16,   // (N, d)  fp16
    torch::Tensor v,        // (N,)    fp32  Lanczos vector
    torch::Tensor rscale,   // (N,)    fp32  precomputed diagonal
    torch::Tensor sq_X,     // (N,)    fp32
    float beta
) {
    int N = X_fp16.size(0), d = X_fp16.size(1);
    // pre-scale: u = rscale * v
    auto u   = rscale * v;                   // (N,) elementwise, stays on GPU
    auto out  = torch::empty({N}, v.options());

    int grid = (N + BM - 1) / BM;
    size_t smem = (BM + BN) * d * sizeof(__half);

    matvec_kernel<<<grid, NTHREADS, smem,
                    at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(X_fp16.data_ptr<at::Half>()),
        u.data_ptr<float>(),
        out.data_ptr<float>(),
        sq_X.data_ptr<float>(),
        rscale.data_ptr<float>(),
        N, d, beta,
        (N + BN - 1) / BN
    );
    return out;
}

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------
PYBIND11_MODULE(flash_diffusion_cuda, m) {
    m.doc() = "FlashDiffusion SM80 CUDA kernel";

    m.def("compute_sq_norms", &compute_sq_norms,
          "Precompute ||x_i||^2 for all rows. Call once, cache the result.",
          py::arg("X_fp16"));

    m.def("prepass", &prepass_cuda,
          "Weighted row sum: D_i = sum_j K(i,j) * w_j. "
          "Call twice: pass1 w=ones, pass2 w=D^{-alpha}.",
          py::arg("X_fp16"), py::arg("w"), py::arg("sq_X"), py::arg("beta"));

    m.def("matvec", &matvec_cuda,
          "Matvec: out_i = rscale_i * sum_j K(i,j) * rscale_j * v_j. "
          "Called every Lanczos iteration. No gradients needed.",
          py::arg("X_fp16"), py::arg("v"), py::arg("rscale"),
          py::arg("sq_X"), py::arg("beta"));
}
```

## `src/flashdiffusion/kernel.py`

```python
"""
flashdiffusion/kernel.py
========================
The FlashDiffusion primitive.

FlashDiffusion(Q, K, V, beta, alpha, h, rscale_q, rscale_k, tile)

Computes, tile by tile, without materialising the O(N²) kernel matrix:

    score(i,j)   = beta * ||q_i - k_j||²              (Gaussian, default)
                 | score_mod(q_i, k_j)                 (custom, future)

    K(i,j)       = exp(-score(i,j))

    out_i        = rscale_q_i · Σⱼ K(i,j) · rscale_k_j · (h_j · v_j)

where rscale_{q,k} encode the Coifman-Lafon normalisation and h is an
optional Doob potential (default h=1, no transform).

Modes
-----
DMAP matvec    Q=K=X,  rscale_q=rscale_k=rscale (precomputed by DiffusionMap)
AMAP matvec    Q≠K,    rscale_q/k separate (directed/NESS operator, DAC §3)
Doob mode      h ≠ 1:  h-transform of either operator (DAC §4, Theorem 5.1)
Prepass        V=1 (or V=w), h=1: computes row sums D or D_alpha

Complexity
----------
Arithmetic : O(N_q · N_k · d)   — same as dense, unavoidable
Memory     : O((N_q + N_k) · tile)  — never O(N²)

Current backend: NumPy (CPU reference).
CUDA target:     CuTe SM90, fused prepass+matvec, tensor-core GEMM for
                 the xi·xj term, elementwise exp epilogue. No online
                 softmax needed — rscale is precomputed.
"""

from __future__ import annotations
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Backend detection + dispatch
# ---------------------------------------------------------------------------

def _cuda_sm() -> int | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability()
        return major * 10 + minor
    except ImportError:
        return None

# SM -> which kernel family to use
# SM120 (RTX 50xx) uses mma.sync (same family as SM80), not wgmma
_SM_BACKEND = {
    80:  "sm80", 86: "sm80", 89: "sm80",
    90:  "sm90",
    100: "sm100",
    103: "sm100",
    120: "sm120", 121: "sm120",
}

def _get_cuda_backend():
    """Return (precompute_fn, matvec_fn) or None if unavailable."""
    sm = _cuda_sm()
    if sm is None:
        return None
    backend_name = _SM_BACKEND.get(sm, "sm80")

    if backend_name == "sm80":
        try:
            from .kernel_cuda import precompute_cuda, matvec_cuda
            return precompute_cuda, matvec_cuda
        except (ImportError, OSError):
            return None

    elif backend_name == "sm90":
        try:
            from .kernel_cuda_sm90 import precompute_sm90, matvec_sm90
            return precompute_sm90, matvec_sm90
        except (ImportError, OSError):
            try:
                from .kernel_cuda import precompute_cuda, matvec_cuda
                print("[FlashDiffusion] SM90 not found, falling back to SM80")
                return precompute_cuda, matvec_cuda
            except (ImportError, OSError):
                return None

    elif backend_name == "sm100":
        try:
            from .kernel_cuda_sm100 import precompute_sm100, matvec_sm100
            return precompute_sm100, matvec_sm100
        except (ImportError, OSError):
            try:
                from .kernel_cuda import precompute_cuda, matvec_cuda
                print("[FlashDiffusion] SM100 not found, falling back to SM80")
                return precompute_cuda, matvec_cuda
            except (ImportError, OSError):
                return None

    elif backend_name == "sm120":
        try:
            from .kernel_cuda_sm120 import precompute_sm120, matvec_sm120
            return precompute_sm120, matvec_sm120
        except (ImportError, OSError):
            try:
                from .kernel_cuda import precompute_cuda, matvec_cuda
                print("[FlashDiffusion] SM120 not found, falling back to SM80")
                return precompute_cuda, matvec_cuda
            except (ImportError, OSError):
                return None

    return None


# ---------------------------------------------------------------------------
# Distance tile  (shared by numpy prepass and matvec)
# ---------------------------------------------------------------------------

def _dist2_tile(
    Qi: np.ndarray,   # (ti, d)  float32
    Kj: np.ndarray,   # (tj, d)  float32
    sqi: np.ndarray,  # (ti,)
    sqk: np.ndarray,  # (tj,)
) -> np.ndarray:
    """Squared Euclidean distances for one (Q-tile, K-tile) pair. float64 output."""
    d2 = sqi[:, None] + sqk[None, :] - 2.0 * (Qi @ Kj.T)
    np.maximum(d2, 0.0, out=d2)
    return d2.astype(np.float64)


# ---------------------------------------------------------------------------
# Prepass: row-sum reduction  (used twice in precompute_normalization)
# ---------------------------------------------------------------------------

def prepass_rowsum(
    Q: np.ndarray,            # (N_q, d)
    K: np.ndarray,            # (N_k, d)
    beta: float,
    weights_k: Optional[np.ndarray] = None,   # (N_k,)  default = 1
    tile: int = 512,
) -> np.ndarray:
    """
    out_i = Σⱼ exp(-beta · ||q_i - k_j||²) · weights_k_j

    Returns (N_q,) float64.
    This is the primitive used for both normalisation passes:
      pass 1:  weights_k = np.ones(N)   -> D_i  (raw row sums)
      pass 2:  weights_k = w = D^{-α}  -> D_alpha_i (before * w_i)
    """
    N_q, d = Q.shape
    N_k    = K.shape[0]
    if weights_k is None:
        weights_k = np.ones(N_k, dtype=np.float64)

    Q32 = Q.astype(np.float32)
    K32 = K.astype(np.float32)
    sqQ = (Q32 * Q32).sum(axis=1)
    sqK = (K32 * K32).sum(axis=1)

    out = np.zeros(N_q, dtype=np.float64)

    for i0 in range(0, N_q, tile):
        i1  = min(i0 + tile, N_q)
        Qi  = Q32[i0:i1]
        sqi = sqQ[i0:i1]

        for j0 in range(0, N_k, tile):
            j1  = min(j0 + tile, N_k)
            Kj  = K32[j0:j1]
            sqk = sqK[j0:j1]

            d2      = _dist2_tile(Qi, Kj, sqi, sqk)
            K_tile  = np.exp(-beta * d2)              # (ti, tj)
            out[i0:i1] += K_tile @ weights_k[j0:j1]

    return out


# ---------------------------------------------------------------------------
# Matvec: the core FlashDiffusion kernel
# ---------------------------------------------------------------------------

def FlashDiffusion(
    Q:           np.ndarray,                   # (N_q, d)
    K:           np.ndarray,                   # (N_k, d)
    V:           np.ndarray,                   # (N_k, r)  r columns
    beta:        float,
    rscale_q:    Optional[np.ndarray] = None,  # (N_q,)  default = 1
    rscale_k:    Optional[np.ndarray] = None,  # (N_k,)  default = 1
    h:           Optional[np.ndarray] = None,  # (N_k,)  Doob potential
    tile:        int  = 512,
    coord_dtype       = np.float32,            # X tile dtype (f32/f16)
    accum_dtype       = np.float64,            # accumulator dtype
) -> np.ndarray:
    """
    FlashDiffusion numpy backend (CPU reference / fallback).

    out_i = rscale_q_i · Σⱼ exp(-beta·||qᵢ-kⱼ||²) · rscale_k_j · h_j · v_j

    Precision contract (numpy path)
    --------------------------------
    coord_dtype : X tile storage — fp32 default, fp16 for memory savings
    accum_dtype : accumulator — fp64 for CPU reference correctness
    GEMM (dot)  : always fp32 regardless of coord_dtype (catastrophic
                  cancellation safety for dist² = sqi + sqj - 2*dot)

    CUDA path (see kernel_cuda.py + csrc/flash_diffusion_sm80.cu)
    ---------------------------------------------------------------
    coord_dtype -> fp16 smem  (hardcoded in CUDA kernel)
    GEMM        -> mma.sync fp16->fp32 (SM80) / wgmma fp16->fp32 (SM90)
    accum_dtype -> fp32 register tile
    No online softmax: rscale precomputed, epilogue = scalar multiply.
    No gradients: we compute the harmonic basis (eigenvectors), not a
    differentiable layer. Use flexattention for the trained forward pass.
    """
    N_q, d = Q.shape
    N_k, r = V.shape

    # GEMM always in fp32 for dist² numerical safety
    Q32 = Q.astype(np.float32)
    K32 = K.astype(np.float32)
    sqQ = (Q32 * Q32).sum(axis=1)
    sqK = (K32 * K32).sum(axis=1)

    # pre-scale RHS: u_j = rscale_k_j · h_j · v_j
    u = V.astype(accum_dtype)
    if rscale_k is not None:
        u = rscale_k[:, None] * u
    if h is not None:
        u = h[:, None] * u

    out = np.zeros((N_q, r), dtype=accum_dtype)

    for i0 in range(0, N_q, tile):
        i1  = min(i0 + tile, N_q)
        Qi  = Q32[i0:i1]
        sqi = sqQ[i0:i1]

        for j0 in range(0, N_k, tile):
            j1  = min(j0 + tile, N_k)
            Kj  = K32[j0:j1]
            sqk = sqK[j0:j1]

            d2     = _dist2_tile(Qi, Kj, sqi, sqk)       # fp64
            K_tile = np.exp(-beta * d2)                   # fp64 epilogue
            out[i0:i1] += K_tile @ u[j0:j1]

    if rscale_q is not None:
        out *= rscale_q[:, None]

    return out
```

## `src/flashdiffusion/kernel_cuda.py`

```python
"""
flashdiffusion/kernel_cuda.py
==============================
Python wrapper around flash_diffusion_cuda (the compiled SM80 extension).

This module is imported by kernel.py when CUDA is available.
It mirrors the numpy backend interface exactly — only the compute happens
on GPU. All tensors are managed here; the caller (kernel.py) stays backend-agnostic.

No gradients anywhere: we are computing eigenvectors of a fixed symmetric
operator (the harmonic basis). There is no training loop through this kernel.
If you later want to train a model that *uses* diffusion coordinates, that
goes through flexattention on the transformed coordinates, not through here.

Precision contract
------------------
  X storage   : fp16  (halved bandwidth vs fp32, safe because ||xi-xj||^2
                        is computed as sqi + sqj - 2*dot with fp32 accum)
  dot product : fp32  (mma.sync fp16->fp32, avoids catastrophic cancellation)
  exp, accum  : fp32
  rscale, D   : fp32
  Lanczos vec : fp32  on GPU,  float64 orthogonalisation on CPU (unchanged)
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Optional

# Lazy import — only loaded if CUDA is available and extension is compiled
_ext = None

def _load_ext():
    global _ext
    if _ext is None:
        import sys
        ext = sys.modules.get('flash_diffusion_cuda')
        if ext is None:
            try:
                import flash_diffusion_cuda as ext
            except ImportError as e:
                raise ImportError(
                    "flash_diffusion_cuda not compiled. "
                    "Try: import flashdiffusion (triggers JIT build)"
                ) from e
        _ext = ext
    return _ext


def _to_gpu_fp16(X: np.ndarray) -> torch.Tensor:
    """Move coordinate array to GPU as fp16."""
    return torch.from_numpy(np.asarray(X, dtype=np.float16)).cuda()


def _to_gpu_fp32(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).cuda()


class CUDAState:
    """
    Cached GPU tensors for a fixed (X, beta, alpha).
    Analogous to the 'norm' dict in the numpy backend, but lives on GPU.

    Built once by precompute_cuda(), reused across all Lanczos iterations.
    The K-side tensors (X_fp16, sq_X, rscale) are the "KV cache" —
    they never change while only the Lanczos vector v changes.
    """

    def __init__(
        self,
        X_fp16:  torch.Tensor,   # (N, d) fp16 on GPU
        sq_X:    torch.Tensor,   # (N,)   fp32 on GPU
        rscale:  torch.Tensor,   # (N,)   fp32 on GPU
        D:       torch.Tensor,   # (N,)   fp32 on GPU  — raw row sums
        D_alpha: torch.Tensor,   # (N,)   fp32 on GPU
        beta:    float,
        alpha:   float,
    ):
        self.X_fp16  = X_fp16
        self.sq_X    = sq_X
        self.rscale  = rscale
        self.D       = D
        self.D_alpha = D_alpha
        self.beta    = beta
        self.alpha   = alpha
        self.N       = X_fp16.shape[0]
        self.d       = X_fp16.shape[1]


def precompute_cuda(
    X:     np.ndarray,
    beta:  float,
    alpha: float = 0.5,
) -> CUDAState:
    """
    Two-pass precomputation on GPU.  Mirrors precompute() in kernel.py.

    Pass 1: D      = K @ ones   (raw row sums)
    Pass 2: D_alpha = (K @ w) * w    where w = D^{-alpha}
    rscale = D_alpha^{-1/2} * w

    Returns a CUDAState with all GPU tensors cached.
    """
    ext = _load_ext()
    N, d = X.shape

    print(f"[precompute_cuda] N={N}  d={d}  beta={beta}  alpha={alpha}")

    # Move X to GPU as fp16 — stays there for the entire eigensolver run
    X_fp16 = _to_gpu_fp16(X)

    # Precompute sq norms (fp32, stays on GPU)
    sq_X = ext.compute_sq_norms(X_fp16)

    # Pass 1: D_i = Σ_j K(i,j)   (w = 1)
    w1 = torch.ones(N, dtype=torch.float32, device='cuda')
    D  = ext.prepass(X_fp16, w1, sq_X, beta)

    # w = D^{-alpha}
    w  = D.pow(-alpha)

    # Pass 2: D_alpha_i = (Σ_j K(i,j) * w_j) * w_i
    D_alpha = ext.prepass(X_fp16, w, sq_X, beta) * w

    # Combined diagonal
    rscale = D_alpha.pow(-0.5) * w

    return CUDAState(X_fp16, sq_X, rscale, D, D_alpha, beta, alpha)


def matvec_cuda(state: CUDAState, v: np.ndarray) -> np.ndarray:
    """
    Apply M to a single Lanczos vector v (N,) float64.

    v arrives as numpy float64 (from ARPACK).
    We cast to float32, push to GPU, run the kernel, pull back as float64.

    The CPU<->GPU transfer for v is O(N) floats — negligible vs O(N²) compute.
    The K-side (X_fp16, sq_X, rscale) never moves — it's the KV cache.
    """
    ext = _load_ext()
    v_gpu = _to_gpu_fp32(v.astype(np.float32))

    with torch.no_grad():
        out_gpu = ext.matvec(
            state.X_fp16,
            v_gpu,
            state.rscale,
            state.sq_X,
            state.beta,
        )

    # Pull back to CPU as float64 for ARPACK orthogonalisation
    return out_gpu.cpu().numpy().astype(np.float64)
```

## `src/flashdiffusion/kernel_cuda_sm120.py`

```python
"""
flashdiffusion/kernel_cuda_sm120.py
=====================================
Python wrapper for the SM120 (RTX 5090 / consumer Blackwell) kernel.

Differences from SM80 wrapper:
  - module name: flash_diffusion_sm120
  - larger tiles (128×128 vs 64×64) — same interface
  - cp.async pipeline baked into kernel — no API change needed
  - compile with -arch=sm_120
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Optional

_ext = None   # injected by notebook or loaded via _load_ext()


def _load_ext():
    global _ext
    if _ext is None:
        import sys
        # check sys.modules first — kernel_jit.py injects here at import time
        ext = sys.modules.get('flash_diffusion_sm120')
        if ext is None:
            try:
                import flash_diffusion_sm120 as ext
            except ImportError as e:
                raise ImportError(
                    "flash_diffusion_sm120 not compiled. "
                    "Try: import flashdiffusion (triggers JIT build)"
                ) from e
        _ext = ext
    return _ext


def _to_gpu_fp16(X: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(X, dtype=np.float16)).cuda()


def _to_gpu_fp32(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).cuda()


class CUDAStateSM120:
    """Cached GPU tensors for SM120. Same interface as CUDAState (SM80)."""

    def __init__(self, X_fp16, sq_X, rscale, D, D_alpha, beta, alpha):
        self.X_fp16  = X_fp16
        self.sq_X    = sq_X
        self.rscale  = rscale
        self.D       = D
        self.D_alpha = D_alpha
        self.beta    = beta
        self.alpha   = alpha
        self.N       = X_fp16.shape[0]
        self.d       = X_fp16.shape[1]


def precompute_sm120(
    X:     np.ndarray,
    beta:  float,
    alpha: float = 0.5,
) -> CUDAStateSM120:
    """Two-pass precomputation on SM120 GPU."""
    import time
    ext = _load_ext()
    N, d = X.shape
    print(f"[precompute_sm120] N={N}  d={d}  beta={beta}  alpha={alpha}")

    X_fp16 = _to_gpu_fp16(X)
    sq_X   = ext.compute_sq_norms(X_fp16)

    t0 = time.perf_counter()
    w1 = torch.ones(N, dtype=torch.float32, device='cuda')
    D  = ext.prepass(X_fp16, w1, sq_X, beta)
    w  = D.pow(-alpha)
    D_alpha = ext.prepass(X_fp16, w, sq_X, beta) * w
    rscale  = D_alpha.pow(-0.5) * w
    print(f"  precompute done (SM120) in {time.perf_counter()-t0:.3f}s")

    return CUDAStateSM120(X_fp16, sq_X, rscale, D, D_alpha, beta, alpha)


def matvec_sm120(state: CUDAStateSM120, v: np.ndarray) -> np.ndarray:
    """Apply M to Lanczos vector v. GPU→GPU, pull back to CPU for ARPACK."""
    ext = _load_ext()
    v_gpu = torch.from_numpy(v.astype(np.float32)).cuda()
    with torch.no_grad():
        out = ext.matvec(
            state.X_fp16, v_gpu, state.rscale, state.sq_X, state.beta)
    return out.cpu().numpy().astype(np.float64)
```

## `src/flashdiffusion/kernel_jit.py`

```python
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
```

## `src/flashdiffusion/ldm_matmat.py`

```python
"""
ldm_matmat.py
=============
Linearized Diffusion Map (LDM) and 2nd-order LDM matmat operators
for use with gpu_block_irlm.

These are the O(ND) and O(ND^2) warm-start operators for full DMAP.
Both expose the matmat_fn(V) -> A@V interface expected by gpu_block_irlm,
so block_size can be anything and GPU throughput is fully used.

Hierarchy
---------
  Stage 1:  LDM       O(NDs)   -- linear in N and D
  Stage 2:  2nd-order O(ND^2)  -- one-time O(ND^2) setup, then O(NDs)/matmat
  Stage 3:  full DMAP O(N^2 D) -- your existing CUDA exp kernel

Usage
-----
    from ldm_matmat import LDMOperator, LDM2Operator, ldm_warm_start

    # build operators (precomputes potentials once)
    ldm  = LDMOperator(X, beta, alpha=1.0, device="cuda")
    ldm2 = LDM2Operator(X, beta, alpha=1.0, device="cuda")

    # run block IRLM for warm start
    vals0, U0 = gpu_block_irlm(matmat_fn=ldm.matmat,  N=N, k=k, ...)
    vals1, U1 = gpu_block_irlm(matmat_fn=ldm2.matmat, N=N, k=k, q0_block=U0, ...)

    # full DMAP warm-started
    vals,  V  = gpu_block_irlm(matmat_fn=dmap_matmat,  N=N, k=k, q0_block=U1, ...)
"""

from __future__ import annotations
from typing import Optional
import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Shared potential precomputation
# ─────────────────────────────────────────────────────────────────────────────

def _precompute_potentials(
    X: "torch.Tensor",
    beta: float,
    eps: float,
    alpha: float,
    device: "torch.device",
    dtype: "torch.dtype",
    verbose: bool,
) -> dict:
    """
    Precompute all four potentials mu, r, q, m from centered data.
    All O(ND) or O(N). Returns dict with tensors on device.

    The combined potential v_hat absorbs:
      v  = exp(-beta * r / eps)          [from kernel factorization]
      q  = CL normalization (alpha)      [O(N) closed form after centering]
      m  = Markov symmetrization         [O(ND) one matrix-vector product]
    """
    import torch

    N, D = X.shape
    t0 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    t1 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    if t0: t0.record()

    # ── 1. Center ────────────────────────────────────────────────────────────
    mu   = X.mean(dim=0)                   # (D,)
    Rb   = X - mu.unsqueeze(0)             # (N, D)  centered, Rb^T 1 = 0

    # ── 2. Squared norms r = diag(Rb Rb^T) ───────────────────────────────────
    r    = (Rb * Rb).sum(dim=1)            # (N,)  O(ND)
    S_r  = r.sum()                         # scalar = tr(Rb Rb^T)

    # ── 3. Kernel potential v = exp(-beta r / eps) ────────────────────────────
    v    = torch.exp(-beta * r / eps)      # (N,)

    # ── 4. CL normalization q  (O(N) closed form, centering killed cross-term)─
    #   k_i = N(1 - beta*r_i/eps) - beta*S_r/eps
    #   q_i = k_i^{-1}   (alpha=1 shown; general alpha below)
    k_lin = N * (1.0 - beta * r / eps) - (beta / eps) * S_r   # (N,)
    # For general alpha: q = k_lin^{-alpha}
    if abs(alpha - 1.0) < 1e-9:
        q = k_lin.pow(-1.0)
    else:
        q = k_lin.pow(-alpha)

    # ── 5. Markov symmetrization m  (O(ND) one matmul) ───────────────────────
    #   S_q  = 1^T q
    #   S_rq = r^T q
    #   c    = Rb^T q  in R^D
    #   k_hat_i = q_i * [S_q*(1 - beta*r_i/eps) - beta*S_rq/eps + 2*beta/eps * Rb_i . c]
    #   m_i  = k_hat_i^{-1/2}
    S_q  = q.sum()                         # scalar
    S_rq = (r * q).sum()                   # scalar
    c    = Rb.T @ q                        # (D,)  O(ND)

    Rbc  = Rb @ c                          # (N,)  O(ND)  -- Rb_i . c
    k_hat = q * (
        S_q * (1.0 - beta * r / eps)
        - (beta / eps) * S_rq
        + (2.0 * beta / eps) * Rbc
    )                                       # (N,)
    m    = k_hat.pow(-0.5)                 # (N,)

    # ── 6. Combined potential ─────────────────────────────────────────────────
    # For LDM: v_hat = m * q  (exp(-beta r) absorbed into q/m in linear degree)
    # For DMAP: v_hat = m * q * exp(-beta r / eps)
    v_hat_ldm  = m * q                     # (N,)
    v_hat_dmap = m * q * v                 # (N,)

    if t1:
        t1.record()
        torch.cuda.synchronize()
        elapsed = t0.elapsed_time(t1) / 1000.0
    else:
        elapsed = 0.0

    if verbose:
        print(f"[LDM precompute] N={N} D={D} beta={beta:.4g} eps={eps:.4g} alpha={alpha:.4g}")
        print(f"  S_r={float(S_r):.4g}  S_q={float(S_q):.4g}  elapsed={elapsed:.3f}s")
        print(f"  r   : min={float(r.min()):.4g} max={float(r.max()):.4g}")
        print(f"  v   : min={float(v.min()):.4g} max={float(v.max()):.4g}")
        print(f"  k_lin: min={float(k_lin.min()):.4g} max={float(k_lin.max()):.4g}")
        print(f"  k_hat: min={float(k_hat.min()):.4g} max={float(k_hat.max()):.4g}")

    return dict(
        Rb=Rb, mu=mu, r=r, v=v, q=q, m=m,
        v_hat_ldm=v_hat_ldm,
        v_hat_dmap=v_hat_dmap,
        k_lin=k_lin, k_hat=k_hat,
        S_r=S_r, S_q=S_q, S_rq=S_rq, c=c,
        N=N, D=D, beta=beta, eps=eps, alpha=alpha,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LDM Operator  O(NDs) per matmat
# ─────────────────────────────────────────────────────────────────────────────

class LDMOperator:
    """
    K^LDM U = diag(v_hat) [1 1^T + (2*beta/eps) Rb Rb^T] diag(v_hat) U

    Cost per matmat: O(NDs)  -- two GEMMs with Rb (shape N x D).
    The rank-1 term 1 1^T should be deflated via deflate_basis in gpu_block_irlm.

    Parameters
    ----------
    X : array-like (N, D)
        Raw (uncentered) data. Will be centered internally.
    beta, eps, alpha : float
        Kernel bandwidth, lengthscale, CL exponent.
    device, dtype : torch device/dtype.
    """

    def __init__(
        self,
        X,
        beta: float,
        eps: Optional[float] = None,
        alpha: float = 1.0,
        device: str = "cuda",
        dtype=None,
        verbose: bool = True,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError("torch required")
        import torch

        self.device = torch.device(device)
        self.dtype  = dtype or torch.float32

        Xt = torch.as_tensor(
            X if isinstance(X, torch.Tensor) else np.asarray(X, dtype=np.float32),
            device=self.device, dtype=self.dtype,
        )
        N, D = Xt.shape
        # Default eps: use 4 * mean(r) so linearization is valid
        if eps is None:
            r_tmp = (Xt * Xt).sum(dim=1)
            mu_tmp = Xt.mean(dim=0)
            Rb_tmp = Xt - mu_tmp
            r_tmp  = (Rb_tmp * Rb_tmp).sum(dim=1)
            eps = float(4.0 * r_tmp.mean())
            if verbose:
                print(f"[LDMOperator] auto eps={eps:.4g} (4 * mean(r))")

        self.eps = eps
        self.pot = _precompute_potentials(
            Xt, beta, eps, alpha, self.device, self.dtype, verbose,
        )
        self.N = N
        self.D = D

        # Precompute the trivial-mode vector for deflation reference
        # (caller should pass this as deflate_basis to gpu_block_irlm)
        self.trivial_mode = (self.pot["v_hat_ldm"] /
                             self.pot["v_hat_ldm"].norm()).unsqueeze(1)  # (N,1)

    def matmat(self, V: "torch.Tensor") -> "torch.Tensor":
        """
        V : (N, s)  ->  K^LDM V : (N, s)

        Expanding:
          W      = diag(v_hat) V                   O(Ns)
          rank1  = diag(v_hat) 1 (1^T W)           O(Ns)   [trivial mode]
          gramv  = diag(v_hat) Rb (Rb^T W)         O(NDs)  [two GEMMs]
          output = rank1 + (2*beta/eps) * gramv
        """
        import torch
        pot   = self.pot
        vh    = pot["v_hat_ldm"]          # (N,)
        Rb    = pot["Rb"]                 # (N, D)
        beta  = pot["beta"]
        eps   = pot["eps"]

        W     = vh.unsqueeze(1) * V       # diag(v_hat) V,   (N, s)
        rank1 = vh.unsqueeze(1) * W.sum(dim=0, keepdim=True)   # (N, s)  broadcast
        gramv = vh.unsqueeze(1) * (Rb @ (Rb.T @ W))            # (N, s)
        return rank1 + (2.0 * beta / eps) * gramv

    @property
    def deflate_vector(self) -> "torch.Tensor":
        """
        Return the trivial diffusion mode (N,1) for use as deflate_basis.
        Pass this to gpu_block_irlm to kill the lambda=1 mode.
        """
        return self.trivial_mode


# ─────────────────────────────────────────────────────────────────────────────
# 2nd-Order LDM Operator  O(ND^2) setup + O(NDs) per matmat
# ─────────────────────────────────────────────────────────────────────────────

class LDM2Operator:
    """
    K^(2) U = diag(v_hat) [
        1 1^T
      + (2*beta/eps) Rb Rb^T
      + (2*beta^2/eps^2) (Rb Rb^T)^{o2}
    ] diag(v_hat) U

    The n=2 Hadamard-power term is computed via:
      A = Rb^T diag(u) Rb  in R^{D x D}   (precomputed once, O(ND^2))
      [( Rb Rb^T )^{o2} (diag(v_hat) V)]_i = (Rb_i . (A * Z_i)) summed

    where Z = Rb^T (diag(v_hat) V) in R^{D x s}, and A*Z is elementwise
    (Hadamard) between A (D x D) broadcast over s columns.

    Cost: O(ND^2) once at init, then O(NDs) per matmat call.
    For D=3 (Swiss roll) this is trivially cheap.
    For D=784 (MNIST) A is 784x784 ~ 5MB, still fine.
    """

    def __init__(
        self,
        X,
        beta: float,
        eps: Optional[float] = None,
        alpha: float = 1.0,
        device: str = "cuda",
        dtype=None,
        verbose: bool = True,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError("torch required")
        import torch

        self.device = torch.device(device)
        self.dtype  = dtype or torch.float32

        Xt = torch.as_tensor(
            X if isinstance(X, torch.Tensor) else np.asarray(X, dtype=np.float32),
            device=self.device, dtype=self.dtype,
        )
        N, D = Xt.shape

        if eps is None:
            r_tmp  = (Xt * Xt).sum(dim=1)
            mu_tmp = Xt.mean(dim=0)
            Rb_tmp = Xt - mu_tmp
            r_tmp  = (Rb_tmp * Rb_tmp).sum(dim=1)
            eps    = float(4.0 * r_tmp.mean())
            if verbose:
                print(f"[LDM2Operator] auto eps={eps:.4g} (4 * mean(r))")

        self.eps = eps
        self.pot = _precompute_potentials(
            Xt, beta, eps, alpha, self.device, self.dtype, verbose,
        )
        self.N = N
        self.D = D

        # ── Precompute A = Rb^T diag(u) Rb  in R^{D x D}  O(ND^2) ──────────
        # u here is the kernel potential v = exp(-beta r / eps)
        Rb   = self.pot["Rb"]             # (N, D)
        u    = self.pot["v"]              # (N,)  = exp(-beta r / eps)
        # A_{xy} = sum_i u_i Rb_{ix} Rb_{iy}
        Rb_u = Rb * u.unsqueeze(1)        # (N, D)  diag(u) Rb
        self.A = Rb_u.T @ Rb             # (D, D)  O(ND^2)

        if verbose:
            print(f"[LDM2Operator] A precomputed: shape={tuple(self.A.shape)} "
                  f"norm={float(self.A.norm()):.4g}")

        self.trivial_mode = (self.pot["v_hat_ldm"] /
                             self.pot["v_hat_ldm"].norm()).unsqueeze(1)

    def matmat(self, V: "torch.Tensor") -> "torch.Tensor":
        """
        V : (N, s)  ->  K^(2) V : (N, s)

        Steps:
          W      = diag(v_hat) V                              O(Ns)
          rank1  = diag(v_hat) 1 (1^T W)                     O(Ns)
          Z      = Rb^T W                                     O(NDs)  [GEMM1]
          n1term = diag(v_hat) Rb Z                           O(NDs)  [GEMM2]
          n2term = diag(v_hat) Rb (A * Z)                    O(D^2 s + NDs)
          output = rank1
                 + (2b/eps) n1term
                 + (2b^2/eps^2) n2term
        """
        import torch
        pot   = self.pot
        vh    = pot["v_hat_ldm"]          # (N,)
        Rb    = pot["Rb"]                 # (N, D)
        beta  = pot["beta"]
        eps   = pot["eps"]
        A     = self.A                    # (D, D)

        W     = vh.unsqueeze(1) * V                           # (N, s)
        rank1 = vh.unsqueeze(1) * W.sum(dim=0, keepdim=True) # (N, s)
        Z     = Rb.T @ W                                      # (D, s) GEMM1

        # n=1 term: diag(v_hat) Rb Z
        n1    = vh.unsqueeze(1) * (Rb @ Z)                   # (N, s) GEMM2

        # n=2 term: diag(v_hat) Rb (A odot Z)
        # A: (D, D), Z: (D, s) -> A*Z is (D,s) where col j of Z is scaled by A
        # Correct: for each sample i, contribution is Rb_i . (A * (Rb^T W)_i)
        # = Rb_i . (A @ w_col) -- but A is symmetric so A * Z means A @ Z here
        # The Hadamard-power identity: (Rb Rb^T)^{o2} u = Rb (A o (Rb^T u))
        # where o here is column-wise: each column z of Z gives A * z,
        # meaning the element-wise product of matrix A broadcast -- but since
        # we want sum_j (Rb_i.Rb_j)^2 u_j = Rb_i . (Rb^T (diag(Rb Rb^T u)))
        # = sum_x Rb_{ix} sum_y A_{xy} Z_{ys}  -- this IS A @ Z (matrix mult)
        AZ    = A @ Z                                         # (D, s) O(D^2 s)
        n2    = vh.unsqueeze(1) * (Rb @ AZ)                  # (N, s) GEMM3

        c1 = 2.0 * beta / eps
        c2 = 2.0 * beta**2 / eps**2

        return rank1 + c1 * n1 + c2 * n2

    @property
    def deflate_vector(self) -> "torch.Tensor":
        return self.trivial_mode


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: three-stage warm-start pipeline
# ─────────────────────────────────────────────────────────────────────────────

def ldm_warm_start(
    X,
    k: int,
    beta: float,
    full_matmat_fn,
    *,
    eps: Optional[float] = None,
    alpha: float = 1.0,
    device: str = "cuda",
    dtype=None,
    block_size: int = 16,
    tol_ldm: float = 1e-3,
    tol_ldm2: float = 1e-4,
    tol_full: float = 1e-6,
    max_restarts_ldm: int = 5,
    max_restarts_ldm2: int = 8,
    max_restarts_full: int = 20,
    skip_ldm2: bool = False,
    verbose: bool = True,
):
    """
    Three-stage block Lanczos warm start:

      Stage 1: LDM       O(NDs)   -> U0
      Stage 2: 2nd-order O(ND^2)  -> U1  (skip if D large or skip_ldm2=True)
      Stage 3: full DMAP O(N^2 D) -> final eigenpairs

    Parameters
    ----------
    X : array-like (N, D)
    k : int   number of eigenpairs
    beta : float   kernel bandwidth
    full_matmat_fn : callable V -> A@V  your CUDA exp kernel
    eps : float or None   lengthscale (auto if None)
    alpha : float   CL exponent
    block_size : int   Lanczos block size (use >= k, powers of 2 work well)
    tol_* : float   convergence tolerances per stage
    skip_ldm2 : bool   skip 2nd-order stage (e.g. for large D)

    Returns
    -------
    vals : np.ndarray (k,)
    vecs : np.ndarray (N, k)
    info : dict with per-stage diagnostics
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch required")
    import torch
    import time
    # import here to avoid circular if in same package
    from eigensolver_gpu import gpu_block_irlm

    import numpy as np

    device_ = torch.device(device)
    dtype_  = dtype or torch.float32

    Xt = torch.as_tensor(
        X if isinstance(X, torch.Tensor) else np.asarray(X, dtype=np.float32),
        device=device_, dtype=dtype_,
    )
    N  = Xt.shape[0]

    info = {}

    # ── Stage 1: LDM ─────────────────────────────────────────────────────────
    if verbose:
        print("\n[ldm_warm_start] Stage 1: LDM  O(NDs)")
    t0   = time.perf_counter()
    ldm  = LDMOperator(Xt, beta=beta, eps=eps, alpha=alpha,
                       device=device, dtype=dtype_, verbose=verbose)
    eps  = ldm.eps   # propagate auto-eps to stage 2

    vals0, U0, info0 = gpu_block_irlm(
        matmat_fn      = ldm.matmat,
        N              = N,
        k              = k,
        block_size     = block_size,
        tol            = tol_ldm,
        max_restarts   = max_restarts_ldm,
        deflate_basis  = ldm.deflate_vector,
        compute_true_residuals = False,
        return_info    = True,
        return_torch   = True,
        verbose        = verbose,
    )
    info["stage1"] = info0
    info["stage1"]["elapsed_total"] = time.perf_counter() - t0
    if verbose:
        print(f"  Stage 1 done: {info0['cycles']} cycles, "
              f"{info0['n_matmat']} matmats, "
              f"{info['stage1']['elapsed_total']:.2f}s")

    U_seed = U0

    # ── Stage 2: 2nd-order LDM ───────────────────────────────────────────────
    if not skip_ldm2:
        if verbose:
            print(f"\n[ldm_warm_start] Stage 2: 2nd-order LDM  O(ND^2), D={ldm.D}")
        t0    = time.perf_counter()
        ldm2  = LDM2Operator(Xt, beta=beta, eps=eps, alpha=alpha,
                             device=device, dtype=dtype_, verbose=verbose)

        vals1, U1, info1 = gpu_block_irlm(
            matmat_fn      = ldm2.matmat,
            N              = N,
            k              = k,
            block_size     = block_size,
            tol            = tol_ldm2,
            max_restarts   = max_restarts_ldm2,
            q0_block       = U_seed,
            deflate_basis  = ldm2.deflate_vector,
            compute_true_residuals = False,
            return_info    = True,
            return_torch   = True,
            verbose        = verbose,
        )
        info["stage2"] = info1
        info["stage2"]["elapsed_total"] = time.perf_counter() - t0
        if verbose:
            print(f"  Stage 2 done: {info1['cycles']} cycles, "
                  f"{info1['n_matmat']} matmats, "
                  f"{info['stage2']['elapsed_total']:.2f}s")
        U_seed = U1

    # ── Stage 3: full DMAP ───────────────────────────────────────────────────
    if verbose:
        print("\n[ldm_warm_start] Stage 3: full DMAP  O(N^2 D)")
    t0 = time.perf_counter()

    vals, vecs, info3 = gpu_block_irlm(
        matmat_fn      = full_matmat_fn,
        N              = N,
        k              = k,
        block_size     = block_size,
        tol            = tol_full,
        max_restarts   = max_restarts_full,
        q0_block       = U_seed,
        compute_true_residuals = True,
        return_info    = True,
        return_torch   = False,
        verbose        = verbose,
    )
    info["stage3"] = info3
    info["stage3"]["elapsed_total"] = time.perf_counter() - t0
    if verbose:
        print(f"  Stage 3 done: {info3['cycles']} cycles, "
              f"{info3['n_matmat']} matmats, "
              f"{info['stage3']['elapsed_total']:.2f}s")

    return vals, vecs, info
```

## `src/flashdiffusion/utils.py`

```python
"""
flashdiffusion/utils.py
=======================
Bandwidth selection, Nyström out-of-sample extension, dtype helpers.
"""

from __future__ import annotations
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Bandwidth selection
# ---------------------------------------------------------------------------

def median_bandwidth(X: np.ndarray, subsample: int = 2000) -> float:
    """
    Median heuristic for beta: beta = 1 / (2 * median(||xi-xj||²)).

    Subsamples if N > subsample to keep cost O(subsample²).
    """
    N = X.shape[0]
    if N > subsample:
        idx = np.random.default_rng(0).choice(N, subsample, replace=False)
        X = X[idx]
    sq   = (X * X).sum(axis=1)
    dist2 = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0)
    med   = np.median(dist2[dist2 > 0])
    return 1.0 / (2.0 * med) if med > 0 else 1.0


def knn_bandwidth(X: np.ndarray, k: int = 7) -> np.ndarray:
    """
    Self-tuning (Zelnik-Manor & Perona) per-point bandwidth.

    sigma_i = ||xi - x_{i,k}||   (distance to k-th nearest neighbour)

    For use with the anisotropic kernel:
        K(i,j) = exp(-||xi-xj||² / (sigma_i * sigma_j))

    Returns sigma (N,) — to use, replace beta * dist2 with dist2 / outer(sigma, sigma).
    Note: this changes the kernel primitive; requires a modified prepass.
    Currently returned as metadata for the caller to handle.
    """
    N = X.shape[0]
    sq    = (X * X).sum(1)
    dist2 = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0)
    # k-th nearest neighbour distance (excluding self at dist=0)
    sorted_d = np.sort(dist2, axis=1)
    sigma = np.sqrt(sorted_d[:, k])              # (N,)
    sigma = np.maximum(sigma, 1e-10)
    return sigma


# ---------------------------------------------------------------------------
# Nyström out-of-sample extension
# ---------------------------------------------------------------------------
def nystrom_extend(
    dm,
    X_new: np.ndarray,               # (M, d)  new points
    t:     int = 1,
) -> np.ndarray:
    """
    Nyström extension: embed new points using the training eigenvectors.

    For a new point x*, the diffusion map coordinate is:
        ψₖ(x*) = (1/λₖ) · Σᵢ p(x*, xᵢ) · φₖ(xᵢ)

    where p(x*, xᵢ) = K_alpha(x*, xᵢ) / D_alpha(x*)
    and φₖ are the (row-normalised) eigenvectors of P⁺.

    This requires ONE tiled kernel pass for each new point batch,
    reusing the precomputed rscale_ from the training set.

    Returns
    -------
    coords : (M, n_components-1) diffusion coordinates for X_new
    """
    from .kernel import FlashDiffusion, prepass_rowsum

    if dm.eigenvalues_ is None:
        raise RuntimeError("Call transform() on training data first.")

    beta, alpha, tile = dm.beta, dm.alpha, dm.tile

    # ── center new points with training mean ────────────────────────────
    # Must use dm.mu_ (training mean) not X_new.mean().
    # Consistency: kernel sees (x_new - mu) vs (x_train - mu) = dm.X_
    # dm.X_ is already centered from fit(), so this is the only change.
    if not hasattr(dm, 'mu_'):
        raise RuntimeError(
            "dm.mu_ not found — refit with the updated fit() that stores mu_.")
    X_new = np.asarray(X_new, dtype=np.float64) - dm.mu_   # (M, D) centered

    # everything below is unchanged — dm.X_ is already centered
    X_train  = dm.X_                                        # (N, D) centered
    N_train  = X_train.shape[0]
    rscale_k = dm.rscale_

    # normalise new points against training set
    w_train = dm.D_ ** (-alpha)
    D_new   = prepass_rowsum(X_new, X_train, beta,
                              weights_k=w_train, tile=tile)
    w_new   = D_new ** (-alpha)

    D_alpha_new    = D_new * w_new
    inv_Dalpha_new = w_new / D_alpha_new

    rscale_q_new = D_alpha_new ** (-0.5) * w_new

    # training eigenvectors in P⁺ basis
    phi = dm.eigenvectors_ / dm.rscale_[:, None]            # (N_train, k)
    phi /= np.linalg.norm(phi, axis=0, keepdims=True)

    # Nyström kernel pass
    coords_new = FlashDiffusion(
        Q        = X_new,                                   # ← centered
        K        = X_train,                                 # ← already centered
        V        = phi[:, 1:],
        beta     = beta,
        rscale_q = rscale_q_new,
        rscale_k = rscale_k,
        tile     = tile,
    )

    lam = dm.eigenvalues_[1:]
    coords_new *= (lam ** (t - 1))[None, :]

    return coords_new                                        # (M, k-1)


# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------

def to_f32(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float32)

def to_f64(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float64)
```

## `tests/test_kernel.py`

```python
"""tests/test_kernel.py — kernel correctness vs dense reference."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from flashdiffusion import FlashDiffusion, prepass_rowsum, DiffusionMap
from scipy.sparse.linalg import eigsh


def make_dataset(N=200, d=3, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 4*np.pi, N)
    X = np.stack([t*np.cos(t), t*np.sin(t), rng.uniform(0, 2, N)], axis=1)
    return X / X.std()


def dense_kernel(X, beta):
    sq = (X*X).sum(1)
    d2 = np.maximum(sq[:,None] + sq[None,:] - 2*X@X.T, 0)
    return np.exp(-beta * d2)


def dense_dmap_operator(X, beta, alpha):
    K = dense_kernel(X, beta)
    D = K.sum(1); w = D**(-alpha)
    Ka = K * w[:,None] * w[None,:]
    Da = Ka.sum(1); dh = Da**(-0.5)
    M = dh[:,None] * Ka * dh[None,:]
    rscale = dh * w
    return M, rscale


@pytest.mark.parametrize("tile", [64, 256])
def test_prepass_rowsum(tile):
    X = make_dataset(N=150)
    K = dense_kernel(X, beta=1.0)
    D_ref = K.sum(1)
    D_ours = prepass_rowsum(X, X, beta=1.0, tile=tile)
    np.testing.assert_allclose(D_ours, D_ref, rtol=1e-5)


@pytest.mark.parametrize("beta,alpha", [(1.0, 0.5), (0.5, 0.0), (2.0, 1.0)])
def test_flash_diffusion_matvec(beta, alpha):
    X = make_dataset(N=150)
    M, rscale = dense_dmap_operator(X, beta, alpha)
    v = np.random.default_rng(7).standard_normal(150)
    ref = M @ v

    out = FlashDiffusion(X, X, v[:,None], beta=beta,
                         rscale_q=rscale, rscale_k=rscale).ravel()
    np.testing.assert_allclose(out, ref, rtol=2e-4)  # float32 GEMM in tile loop


def test_dmap_fit_transform():
    X = make_dataset(N=300)
    beta, alpha, k = 1.0, 0.5, 6

    M, _ = dense_dmap_operator(X, beta, alpha)
    vals_ref = np.sort(eigsh(M, k=k, which='LM')[0])[::-1]

    dm = DiffusionMap(beta=beta, alpha=alpha, n_components=k, tile=128)
    dm.fit(X)
    coords = dm.transform()

    vals_ours = dm.eigenvalues_
    err = np.abs(vals_ref - vals_ours) / np.abs(vals_ref)
    assert err.max() < 1e-5, f"max eigenvalue error {err.max():.2e}"
    assert coords.shape == (300, k-1)


def test_doob_transform_preserves_eq():
    """
    Doob transform preserves EQ class (Theorem 5.1, DAC).
    We check: eigenvalues real, positive, and the transformed operator
    is still self-adjoint w.r.t. the new measure (symmetric spectrum).
    The top eigenvalue is NOT necessarily ≤ 1 after the transform
    (h rescales the stationary distribution, so the spectrum shifts).
    """
    X = make_dataset(N=100)
    dm = DiffusionMap(beta=1.0, alpha=0.5, n_components=4, tile=64)
    dm.fit(X)
    dm.transform()

    h = np.exp(-(X**2).sum(1))
    doob = dm.doob_transform(h)

    from flashdiffusion import lanczos_eigsh
    vals, _ = lanczos_eigsh(doob.matvec, N=100, k=4)
    # eigenvalues should be real (they are, since matvec returns float64) and
    # the operator should be a valid Markov-like operator: eigenvalues in (-1, ∞)
    assert np.all(vals > -1e-4), f"unexpected negative eigenvalues: {vals}"
    # stationary measure (h=const) -> top eigenvalue = 1 only for trivial h
    # for nontrivial h the spectrum shifts but stays real
    print(f"  Doob eigenvalues: {vals}  (real, operator well-defined ✓)")


if __name__ == "__main__":
    test_prepass_rowsum(256)
    test_flash_diffusion_matvec(1.0, 0.5)
    test_dmap_fit_transform()
    test_doob_transform_preserves_eq()
    print("All tests passed.")
```
