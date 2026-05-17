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
