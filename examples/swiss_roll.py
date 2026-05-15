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
