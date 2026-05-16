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

## Citation

```bibtex
@software{flashdiffusion2025,
  title  = {FlashDiffusion},
  year   = {2025},
  note   = {Tiled diffusion maps eigensolver}
}
```
