# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Implementation of "Joker: Joint Optimization Framework for Lightweight Kernel Machines" — a block coordinate descent (BCD) framework for training large-scale kernel machines efficiently. Uses random Fourier features (RFF) and Fastfood approximations to scale kernel methods to millions of samples.

Core optimization problem:
```
min_θ  ½⟨θ,θ⟩ + c Σᵢ ℓ(yᵢ, ⟨θ, φ(xᵢ)⟩)
```
where `c` is the penalty parameter (equivalent to `1/λ` in the paper).

## Running experiments

No build system or package manager. Run directly from `src/`:

```bash
cd src
python experiments.py --dataset <name> --criterion <loss> [options]
```

Datasets: `susy`, `msd`, `higgs`, `houseelec`, `cifar5m`
Criteria: `mse`, `huber`, `svm`, `log`, `svr`
Kernels: `rbf`, `student`, `lap`

See README.md for dataset-specific example commands.

## Configuration

Two-tier config:
- `src/static_config.py` — local overrides (`DATA_DIR`, `MACHINE_NAME`). Edit this to point to your dataset directory.
- `src/run_config.py` / `RuntimeConfig` — CLI argument parsing, factory methods for task/criterion/kernel, and CSV result logging.

Datasets must be downloaded separately as `.npz` files (see `data/<name>/README.md` for download instructions) and placed under `DATA_DIR`.

## Architecture

```
src/
  experiments.py       Entry point — creates model, runs fit/predict/evaluate
  joker.py             Joker (exact kernel) and InexactJoker (RFF/Fastfood approximation)
  run_config.py        CLI args, result recording to CSV
  static_config.py     Local paths
  kernels/
    kernel.py          Kernel functions: KeRBF, KeLaplace, KeStudenT (+KeLinear, KePoly, KeSigmoid unused)
    fastfood.py        Fastfood kernel approximation via fast Hadamard transform
  optim/
    blk.py             Block structure for data iteration (NormalBlocks, OptimizationBlocks)
    criterion.py       Loss functions: MSE, Huber, SquaredHinge, LogLoss, eps-insensitive (SVR)
    trust_region.py    Trust-region optimizer with CG-Steihaug method and box constraints
  task/
    task.py            Dataset loading from .npz files and evaluation metrics
```

**Algorithm flow:**
1. Load dataset → compute median-heuristic kernel bandwidth (`sig2` from 10K pairwise distances)
2. Create `Joker` (exact, n_rff=-1) or `InexactJoker` (RFF or Fastfood) model
3. BCD outer loop: pick a block of training samples, solve trust-region subproblem via CG-Steihaug, update `alpha`/`beta` dual variables
4. Predict on test set, evaluate with task-specific metric, log results

**Key design decisions:**
- `Joker` maintains `alpha` (N × n_target) and `beta = K @ alpha` (N × n_target) as dual variables. Blocks index into these without data copies.
- `InexactJoker` replaces the N×N kernel matrix with explicit feature vectors (n_features × d), so `beta` becomes n_features × n_target — much smaller.
- RFF is prioritized over Fastfood when both are specified (see `experiments.py:26`).
- `incore=True` (default) keeps all training data in GPU memory. Set `--no-incore` for datasets that don't fit.
