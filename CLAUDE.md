# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A DTU special course exploring how to model CO₂ time-series data with Hidden Markov Models (see `README.md`). The deliverables are a flexible HMM library (`src/`), runnable experiments (`drivers/`), Jupyter notebooks per week (`week_*.ipynb`), and a LaTeX report (`report/`). The math is JAX-based and gradient-optimised, not the classic Baum-Welch EM.

## Commands

Package manager is `uv`; Python is pinned to 3.11.

```bash
# Run the test suite (pytest is NOT installed — use unittest)
uv run python -m unittest discover -s tests -p "test_*.py"

# Run a single test module / case
uv run python -m unittest tests.v4.test_modules.test_hmm
uv run python -m unittest tests.v4.test_modules.test_hmm.TestHMM.<method>

# Fit every model on one room, in seed order (see drivers/models/README.md)
uv run python -m drivers.models.run --room 9 --states K

# Add a dependency
uv add <package>
```

Always run drivers with module syntax (`-m`) from the repo root so `src` and `drivers` resolve.

The full run takes minutes (JAX compile + 121 tests ≈ 2.5 min). Run tests in the background rather than blocking.

## Critical conventions

- **Only `src/api/v4` is live.** There is no `src/api/v1`–`v3` anymore; everything under `drivers/legacy/` is historical and must not be extended. New drivers, tests, and report results target `src/api/v4` exclusively. (The `README.md` "Drivers"/"API versioning" sections describe the old `drivers/main.py` + `drivers/models/` layout and are stale — trust `drivers/models/README.md` instead.)
- **Enable float64.** Fitting is numerically sensitive; drivers set `jax.config.update("jax_enable_x64", True)` at the top of the file before importing anything. Do the same in any new fitting entry point or notebook.
- **`DATA_PATH` env var (`.env`) points to the data root**, which is gitignored. `drivers.utils.load_train_data(data_name, tag)` reads `{DATA_PATH}/{data_name}/{tag}/{y,X}_{train,test}.csv`. `data_name` is a dataset (`b1`, `dtu`); `tag` is a split (e.g. `jans-split`).
- **Fitted models are pickled** to `results/models/{data_name}/{tag}/{model}.pkl`; other outputs go under `results/`. Models seed from previously-fitted ones (e.g. `ar_hmm` loads `ordinary_hmm`), so fit in dependency order — the `INIT_MODELS` registry in `drivers/legacy/fit_model_b1.py` (run with `MODEL_NAME="all"`) enforces this.

## Architecture

The library is layered: abstract bases in `src/base/`, concrete JAX implementations in `src/api/v4/`, orchestration in the `HMM` class.

**An HMM = a transition + an emission, composed as a JAX pytree.** `HMMParams` (`src/api/v4/hmm_models/hmm_params.py`) subclasses `BaseHMM`, which is an `equinox.Module` holding two fields — `transition: BaseTransition` and `emission: BaseEmission`. Because it's an eqx pytree, gradients flow through all trainable parameters automatically.

- **Emissions** (`src/api/v4/emissions/`) — `GaussEmission`, `AutoregressiveGaussEmission`. Each exposes `density`, `cdf`, `mu`, `step`. Construct via `.from_params(mu=, sigma=, phi=)`.
- **Transitions** (`src/api/v4/transitions/`) — `StaticTransition`, `StaticTransitionHigherOrder` (second-order), `DynamicTransition` (covariate-driven), and continuous-time `ContinuousStaticTransition` / `ContinuousDynamicTransition` (build the matrix as `expm(Q·gap)`). Parameterised by `transition_logits`; build via `.from_params(transition_matrix)`.
- **Inference** (`src/api/v4/algorithms/`) — `ForwardAlgorithm` (`BaseInference` subclass) runs the forward pass with `jax.lax.scan`, returning `ForwardOutput` (filtered `utt`, one-step-ahead predictive `ut`, likelihood factors `ft`).
- **Solvers** (`src/api/v4/solvers/`) — `LBFGSSolver` (default) and `GradientSolver` (`BaseSolver` subclasses). They minimise the negative log-likelihood over the eqx pytree.

**The `HMM` class (`src/api/v4/hmm_models/hmm.py`) is the high-level entry point** most code touches. It wraps `HMMParams`, computes the initial state distribution (stationary by default), and provides `fit`, `log_likelihood`, `pseudo_residuals`, `predict_emission`, and `update_param`. `fit` runs the solver in a loop until the LL change falls below `tol`, storing per-iteration LLs in `ll_fits` and populating `hmm_results` / `state_results`.

**Time is a first argument everywhere.** `ts` is an optional per-observation waiting-time array threaded through emissions, transitions, and inference. Discrete models default to unit steps (`jnp.arange(len(ys))`); continuous models require real `ts` gaps. `t_pred` in `predict_emission` is absolute distances from the last observation (positive, strictly increasing).

**Freezing parameters.** `fit(..., frozen=...)` takes a dict: `{"mu0": False}` freezes a whole leaf; a tuple value (e.g. `{"phi_tilde": (0, 3)}`) freezes specific elements. See `BaseSolver._parse_frozen` / `_build_filter_spec`.

## Drivers

`drivers/models/` holds the current experiment entry points: one driver per model variant (`hmm_1`, `hmm_beta_1`, `ar1_hmm_1`, …), `run.py` to fit them all in seed order, and tools for saved fits (`replot`, `decode_states`, `show_params`, `state_confusion`, `feedback_states`, `transition_plots`). Its `README.md` documents the CLI, losses, fit order and output layout. The top-level `drivers/` keeps only what these depend on: `utils.py` (shared IO/covariate helpers) and the data-cleaning scripts that write `{DATA_PATH}/room_XXX/30min/` (`clean_303b_data.py`, `clean_dtu_data.py`, `fetch_humidity.py`). The old b1 workflow (`fit_model_b1.py`, `{transition,emission,beta}_params.py`, `main_plots.py`, `test_statistics.py`, `latex.py`, …) lives in `drivers/legacy/`.

## Report

LaTeX source in `report/` builds with `latexmk -xelatex` (VS Code LaTeX Workshop is configured for XeLaTeX on save). `report/main.pdf` is the compiled output.
