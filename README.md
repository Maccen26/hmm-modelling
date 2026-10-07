# HMM

This project is a special course at the Technical University of Denmark (DTU) exploring how to model $\text{CO}_2$ data with Hidden Markov Models. It was conducted in the Spring Semester 2026 (5 ECTS) and leads up to a Bachelor Project. The focus is on designing flexible HMM software, exploring and interpreting HMM states, and testing different HMM variants (ordinary, autoregressive, second-order, covariate-driven, and continuous-time) on $\text{CO}_2$ time series data.

Package manager: `uv`. Python 3.11. Models are JAX-based and fitted by gradient/L-BFGS optimisation of the negative log-likelihood (not classic Baum-Welch EM).

## Repository layout

- **`src/`** — The HMM library. Abstract base classes in `src/base/` (emission, transition, HMM, solver, inference) and the concrete JAX implementations in `src/api/v4/`. Also holds shared config (`src/config/`) and data loading (`src/data.py`).
- **`tests/`** — Unit and integration tests for the library. Only `tests/v4` is current.
- **`drivers/`** — Runnable entry points that use `src/api/v4` to fit models and produce the result tables/figures consumed by the report. See the [Drivers](#drivers) section below.
- **`report/`** — LaTeX source for the written report (`main.tex`, `sections/`, `preamble/`, `references.bib`) along with build artefacts and the compiled `main.pdf`.
- **`week_*.ipynb`**, **`data_exploration.ipynb`**, **`model_predictions.ipynb`** — weekly notebooks for exploration, model fitting, and prediction.

## The library (`src/api/v4`)

An HMM is composed of a **transition** and an **emission**, held together as a JAX (Equinox) pytree so gradients flow through all trainable parameters. The high-level `HMM` class wraps these and provides `fit`, `log_likelihood`, `pseudo_residuals`, and `predict_emission`.

- **Emissions** — `GaussEmission`, `AutoregressiveGaussEmission`.
- **Transitions** — `StaticTransition`, `StaticTransitionHigherOrder` (second-order), `DynamicTransition` (covariate-driven), and continuous-time `ContinuousStaticTransition` / `ContinuousDynamicTransition` (matrix built as `expm(Q·gap)`).
- **Inference** — `ForwardAlgorithm` (forward pass via `jax.lax.scan`).
- **Solvers** — `LBFGSSolver` (default) and `GradientSolver`.

Minimal usage:

```python
import jax
jax.config.update("jax_enable_x64", True)  # required: fitting is numerically sensitive

from src.api.v4 import HMM, StaticTransition, GaussEmission

transition = StaticTransition.from_params(transition_matrix)
emission = GaussEmission.from_params(mu=mu, sigma=sigma)
model = HMM(transition=transition, emission=emission)
model.fit(ys=ys, xs=Xs)
```

## Data

Data lives outside the repo (the `data/` directory is gitignored). The `DATA_PATH` environment variable in `.env` points to the data root, and datasets are laid out as `{DATA_PATH}/{data_name}/{tag}/{y,X}_{train,test}.csv`, where `data_name` is a dataset (e.g. `b1`, `dtu`) and `tag` is a split (e.g. `jans-split`).

## Drivers

The `drivers/` package holds the runnable entry points that fit each HMM variant, save the fitted model, and produce the diagnostic plots, statistics, and LaTeX tables for the report. Fitted models are pickled to `results/models/{data_name}/{tag}/{model}.pkl`, and all other outputs are written under `results/`.

- **`drivers/fit_model_b1.py`** — the main fitting driver. `init_<model>()` functions build each variant (several seed from a previously fitted model), `fit_model()` fits and pickles, and the `INIT_MODELS` registry maps model names → initialisers. Set `MODEL_NAME = "all"` to fit every model in dependency order. Available models:
  - `ordinary_hmm` — ordinary 4-state Gaussian HMM.
  - `ar_hmm` — AR(1) Gaussian emission (seeds from `ordinary_hmm`).
  - `ar_2_hmm` — AR(2) Gaussian emission (seeds from `ar_hmm`).
  - `second_order_hmm` — second-order transition, AR(1) emission.
  - `ar_2_second_order_hmm` — second-order transition, AR(2) emission.
  - `covariate_hmm` — dynamic (covariate-driven) transition (seeds from `ordinary_hmm`).
  - `ar_1_covariate_hmm`, `ar_2_covariate_hmm` — covariate transition with AR(1)/AR(2) emission.
- **`drivers/utils.py`** — shared helpers: `load_train_data` / `load_test_data`, `save_model` / `load_model`, `format_transition_matrix`, `load_time_covariates` (cyclic time-of-day covariates), `plot_hmm_diagnostics`, `write_latex_table`.
- **`drivers/main_plots.py`** — writes a diagnostics PNG (log-likelihood trace, Q–Q of pseudo-residuals, ACF) per model to `results/plots/{data_name}/{tag}/`.
- **`drivers/test_statistics.py`** — per-model stats (#params, log-likelihood, AIC, BIC) and likelihood-ratio tests of nested model pairs; writes `model_stats.csv` and `lrt_comparison.csv` to `results/test_statistics/{data_name}/{tag}/`.
- **`drivers/emission_params.py`**, **`transition_params.py`**, **`beta_params.py`** — extract fitted emission parameters, transition matrices, and covariate (beta) coefficients to CSVs under `results/`.
- **`drivers/latex.py`** — renders the report LaTeX tables from the CSVs above into `report/model_results/`.
- **`drivers/legacy/`** — an earlier iteration of the pipeline, kept for reference only; do not use or extend.

### Running

Run from the **repository root** using module syntax (`-m`) so the `drivers` and `src` packages resolve. Each driver reads `DATA_NAME` / `TAG` from its `__main__` block — edit these to target a different dataset/split.

```bash
# Fit and save all models (edit DATA_NAME / TAG / MODEL_NAME in __main__)
uv run python -m drivers.fit_model_b1

# Regenerate plots / statistics / params from already-fitted models
uv run python -m drivers.main_plots
uv run python -m drivers.test_statistics
uv run python -m drivers.emission_params
uv run python -m drivers.transition_params
uv run python -m drivers.beta_params

# Render the report tables from the computed CSVs
uv run python -m drivers.latex
```

Because several models seed from a previously fitted one, fit with `MODEL_NAME = "all"` (which fits in the correct dependency order) before running the individual plot, statistics, or parameter drivers.

## Tests

`pytest` is not installed; run the suite with `unittest`:

```bash
uv run python -m unittest discover -s tests -p "test_*.py"
```

Only `tests/v4` is current.

## Report

LaTeX source in `report/` builds with `latexmk -xelatex` (VS Code LaTeX Workshop is configured for XeLaTeX on save). `report/main.pdf` is the compiled output.

