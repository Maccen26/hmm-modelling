# HMM 

This project is a special course at the Technical University of Denmark (DTU) exploring how to model $\text{CO}_2$ data with Hidden Markov Models. It was conducted in the Spring Semester 2026 (5 ECTS) and leads up to a Bachelor Project. The focus is on designing flexible HMM software, exploring and interpreting HMM states, and testing different HMM variants (ordinary, AR, second-order) on $\text{CO}_2$ time series data.

Package manager: `uv`.

## Repository layout

- **`src/`** — The HMM library. Contains the base classes (emission, transition, HMM), optimisation utilities, data loading, and the versioned APIs under `src/api/v1`–`v4` that compose emissions and transitions into concrete models.
- **`tests/`** — Unit and integration tests for the library. Only `tests/v4` is current.
- **`drivers/`** — Runnable entry points that use `src` to fit models, generate plots, and produce the result tables/figures consumed by the report. See the [Drivers](#drivers) section below.
- **`report/`** — LaTeX source for the written report (`main.tex`, `sections/`, `preamble/`, `references.bib`) along with build artefacts and the compiled `main.pdf`.

## Drivers

The `drivers/` package holds the runnable entry points that fit each HMM variant, save the
fitted model, and produce the diagnostic plots and comparison tables. Every fitted `HMM`
object is pickled into the root `results/` directory (`results/models/<name>.pkl`), and all
outputs are written under `results/`.

### Layout

- **`drivers/main.py`** — runs the full pipeline: fit all models → diagnostic plots → test statistics.
- **`drivers/utils.py`** — shared helpers: `save_model`, `load_model`, `format_transition_matrix`, `load_time_covariates` (cyclic time-of-day covariates), `plot_hmm_diagnostics`, `write_latex_table`.
- **`drivers/models/`** — one driver per model variant, each with a `run_<model>()` function:
  - `ordinary_hmm.py` — ordinary 4-state Gaussian HMM.
  - `ar_hmm.py` — AR(1) Gaussian emission (seeds from `ordinary_hmm`).
  - `ar_2_hmm.py` — AR(2) Gaussian emission (seeds from `ar_hmm`).
  - `second_order_hmm.py` — second-order (HMM(2)) transition, AR(1) emission (seeds from `ar_hmm`).
  - `ar_2_second_order_hmm.py` — second-order transition, AR(2) emission (seeds from `second_order_hmm` / `ar_2_hmm`).
  - `covariate_hmm.py` — dynamic (covariate-driven) transition with time-of-day covariates; an extension of the ordinary HMM (seeds from `ordinary_hmm`).
  - `main_models.py` — fits every model in dependency order and saves each to `results/models/`.
- **`drivers/plots/`** — diagnostic plot drivers (log-likelihood trace, Q–Q of pseudo-residuals, ACF):
  - `plot_base_co2.py` — plot of the raw CO₂ observations.
  - `plot_<model>_diag.py` — one per model (`ordinary`, `ar`, `ar_2`, `second_order`, `ar_2_second_order`, `covariate`).
  - `main_plots.py` — writes a diagnostics PNG per model to `results/plots/`.
- **`drivers/test_statistics.py`** — builds the per-model stats table (log-likelihood, AIC, BIC) and the likelihood-ratio-test comparison of nested models (including `ordinary_hmm` → `covariate_hmm`); writes CSVs to `results/test_statistics/` and LaTeX tables to `report/model_results/comparison/`.

### Running

Run from the **repository root** using module syntax (`-m`) so the `drivers` and `src`
packages resolve:

```bash
# Full pipeline: fit all models, generate plots, compute test statistics
python -m drivers.main

# Only fit and save all models
python -m drivers.models.main_models

# A single model (dependencies must already exist in results/models/)
python -m drivers.models.ordinary_hmm
python -m drivers.models.covariate_hmm   # requires results/models/ordinary_hmm.pkl

# Only regenerate plots / statistics from already-fitted models
python -m drivers.plots.main_plots
python -m drivers.test_statistics
```

Because several models seed from a previously fitted one, run `drivers.main` or
`drivers.models.main_models` first (they fit in the correct dependency order) before running
individual plot or statistics drivers.

## API versioning — only v4 is active

The `src/api/` directory contains four iterations of the modelling API (`v1`, `v2`, `v3`, `v4`). **Only `v4` should be treated as the working, supported API.** All earlier versions (`v1`, `v2`, `v3`) and anything under `src/deprecated/` are kept for historical reference only and should not be used or extended. New drivers, tests, and report results target `src/api/v4` exclusively.

## AI disclosure

1. Copilot chat completion has been used.
2. No agents have written source code inside the directory /src/.
3. Agents have been used to debug JAX modules (sometimes).
4. Claude has been used to find sources and explain concepts.
5. Claude Code has been used to generate documentation about the code and code for plotting. 
