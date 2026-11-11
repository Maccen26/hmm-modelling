# Model drivers


Each file fits one model on one room, saves it, and evaluates it on the validation split.
Run from the repo root with module syntax:

```bash
uv run python -m drivers.models.<driver> --room 9 [--states K]   # one model on room_009
uv run python -m drivers.models.run --room 9 --states K          # every model, in fit order
uv run python -m drivers.models.run --room 9 --states K --tag 30min       # same, on another data tag
uv run python -m drivers.models.run --room 9 --states K --models hmm_1 ar1_hmm_1
```

`--room` / `-r` is the room number (`9` → `room_009`). `--states` / `-K` sets the number of hidden states; without it a driver uses its own default.
`--tag` is the data tag, i.e. the binning folder the room CSVs are read from (`{DATA_PATH}/{room}/{tag}/`); it defaults to `TAG = "30min"` in `utils.py`.
Observation columns and covariates are shared defaults in `utils.py` (`OBS_COLS`, `COVARIATE_COLS`).

## Losses

The objective is the NLL plus an optional ridge penalty `beta_lam · Σβ²` on the transition covariate effects, `Loss(beta_lam)`.
Each driver's `LOSS` fixes it: `hmm_beta_1`, `ar1_hmm_beta_1` and `ar1_hmm_beta_1_shared` use λ_β = 1e-2 (their betas diverge under plain NLL), and every other driver uses plain NLL.
Since the loss is fixed per driver, the model slug identifies it and the loss is not part of the output path; its slug (`nll`, `ridge_beta_lam0.01`) appears in the plot titles.

## Fit order

Drivers don't refit their seed models; they load them from `results/` for the same room/tag/states.
If a seed is missing, the driver raises an error naming the command to run. `run` fits in this order:

| Driver | Base loss | Seeded from |
|---|---|---|
| `hmm_1` | `nll` | quantiles of the data |
| `hmm_beta_1` | `ridge_beta_lam0.01` | `hmm_1` |
| `hmm_beta_1_shared` | `nll` | `hmm_1` |
| `ar1_hmm_1` | `nll` | `hmm_1` |
| `ar1_hmm_beta_1` | `ridge_beta_lam0.01` | `ar1_hmm_1` (emission) + `hmm_beta_1` (transition) |
| `ar1_hmm_beta_1_shared` | `ridge_beta_lam0.01` | `ar1_hmm_1` (emission) + `hmm_beta_1_shared` (transition) |

## Outputs

Both folders sit at the repo root, not inside `drivers/`.

- `results/{room}/{tag}/states_{K}/{model}.pkl`: the fitted model, its config, and the training data
- `results/{room}/{tag}/states_{K}/predictive_metrics.csv`: validation R², RMSE, and MAPE per forecast horizon, shared by all models of that room/tag/K
- `plots/{room}/{tag}/states_{K}/{model}.png` and `{model}_first_week.png`: residual diagnostics and forecast plots; titles show room, K and loss

## Tools for saved fits

The post-hoc analysis tools live in `drivers/analysis/` and read the fits above:

```bash
uv run python -m drivers.analysis.replot --room 9 [--tag 30min] [--states 6] [--models hmm_1]
uv run python -m drivers.analysis.decode_states ar1_hmm_beta_1 --room 9 -K 6
uv run python -m drivers.analysis.show_params ar1_hmm_beta_1 --room 9 -K 6
uv run python -m drivers.analysis.state_confusion ar1_hmm_beta_1 --room 9 -K 6 --horizon 4 [--split val|test]
uv run python -m drivers.analysis.feedback_states ar1_hmm_beta_1 --room 9 -K 6 [--split train|val|test] [--paths 10000]
uv run python -m drivers.analysis.transition_plots ar1_hmm_beta_1 --room 9 -K 4 --state 2 [--week 2023-11-27]
uv run python -m drivers.analysis.sample_size --room 9 -K 5 [--p 0.05] [--R 100] [--n-max 150] [--n-step 5] [--tol 1e-3] [--init-noise 0.1]
```

`state_confusion` plots the confusion matrix of the H-step predicted state (argmax of the rolling forecast's state distribution) against the smoothed state (argmax of `ForwardBackwardAlgorithm`).
`feedback_states` samples state paths with `BackwardSampler` and plots which states the feedback events fall in, with the exact forward-backward share as a check.
`transition_plots` plots the covariate-driven transition probabilities out of one state.
`sample_size` simulates R datasets from the fit, refits each on its data up to the N-th complaint on the grid N = n_step, 2·n_step, …, N_MAX (warm-started from the previous grid point), and plots the 95 % CI and prediction interval of p̂ = N / Σ P(occupied | y₁:ₜ) against N; datasets, refitted models, `p_hat.csv` and `summary.csv` go to `results/.../sample_size/`, saved after every N.

`decode_states`, `show_params`, `state_confusion`, `feedback_states` and `transition_plots` read from the `30min` tag unless `--tag` is given.
