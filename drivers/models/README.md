# Model drivers

Each file fits one model on one room, saves it, and evaluates it on the validation split.
Run from the repo root with module syntax:

```bash
uv run python -m drivers.models.<driver> --room 9 [--states K]   # one model on room_009
uv run python -m drivers.models.run --room 9 --states K          # every model, in fit order
uv run python -m drivers.models.run --room 9 --states K --sigma-lam 1e-4   # same, plus a ridge on σ²
uv run python -m drivers.models.run --room 9 --states K --models hmm_1 ar1_hmm_1
```

`--room` / `-r` is the room number (`9` → `room_009`). `--states` / `-K` sets the number of hidden states; without it a driver uses its own default.
Observation columns and covariates are shared defaults in `utils.py` (`OBS_COLS`, `COVARIATE_COLS`). All room data is 30-min binned, so there is no tag on the command line.

## Losses

The objective is the NLL plus two optional ridge penalties, `Loss(beta_lam, sigma_lam)`:
`beta_lam · Σβ²` on the transition covariate effects and `sigma_lam · Σσ²` on the emission variances.

- Each driver's `LOSS` fixes its β-ridge: `hmm_beta_1`, `ar1_hmm_beta_1` and `ar1_hmm_beta_1_shared` use λ_β = 1e-2 (their betas diverge under plain NLL), and every other driver uses plain NLL.
- `--sigma-lam` adds the σ-ridge on top. The suggested value is `SIGMA_LAM = 1e-4` in `utils.py`: about 20 nats on a σ≈450 ppm state, and negligible on the σ≈20–30 ppm states.

The loss slug names the results/plots folder (`nll`, `ridge_sigma_lam0.0001`, `ridge_beta_lam0.01`, `ridge_beta_lam0.01_ridge_sigma_lam0.0001`, …), so changing a λ never overwrites earlier fits.

## Fit order

Drivers don't refit their seed models; they load them from `results/` for the same room/states, using **the seed's own β-ridge and this run's σ-ridge**. Each `--sigma-lam` value is therefore its own chain.
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

- `results/{room}/{loss}/states_{K}/{model}.pkl`: the fitted model, its config, and the training data
- `results/{room}/{loss}/states_{K}/predictive_metrics.csv`: validation R², RMSE, and MAPE per forecast horizon, shared by the models fitted with that loss
- `plots/{room}/{loss}/states_{K}/{model}.png` and `{model}_first_week.png`: residual diagnostics and forecast plots; titles show room, K and loss

## Tools for saved fits

```bash
uv run python -m drivers.models.replot --room 9 [--loss nll] [--states 6] [--models hmm_1]
uv run python -m drivers.models.decode_states ar1_hmm_beta_1 --room 9 -K 6
uv run python -m drivers.models.show_params ar1_hmm_beta_1 --room 9 -K 6
```

`decode_states` and `show_params` look in the driver's own loss folder unless `--loss` is given.
