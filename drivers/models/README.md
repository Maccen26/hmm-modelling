# Model drivers

Each file fits one model on one room, saves it, and evaluates it on the validation split.
Run from the repo root with module syntax:

```bash
uv run python -m drivers.models.<driver> --room 9 [--states K]   # one model on room_009
uv run python -m drivers.models.run --room 9 --states K          # every model, in fit order
uv run python -m drivers.models.run --room 9 --states K --models hmm_1 ar1_hmm_1
```

`--room` / `-r` is the room number (`9` → `room_009`). `--states` / `-K` sets the number of hidden states; without it a driver uses its own default.
Observation columns and covariates are shared defaults in `utils.py` (`OBS_COLS`, `COVARIATE_COLS`). All room data is 30-min binned, so there is no tag on the command line.

## Losses

Each driver declares the objective it is fitted with as `LOSS = Loss(name, lam)` (`nll`, `ridge_beta`, `ridge_beta_sigma`).
The loss slug (`nll`, `ridge_beta_lam1`, `ridge_beta_sigma_lam0.01`) names the results/plots folder, so changing λ never overwrites earlier fits.

## Fit order

Drivers don't refit their seed models; they load them from `results/` for the same room/states, from the folder of **the seed's own loss**.
If a seed is missing, the driver raises an error naming the command to run. `run` fits in this order:

| Driver | Loss | Seeded from |
|---|---|---|
| `hmm_1` | `nll` | quantiles of the data |
| `hmm_beta_1` | `ridge_beta_lam1` | `hmm_1` |
| `hmm_beta_1_shared` | `nll` | `hmm_1` |
| `ar1_hmm_1` | `nll` | `hmm_1` |
| `ar1_hmm_beta_1` | `ridge_beta_sigma_lam0.01` | `ar1_hmm_1` (emission) + `hmm_beta_1` (transition) |
| `ar1_hmm_beta_1_shared` | `nll` | `ar1_hmm_1` (emission) + `hmm_beta_1_shared` (transition) |

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
