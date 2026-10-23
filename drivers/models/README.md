# Model drivers

Each file fits one model, saves it, and evaluates it on the validation split.
Run from the repo root with module syntax:

```bash
uv run python -m drivers.models.<driver> [--states K]
```

`--states` / `-K` sets the number of hidden states. Without it, the driver uses the default in its `__main__` block, where the room, tag, observation columns, and covariates are also set.

## Fit order

Drivers don't refit their seed models; they load them from `results/` for the same room/tag/states.
If a seed is missing, the driver raises an error naming the command to run. Fit in this order:

| Driver | Seeded from |
|---|---|
| `hmm_1` | quantiles of the data |
| `hmm_beta_1`, `hmm_beta_1_shared`, `ar1_hmm_1` | `hmm_1` |
| `ar1_hmm_beta_1` | `ar1_hmm_1` (emission) + `hmm_beta_1` (transition) |
| `ar1_hmm_beta_1_shared` | `ar1_hmm_1` (emission) + `hmm_beta_1_shared` (transition) |

Example: fit every model with 6 states:

```bash
for m in hmm_1 hmm_beta_1 hmm_beta_1_shared ar1_hmm_1 ar1_hmm_beta_1 ar1_hmm_beta_1_shared; do
  uv run python -m drivers.models.$m --states 6
done
```

## Outputs

- `results/{room}/{tag}/states_{K}/{model}.pkl`: the fitted model, its config, and the training data
- `results/{room}/{tag}/states_{K}/predictive_metrics.csv`: validation R², RMSE, and MAPE per forecast horizon, shared by all models
- `plots/{room}/{tag}/states_{K}/{model}.png` and `{model}_first_week.png`: residual diagnostics and forecast plots
