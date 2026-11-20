# Model drivers

`random_search` fits the model chain HMM(1) → HMM(β,1) → AR(1)-HMM(1) → AR(1)-HMM(β,1) on one room for one number of states.
Run from the repo root with module syntax:

```bash
uv run python -m drivers.models.random_search --room 15 --states 4                 # 10 random starts per model
uv run python -m drivers.models.random_search --room 15 --states 4 --inits 20      # more random starts
uv run python -m drivers.models.random_search --room 15 --states 4 --tag calibrated_anchor
```

`--room` / `-r` is the room number (`15` → `room_015`), `--states` / `-K` the number of hidden states, and `--tag` the data folder the room CSVs are read from (`{DATA_PATH}/{room}/{tag}/`, default `calibrated_anchor`).
Observation columns and covariates are shared defaults in `utils.py` (`OBS_COLS`, `COVARIATE_COLS`).

## Search

Every model gets one seeded start plus `--inits` random starts:

- **Seeded:** `hmm_1` starts from quantiles of the data. Each later model starts from the models already chosen earlier in the chain: `hmm_beta_1` and `ar1_hmm_1` from `hmm_1`, and `ar1_hmm_beta_1` from the `ar1_hmm_1` emission plus the `hmm_beta_1` transition.
- **Random:** μ at random data quantiles, σ a random fraction of the data std, φ ∈ (0, 0.95), a diagonal-heavy random Γ and, for the β models, β ~ N(0, 0.5²).

Each start is fitted with L-BFGS. A start is rejected if it shows state collapse: σ < 1 ppm, σ > 10⁴ ppm, φ ≥ 0.999, two states that duplicate each other, or non-finite parameters or forecasts.
The admissible start with the lowest mean validation RMSE over `HORIZONS` is kept and seeds the next model.

## Losses

The objective is the NLL plus an optional ridge penalty `beta_lam · Σβ²` on the transition covariate effects (`Loss(beta_lam)`).
The β models use λ_β = 1e-2 (their betas diverge under plain NLL). `hmm_1` and `ar1_hmm_1` use plain NLL. The losses are fixed in `CHAIN` in `random_search.py`.

## Outputs

Both folders sit at the repo root, not inside `drivers/`.

- `results/{room}/{tag}/states_{K}/{slug}.pkl`: the chosen fit, its config, the training data and which start won (`init`)
- `results/{room}/{tag}/states_{K}/candidates.csv`: every start, with its loss, validation metrics, parameters and rejection reason
- `results/{room}/{tag}/states_{K}/val_metrics.csv` / `test_metrics.csv`: R², RMSE and MAPE per horizon for the chosen models. Test forecasts filter through train and validation first, so they start from the state the model would be in online.

Diagnostic plots of the fits are made by `drivers.analysis.replot` (see `../analysis/README.md`).

`evaluate.py` holds the rolling forecasts, metrics and diagnostic plot helpers. `utils.py` holds the config, data loading, likelihood and seeding helpers.
