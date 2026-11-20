# Analysis drivers

Tools that read the saved `random_search` fits in `results/{room}/{tag}/states_{K}/`. None of them refit. Tools taking `--tag` default to `calibrated_anchor`.

## Thesis tables and plots

```bash
uv run python -m drivers.analysis.model_comparison_table results/room_015/calibrated_anchor --states 3 4 5 6 --horizon 6
uv run python -m drivers.analysis.test_metrics_table results/room_015/calibrated_anchor \
    --rows hmm_1:4 hmm_beta_1:4 ar1_hmm_1:4 ar1_hmm_beta_1:3 ar1_hmm_beta_1:4
uv run python -m drivers.analysis.standard_errors results/room_015/calibrated_anchor -K 4   # writes {model}_standard_errors.csv
uv run python -m drivers.analysis.params_table results/room_015/calibrated_anchor -K 4      # reads the standard errors
uv run python -m drivers.analysis.replot --room 15 --states 3 4 5 6 7                       # validation / first-week / training-residual plots
```

The tables are written to `Thesis/Tables/<name>.tex`, named after the results folder. `predictive_metrics_across_states_table` writes the RMSE/MAPE-per-K table and holds the metrics loaders the two table scripts above share.

## State analysis

```bash
uv run python -m drivers.analysis.show_params ar1_hmm_beta_1 --room 15 -K 4
uv run python -m drivers.analysis.decode_states ar1_hmm_beta_1 --room 15 -K 4 [--split train] [--filtered]
uv run python -m drivers.analysis.feedback_states ar1_hmm_beta_1 --room 15 -K 4 [--split train|val|test] [--paths 10000]
uv run python -m drivers.analysis.feedback_states_full ar1_hmm_beta_1 --room 15 -K 4
uv run python -m drivers.analysis.transition_plots ar1_hmm_beta_1 --room 15 -K 4 --state 2 [--week 2023-11-27]
uv run python -m drivers.analysis.state_confusion ar1_hmm_beta_1 --room 15 -K 4 --horizon 4 [--split val|test]
```

- `decode_states` plots smoothed (or `--filtered`) state probabilities over a window of the data.
- `feedback_states` samples state paths with `BackwardSampler` and plots which states the feedback events fall in. `feedback_states_full` does the same over the whole series.
- `transition_plots` plots the covariate-driven transition probabilities out of one state over a normal week.
- `state_confusion` compares the H-step predicted state with the smoothed state.

Simulation-based tools live in `drivers/simulation/` (see its `README.md`).
