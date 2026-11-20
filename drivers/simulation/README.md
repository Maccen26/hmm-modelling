# Simulation drivers

All of them read a saved `random_search` fit from `results/{room}/{tag}/states_{K}/`. The tag defaults to `calibrated_anchor`.

## Sampled state paths

```bash
uv run python -m drivers.simulation.sample_paths ar1_hmm_beta_1 --room 15 -K 4      # 100,000 paths
uv run python -m drivers.simulation.state_analysis ar1_hmm_beta_1 --room 15 -K 4
```

- `sample_paths` runs the forward filter over the whole observed series (train + val + test) and draws state paths from P(z_1:T | y_1:T) with `BackwardSampler`. It saves the paths, CO2, timestamps and off-day flag to `results/.../states_{K}/simulation/{model}_paths.npz` (about 37 MB for 100,000 paths).
- `state_analysis` reads that file. For each state it computes the mean drift y_t - y_{t-1} and the share of time spent in the state, per path (mean and 2.5-97.5 % range across paths) → `{model}_state_summary.csv`. It also plots the mean state probability over the day, for working days (working hours shaded) and for off days → `plots/.../states_{K}/simulation/`.

## Feedback sample size

```bash
uv run python -m drivers.simulation.sample_size --room 15 -K 4 --p 0 0 0 0.05
uv run python -m drivers.simulation.plot_sample_size --room 15 -K 4
```

`sample_size` simulates datasets from a saved fit and estimates P(feedback | state) and the vote cut-points as the number of votes grows. Outputs go to `results/.../states_{K}/sample_size/` and `plots/.../sample_size/`.
