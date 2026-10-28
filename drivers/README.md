# Drivers

Runnable entry points on top of `src/api/v5`. Run everything from the repo root with module syntax (`uv run python -m drivers.<folder>.<script>`).

| Folder | What it does | Writes to |
|---|---|---|
| `data/` | Cleans raw sensor exports into train/val/test CSVs (`clean_303b_data`, `clean_dtu_data`) and downloads outdoor humidity (`fetch_humidity`) | `{DATA_PATH}/room_XXX/30min/` |
| `models/` | One driver per model variant, plus `run` to fit them all in seed order and `evaluate` for validation metrics and plots. See `models/README.md` | `results/`, `plots/` |
| `analysis/` | Post-hoc tools for saved fits: `replot`, `decode_states`, `show_params`, `state_confusion`, `feedback_states`, `transition_plots` | `plots/` |
| `legacy/` | The old b1/v4 workflow. Kept for reference; don't extend it | |

`utils.py` holds the shared IO and covariate helpers (`load_y_df`, `build_covariates`, `save_model`/`load_model`, metrics).

Typical order: `data` → `models.run` → `analysis.*`. `results/` and `plots/` live at the repo root.
