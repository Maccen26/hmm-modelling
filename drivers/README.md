# Drivers

Runnable entry points on top of `src/api/v5`. Run everything from the repo root with module syntax (`uv run python -m drivers.<folder>.<script>`).

| Folder | What it does | Writes to |
|---|---|---|
| `data/` | Cleans a room's raw CO2 with the causal anchor baseline (`clean_room_data_anchor`), and makes the Data-chapter plots and tables (`create_data_plots`, `create_summary_tables`). See `data/README.md` | `{DATA_PATH}/room_XXX/calibrated_anchor/`, `plots/room_XXX/summary/`, `Thesis/Tables/data/` |
| `models/` | `random_search` fits the four-model chain (seeded start + random starts, best admissible start kept). See `models/README.md` | `results/`, `plots/` |
| `analysis/` | Thesis tables, diagnostic plots and state analysis for saved fits. See `analysis/README.md` | `Thesis/Tables/`, `plots/`, `results/` |
| `simulation/` | Sampled state paths and the state analysis built on them, plus the feedback sample-size simulation. See `simulation/README.md` | `results/`, `plots/` |

`utils.py` holds the shared IO, covariate and LaTeX-table helpers (`load_y_df`, `build_covariates`, `save_model`/`load_model`, `save_thesis_table`).

Typical order: `data.clean_room_data_anchor` → `models.random_search` → `analysis.*` / `simulation.*`. `results/` and `plots/` live at the repo root, laid out as `{room}/{tag}/states_{K}/`; the tag defaults to `calibrated_anchor` (`TAG` in `models/utils.py`).
