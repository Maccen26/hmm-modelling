# Data drivers

```bash
uv run python -m drivers.data.clean_room_data_anchor "Room 015" ["Room 009" ...]   # raw CO2 -> {DATA_PATH}/room_015/calibrated_anchor/
uv run python -m drivers.data.create_data_plots --room 15                           # Data-chapter figures -> plots/room_015/summary/
uv run python -m drivers.data.create_summary_tables --gap-room 15                   # Data-chapter tables  -> Thesis/Tables/data/
```

- `clean_room_data_anchor`: subtracts a causal fresh-air baseline from the raw CO2 readings (see its docstring for the two update rules), bins to 30 min, splits train → val → test, and writes `y_*.csv`, `floor_*.csv`, `anchors.csv` and `metadata.csv`.
- `create_data_plots`: raw gaps and their interpolation, the raw readings against the anchor baseline, and, on the anchor-calibrated series, log CO2, ACF/PACF, diurnal/weekly profiles, off days and the split.
- `create_summary_tables`: raw CO2 quantiles, waiting times, gaps and feedback counts.
- `clean_room_data`: the shared load / split / save / feedback helpers the scripts above use.
