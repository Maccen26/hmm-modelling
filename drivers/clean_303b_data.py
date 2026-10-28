"""Clean a DATA303B room's raw sensor series into train/val/test arrays under `data/<room>/<tag>/`.

Sibling of `drivers/clean_dtu_data.py` for the `raw/DATA303B/climify_office_NNN.csv` export, which
holds the same sensors over 2023-01 -> 2026 but with per-device column names
(`device_<id>_co2`, ...) and a tz-aware UTC `Time` column. Only loading differs; segmentation,
CO2 floor calibration, feedback and splitting reuse the DTU pipeline unchanged.

Humidity is not cleaned here: the moisture-excess calibration needs outdoor humidity, which is
only available for the DTU period (Nov 2023 -> Feb 2024), so the default is CO2 only.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from drivers.clean_dtu_data import (
    Y_COL_RANGES,
    add_feedback,
    calibrate_baseline_co2,
    longest_gap_free_segment,
    room_slug,
    save_csv,
    save_metadata,
    split_three_way,
)
from drivers.utils import load_base_data_path


# --- Paths --------------------------------------------------------------

def series_path(room_name: str) -> Path:
    """"Room 009" -> `{DATA_PATH}/raw/DATA303B/climify_office_009.csv`."""
    number = room_name.strip().split()[-1]
    return Path(load_base_data_path()) / "raw" / "DATA303B" / f"climify_office_{number}.csv"


# --- Load and clean the raw series --------------------------------------

DEVICE_COL = re.compile(r"^device_.+_(\w+)$")


def load_raw_series(room_name: str, y_cols: tuple[str, ...]) -> tuple[pd.DataFrame, list[str]]:
    """Raw (datetime, *y_cols) readings with NaNs, out-of-range values and duplicates dropped.

    `device_<id>_<col>` columns are renamed to `<col>`, and `Time` (tz-aware UTC) becomes a naive
    UTC `datetime` -- the same convention as the DTU export, so feedback bins line up.
    `co2` is required and raises if absent; other missing columns are skipped with a warning.
    """
    df = pd.read_csv(series_path(room_name))
    df = df.rename(columns=lambda c: m.group(1) if (m := DEVICE_COL.match(c)) else c)
    df = df.rename(columns={"Time": "datetime"})

    if "co2" not in y_cols:
        raise ValueError(f"y_cols must include 'co2', got {y_cols}")
    if "co2" not in df.columns:
        raise ValueError(f"{room_name}: required column 'co2' not found in {series_path(room_name)}")

    cols = []
    for col in y_cols:
        if col != "co2" and col not in df.columns:
            print(f"  {room_name}: skipping missing column '{col}'")
            continue
        cols.append(col)

    df = df.dropna(subset=cols).copy()
    df["datetime"] = pd.to_datetime(df["datetime"], format="ISO8601", utc=True).dt.tz_convert(None)

    for col in cols:
        lo, hi = Y_COL_RANGES.get(col, (-np.inf, np.inf))
        df = df[(df[col] >= lo) & (df[col] <= hi)]

    df = (
        df.drop_duplicates(subset=["datetime", *cols])
        .sort_values("datetime")
        .reset_index(drop=True)[["datetime", *cols]]
    )
    return df, cols


# --- Driver -------------------------------------------------------------

def clean_303b_data(
    room_name: str = "Room 009",
    tag: str = "2023_2026",
    val_size: float = 0.15,
    test_size: float = 0.20,
    bin: str = "30min",
    baseline_ppm: float = 400.0,
    baseline_window: str = "8D",
    baseline_quantile: float = 0.02,
    y_cols: tuple[str, ...] = ("co2",),
) -> None:
    """Clean one DATA303B room and write headered y train/val/test CSVs to `{DATA_PATH}/<room>/<tag>/`."""
    df_raw, y_cols = load_raw_series(room_name, y_cols=y_cols)
    print(f"{room_name}: {len(df_raw)} cleaned raw observations, columns {y_cols}")
    print(f"  from {df_raw['datetime'].iloc[0]} to {df_raw['datetime'].iloc[-1]}")

    seg_df = longest_gap_free_segment(df_raw, y_cols=y_cols, bin=bin)
    start, end = seg_df.index[0], seg_df.index[-1]
    print(f"  longest gap-free {bin} segment: {len(seg_df)} bins")
    print(f"  window {start} -> {end}  ({(end - start).total_seconds() / 3600:.1f} h)")

    seg_df, co2_floor = calibrate_baseline_co2(
        seg_df, baseline=baseline_ppm, window=baseline_window, quantile=baseline_quantile,
    )
    start = seg_df.index[0]
    print(f"  dropped first {baseline_window} (floor warm-up): {len(seg_df)} bins from {start}")
    seg_df = add_feedback(seg_df, room_name, bin=bin)

    co2_floor_arr = np.asarray(co2_floor.values, dtype=float)
    print(f"  sensor floor ({baseline_window}, q={baseline_quantile:g}): "
          f"{co2_floor_arr.min():.1f} -> {co2_floor_arr.max():.1f} ppm "
          f"(drift {co2_floor_arr.max() - co2_floor_arr.min():+.1f} ppm), rebased to {baseline_ppm:.0f}")

    dates = pd.DatetimeIndex(seg_df.index)
    n = len(seg_df)
    train_idx, val_idx = split_three_way(n, val_size=val_size, test_size=test_size)

    out_df = seg_df.reset_index().rename(columns={"index": "datetime"})
    floor_df = pd.DataFrame({"floor": co2_floor_arr}, index=dates).reset_index().rename(columns={"index": "datetime"})

    slices = {
        "train": slice(None, train_idx),
        "val": slice(train_idx, val_idx),
        "test": slice(val_idx, None),
    }
    data_name = room_slug(room_name)
    meta = {}
    for arr_name, df in [("y", out_df), ("floor", floor_df)]:
        for arr_type, sl in slices.items():
            part = df.iloc[sl]
            meta[f"{arr_name}_{arr_type}_shape"] = part.shape
            save_csv(part, data_name=data_name, tag=tag, arr_name=arr_name, arr_type=arr_type)

    print(f"  split (train/val/test): {train_idx}/{val_idx - train_idx}/{n - val_idx} bins")
    print(f"  boundaries: {dates[train_idx]} | {dates[val_idx]}")

    save_metadata(
        data_name=data_name,
        tag=tag,
        room_name=room_name,
        bin=bin,
        start=start,
        end=end,
        n=n,
        val_size=val_size,
        test_size=test_size,
        train_idx=train_idx,
        val_idx=val_idx,
        dates=dates,
        baseline_ppm=baseline_ppm,
        floor=co2_floor_arr,
        baseline_window=baseline_window,
        baseline_quantile=baseline_quantile,
        y_cols=y_cols,
        meta=meta,
    )


if __name__ == "__main__":

    ROOM_LIST = [
        "Room 001",
        "Room 003",
        "Room 004",
        "Room 005",
        "Room 007",
        "Room 008",
        "Room 009",
        "Room 010",
        "Room 012",
        "Room 013",
        "Room 014",
        "Room 015",
        "Room 016",
        "Room 017",
        "Room 018",
        "Room 019",
        ]

    TAG = "2023_2026"
    VAL_SIZE = 0.15
    TEST_SIZE = 0.20
    Y_COLS = ("co2",)

    for room_name in ROOM_LIST:
        try:
            clean_303b_data(
                room_name=room_name,
                tag=TAG,
                val_size=VAL_SIZE,
                test_size=TEST_SIZE,
                y_cols=Y_COLS,
            )
        except ValueError as e:
            print(f"{room_name}: skipped -- {e}")
