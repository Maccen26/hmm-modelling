"""Clean a DTU room's raw CO2 series into train/val/test arrays under `data/<room>/<tag>/`.

Port of the cleaning pipeline in `week_5.ipynb`: drop NaN / saturated / duplicate readings,
resample to a regular grid, keep the longest gap-free segment, build the covariate matrix
(off-day flag, weekly / daily Fourier terms, hourly weather), shift the series so its minimum
sits at the outdoor baseline, then split chronologically into train -> val -> test and z-score
the covariates on training statistics only.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from drivers.data_b1 import save_arr
from drivers.utils import load_base_data_path

WEATHER_COLS = [
    "mean_temp",
    "mean_relative_hum",
    "mean_wind_speed",
    "mean_pressure",
    "mean_cloud_cover",
    "mean_radiation",
]

SATURATION_PPM = 4000.0


# --- Paths --------------------------------------------------------------

def series_path(room_name: str) -> Path:
    return Path(load_base_data_path()) / "raw" / "dtu" / "timeseries" / f"{room_name}.csv"


def weather_path() -> Path:
    return Path(load_base_data_path()) / "raw" / "dtu" / "weather.csv"


def room_slug(room_name: str) -> str:
    """"Room 009" -> "room_009": the `data_name` the arrays are saved under."""
    return room_name.strip().lower().replace(" ", "_")


# --- Load and clean the raw series --------------------------------------

def load_raw_series(room_name: str) -> pd.DataFrame:
    """Raw (datetime, co2) readings with NaNs, sensor saturation and duplicates dropped."""
    df = pd.read_csv(series_path(room_name))
    df = df.dropna(subset=["co2"]).copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    non_saturated = df[df["co2"] <= SATURATION_PPM]
    return (
        non_saturated.drop_duplicates(subset=["datetime", "co2"])
        .sort_values("datetime")
        .reset_index(drop=True)[["datetime", "co2"]]
    )


# --- Covariates ---------------------------------------------------------

def _is_holiday_mask(dt: pd.DatetimeIndex) -> np.ndarray:
    import holidays as _holidays

    dk = _holidays.Denmark(years=sorted(set(dt.year)))
    return np.array([d.date() in dk for d in dt])


def _weather_features(datetimes, weather_cols: list[str]) -> np.ndarray:
    """Match hourly weather (UTC) to each local, tz-naive observation time."""
    if not weather_cols:
        return np.empty((len(datetimes), 0))

    weather_df = pd.read_csv(weather_path(), parse_dates=["DateFrom", "DateTo"])
    weather_df = (
        weather_df[["DateFrom", *weather_cols]]
        .dropna()
        .sort_values("DateFrom")
        .reset_index(drop=True)
    )

    local = pd.DatetimeIndex(datetimes).tz_localize(
        "Europe/Copenhagen", ambiguous="NaT", nonexistent="shift_forward"
    )
    left = pd.DataFrame({"t": local.tz_convert("UTC")})
    left["order"] = np.arange(len(left))
    left = left.sort_values("t")
    merged = pd.merge_asof(
        left, weather_df.rename(columns={"DateFrom": "t"}), on="t", direction="nearest",
    )
    merged = merged.sort_values("order")
    return merged[weather_cols].ffill().bfill().to_numpy()


def _fourier(angle, harmonics, label):
    """sin/cos pairs for each requested harmonic of a 2*pi-normalised angle."""
    cols, names = [], []
    for k in harmonics:
        cols += [np.sin(k * angle), np.cos(k * angle)]
        names += [f"sin_{label}_{k}", f"cos_{label}_{k}"]
    return cols, names


def build_covariates(
    datetimes,
    tod_harmonics: tuple[int, ...] = (1, 2, 3),
    week_harmonics: tuple[int, ...] = (1,),
    use_off_day: bool = True,
    weather_cols: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Return (X, names): an (N, D) covariate matrix and its D column names."""
    weather_cols = WEATHER_COLS if weather_cols is None else weather_cols
    dt = pd.DatetimeIndex(datetimes)
    cols, names = [], []

    if use_off_day:
        is_weekend = dt.dayofweek >= 5
        cols.append((is_weekend | _is_holiday_mask(dt)).astype(float))
        names.append("off_day")

    seconds_into_week = dt.dayofweek * 86400 + dt.hour * 3600 + dt.minute * 60 + dt.second
    c, n = _fourier(2 * np.pi * seconds_into_week / (7 * 86400), week_harmonics, "week")
    cols += c
    names += n

    seconds_into_day = dt.hour * 3600 + dt.minute * 60 + dt.second
    c, n = _fourier(2 * np.pi * seconds_into_day / 86400, tod_harmonics, "tod")
    cols += c
    names += n

    X = np.column_stack(cols) if cols else np.empty((len(dt), 0))
    return np.column_stack([X, _weather_features(datetimes, weather_cols)]), [*names, *weather_cols]


# --- Aggregation, calibration, splitting --------------------------------

def longest_gap_free_segment(df_raw: pd.DataFrame, bin: str = "30min") -> pd.Series:
    """Resample to a regular grid and return the longest run of consecutively filled bins.

    An empty bin is a gap longer than one bin, which splits the grid into segments whose
    neighbours are exactly one bin apart -- what the discrete models assume.
    """
    binned = df_raw.set_index("datetime")["co2"].resample(bin).mean()

    filled = binned.notna().values
    seg_id = np.cumsum(~filled)                  # increments at every empty bin
    segments = [binned[filled & (seg_id == s)] for s in np.unique(seg_id[filled])]
    return max(segments, key=len)


def calibrate_baseline(ys: np.ndarray, baseline: float = 400.0) -> np.ndarray:
    """Shift a CO2 series so its minimum sits at `baseline` ppm (sensor-floor correction).

    A constant offset: every difference in the series is preserved, only the level moves.
    """
    ys = np.asarray(ys, dtype=float)
    return ys - np.min(ys) + baseline


def split_three_way(n: int, val_size: float, test_size: float) -> tuple[int, int]:
    """Chronological boundaries (train_idx, val_idx): train -> val -> test."""
    if not (0 < val_size + test_size < 1):
        raise ValueError(
            f"val_size + test_size must lie in (0, 1), got {val_size} + {test_size} "
            f"= {val_size + test_size}"
        )
    if val_size < 0 or test_size < 0:
        raise ValueError(f"val_size and test_size must be non-negative, got {val_size}, {test_size}")

    train_idx = int(n * (1 - val_size - test_size))
    val_idx = int(n * (1 - test_size))
    return train_idx, val_idx


def standardise(X: np.ndarray, train_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score X with training statistics only, so nothing from val/test leaks into the scaling."""
    mean = X[:train_idx].mean(axis=0)
    std = X[:train_idx].std(axis=0)
    std = np.where(std == 0, 1.0, std)           # a flag can be constant in a short segment
    return (X - mean) / std, mean, std


# --- Driver -------------------------------------------------------------

def clean_dtu_data(
    room_name: str = "Room 009",
    tag: str = "30min",
    val_size: float = 0.15,
    test_size: float = 0.20,
    bin: str = "30min",
    baseline_ppm: float = 400.0,
    tod_harmonics: tuple[int, ...] = (1, 2, 3),
    week_harmonics: tuple[int, ...] = (1,),
    use_off_day: bool = True,
    weather_cols: list[str] | None = None,
) -> None:
    """Clean one DTU room and write y/X train/val/test arrays to `{DATA_PATH}/<room>/<tag>/`."""
    df_raw = load_raw_series(room_name)
    print(f"{room_name}: {len(df_raw)} cleaned raw observations")
    print(f"  from {df_raw['datetime'].iloc[0]} to {df_raw['datetime'].iloc[-1]}")

    seg = longest_gap_free_segment(df_raw, bin=bin)
    start, end = seg.index[0], seg.index[-1]
    print(f"  longest gap-free {bin} segment: {len(seg)} bins")
    print(f"  window {start} -> {end}  ({(end - start).total_seconds() / 3600:.1f} h)")

    ys_uncal = np.asarray(seg.values, dtype=float)
    ys = calibrate_baseline(ys_uncal, baseline=baseline_ppm)
    offset = float(np.min(ys_uncal) - baseline_ppm)   # the constant subtracted
    print(f"  baseline offset applied: {-offset:+.2f} ppm")

    dates = pd.DatetimeIndex(seg.index)
    X, covariate_cols = build_covariates(
        dates,
        tod_harmonics=tod_harmonics,
        week_harmonics=week_harmonics,
        use_off_day=use_off_day,
        weather_cols=weather_cols,
    )

    train_idx, val_idx = split_three_way(len(ys), val_size=val_size, test_size=test_size)
    X_std, x_mean, x_std = standardise(X, train_idx)

    slices = {
        "train": slice(None, train_idx),
        "val": slice(train_idx, val_idx),
        "test": slice(val_idx, None),
    }
    data_name = room_slug(room_name)
    meta = {}
    for arr_name, arr in (("y", ys), ("X", X_std)):
        for arr_type, sl in slices.items():
            part = np.atleast_2d(arr[sl])
            if part.shape[0] == 1 and arr_name == "y":
                part = part.T
            meta[f"{arr_name}_{arr_type}_shape"] = part.shape
            save_arr(arr=part, data_name=data_name, tag=tag, arr_name=arr_name, arr_type=arr_type)

    print(f"  split (train/val/test): {train_idx}/{val_idx - train_idx}/{len(ys) - val_idx} bins")
    print(f"  boundaries: {dates[train_idx]} | {dates[val_idx]}")
    print(f"  covariates: {len(covariate_cols)} ({', '.join(covariate_cols)})")

    save_metadata(
        data_name=data_name,
        tag=tag,
        room_name=room_name,
        bin=bin,
        start=start,
        end=end,
        n=len(ys),
        val_size=val_size,
        test_size=test_size,
        train_idx=train_idx,
        val_idx=val_idx,
        dates=dates,
        baseline_ppm=baseline_ppm,
        offset=offset,
        covariate_cols=covariate_cols,
        x_mean=x_mean,
        x_std=x_std,
        meta=meta,
    )


def save_metadata(
    data_name, tag, room_name, bin, start, end, n, val_size, test_size, train_idx, val_idx,
    dates, baseline_ppm, offset, covariate_cols, x_mean, x_std, meta,
) -> None:
    """One-row `metadata.csv` recording the window, the split and the reversible transforms."""
    record = {
        "room_name": room_name,
        "data_name": data_name,
        "tag": tag,
        "bin": bin,
        "start": start,
        "end": end,
        "n_bins": n,
        "val_size": val_size,
        "test_size": test_size,
        "n_train": train_idx,
        "n_val": val_idx - train_idx,
        "n_test": n - val_idx,
        "train_end": dates[train_idx],
        "val_end": dates[val_idx],
        "baseline_ppm": baseline_ppm,
        "baseline_offset": offset,           # ys = ys_uncalibrated - baseline_offset
        "covariate_cols": "|".join(covariate_cols),
    }
    record.update(meta)
    # Train mean/std per covariate, so the standardisation can be undone or reapplied.
    record.update({f"mean_{c}": m for c, m in zip(covariate_cols, x_mean)})
    record.update({f"std_{c}": s for c, s in zip(covariate_cols, x_std)})

    save_path = os.path.join(load_base_data_path(), data_name, tag, "metadata.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pd.DataFrame([record]).to_csv(save_path, index=False, sep=",")
    print(f"  wrote {save_path}")


if __name__ == "__main__":
    ROOM_NAME = "Room 009"
    TAG = "30min"
    VAL_SIZE = 0.15
    TEST_SIZE = 0.20

    clean_dtu_data(
        room_name=ROOM_NAME,
        tag=TAG,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
    )
