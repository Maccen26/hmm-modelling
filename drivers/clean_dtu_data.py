"""Clean a DTU room's raw CO2 series into train/val/test arrays under `data/<room>/<tag>/`.

Port of the cleaning pipeline in `week_5.ipynb`: drop NaN / saturated / duplicate readings,
resample to a regular grid, keep the longest gap-free segment, build the covariate matrix
(off-day flag, weekly / daily Fourier terms, hourly weather), subtract the sensor's slowly
drifting zero so an empty room sits at the outdoor baseline, then split chronologically into
train -> val -> test and z-score the covariates on training statistics only.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from drivers.data_b1 import save_arr
from drivers.utils import build_covariates, load_base_data_path, standardise

SATURATION_PPM = 4000.0


# --- Paths --------------------------------------------------------------

def series_path(room_name: str) -> Path:
    return Path(load_base_data_path()) / "raw" / "dtu" / "timeseries" / f"{room_name}.csv"


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


def sensor_floor(
    seg: pd.Series, window: str = "28D", quantile: float = 0.02, min_periods: int = 10,
) -> pd.Series:
    """Estimate what the sensor reads with the room empty, as a function of time.

    A centred rolling low quantile: for each bin, the window is every bin within half of
    `window` either side of it, and the floor is that window's `quantile`. Stepping one bin
    forward drops one observation off the back and adds one to the front, so the estimate can
    only creep -- which is the whole point, since a floor that jumps injects a fake step into
    the series (see `calibrate_baseline`).

    Choosing `quantile`: it must stay below the fraction of bins in which the room is genuinely
    empty, or the "floor" starts tracking occupancy and the calibration subtracts real CO2.
    For Room 009 ~16% of bins sit below 500 ppm, so 0.02 is comfortably clear of occupancy
    while still resting on ~27 observations rather than a single noisy minimum.

    The window straddles the train/val/test boundaries, so a held-out bin's calibration is
    informed by held-out data. That is a property of sensor calibration rather than of the
    model -- no CO2 *level* crosses the split, only a low quantile of the surrounding weeks --
    but it is worth stating explicitly when the held-out scores are reported.
    """
    return seg.rolling(window, center=True, min_periods=min_periods).quantile(quantile)


def calibrate_baseline(
    seg: pd.Series, baseline: float = 400.0, window: str = "28D", quantile: float = 0.02,
) -> tuple[pd.Series, pd.Series]:
    """Remove the sensor's drifting zero: `y_cal = y_raw - floor(t) + baseline`.

    The NDIR sensors drift, so the level at which a room reads "empty" wanders by ~100 ppm
    over a few months. Subtracting a smoothly varying floor takes that out while leaving the
    short-run dynamics -- the peaks, decays and gaps the HMM actually models -- untouched.

    Must run on the *aggregated* series, not the raw readings: a 30min bin averages several
    readings, so calibrating the raw series leaves the binned floor above `baseline`.

    Returns the calibrated series and the floor that was subtracted (saved alongside the data
    so the transform can be undone).
    """
    floor = sensor_floor(seg, window=window, quantile=quantile)
    if floor.isna().any():
        raise ValueError(
            f"sensor floor undefined for {int(floor.isna().sum())} of {len(floor)} bins -- "
            f"the segment is too short for a {window} window"
        )
    return seg - floor + baseline, floor



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


# --- Driver -------------------------------------------------------------

def clean_dtu_data(
    room_name: str = "Room 009",
    tag: str = "30min",
    val_size: float = 0.15,
    test_size: float = 0.20,
    bin: str = "30min",
    baseline_ppm: float = 400.0,
    baseline_window: str = "28D",
    baseline_quantile: float = 0.02,
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

    seg, floor = calibrate_baseline(
        seg, baseline=baseline_ppm, window=baseline_window, quantile=baseline_quantile,
    )
    ys = np.asarray(seg.values, dtype=float)
    floor_arr = np.asarray(floor.values, dtype=float)
    print(f"  sensor floor ({baseline_window}, q={baseline_quantile:g}): "
          f"{floor_arr.min():.1f} -> {floor_arr.max():.1f} ppm "
          f"(drift {floor_arr.max() - floor_arr.min():+.1f} ppm), rebased to {baseline_ppm:.0f}")

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
    for arr_name, arr in (("y", ys), ("X", X_std), ("floor", floor_arr)):
        for arr_type, sl in slices.items():
            part = np.atleast_2d(arr[sl])
            if part.shape[0] == 1 and arr_name in ("y", "floor"):
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
        floor=floor_arr,
        baseline_window=baseline_window,
        baseline_quantile=baseline_quantile,
        covariate_cols=covariate_cols,
        x_mean=x_mean,
        x_std=x_std,
        meta=meta,
    )


def save_metadata(
    data_name, tag, room_name, bin, start, end, n, val_size, test_size, train_idx, val_idx,
    dates, baseline_ppm, floor, baseline_window, baseline_quantile,
    covariate_cols, x_mean, x_std, meta,
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
        # The floor is per-bin, so it is saved as floor_{train,val,test}.csv rather than here;
        # y_uncalibrated = y + floor - baseline_ppm. These are the knobs and its span.
        "baseline_window": baseline_window,
        "baseline_quantile": baseline_quantile,
        "baseline_floor_start": floor[0],
        "baseline_floor_end": floor[-1],
        "baseline_floor_drift": floor.max() - floor.min(),
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

    ROOM_LIST = [
        "Room 001",
        "Room 003", 
        "Room 004",
        "Room 005",
        "Room 007",
        "Room 008",
        "Room 009", 
        "Room 012", 
        "Room 013", 
        "Room 014", 
        "Room 015", 
        "Room 016"
        ]

    ROOM_NAME = "Room 012"
    TAG = "30min"
    VAL_SIZE = 0.15
    TEST_SIZE = 0.20

    for room_name in ROOM_LIST:
        clean_dtu_data(
            room_name=room_name,
            tag=TAG,
            val_size=VAL_SIZE,
            test_size=TEST_SIZE,
        )

    #clean_dtu_data(
    #    room_name=ROOM_NAME,
    #    tag=TAG,
    #    val_size=VAL_SIZE,
    #    test_size=TEST_SIZE,
    #)
