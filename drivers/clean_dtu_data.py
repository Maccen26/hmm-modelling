"""Clean a DTU room's raw sensor series into train/val/test arrays under `data/<room>/<tag>/`.

Port of the cleaning pipeline in `week_5.ipynb`: drop NaN / saturated / duplicate readings,
resample to a regular grid, keep the longest gap-free segment, subtract the CO2 sensor's
slowly drifting zero so an empty room sits at the outdoor baseline, then split
chronologically into train -> val -> test.

Covariate construction (off-day flag, weekly / daily Fourier terms, hourly weather) is a
fit-time concern, not a clean-time one -- see `build_covariates` / `standardise` in
`drivers/utils.py`, used directly by the week 5/6 notebooks.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from drivers.utils import load_base_data_path

SATURATION_PPM = 4000.0
Y_COL_RANGES = {            # inclusive plausibility bounds, applied per column
    "co2": (0.0, SATURATION_PPM),
    "humidity": (0.0, 100.0),
    "temperature": (-10.0, 60.0),
}


# --- Paths --------------------------------------------------------------

def series_path(room_name: str) -> Path:
    return Path(load_base_data_path()) / "raw" / "dtu" / "timeseries" / f"{room_name}.csv"


def room_slug(room_name: str) -> str:
    """"Room 009" -> "room_009": the `data_name` the arrays are saved under."""
    return room_name.strip().lower().replace(" ", "_")


# --- Load and clean the raw series --------------------------------------

def load_raw_series(room_name: str, y_cols: tuple[str, ...]) -> tuple[pd.DataFrame, list[str]]:
    """Raw (datetime, *y_cols) readings with NaNs, out-of-range values and duplicates dropped.

    `co2` is required -- it drives segmentation and calibration -- and raises if absent.
    Any other requested column that is missing from the file is skipped with a warning.
    Returns the cleaned frame alongside the list of columns that actually survived.
    """
    df = pd.read_csv(series_path(room_name))

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
    df["datetime"] = pd.to_datetime(df["datetime"])

    for col in cols:
        lo, hi = Y_COL_RANGES.get(col, (-np.inf, np.inf))
        df = df[(df[col] >= lo) & (df[col] <= hi)]

    df = (
        df.drop_duplicates(subset=["datetime", *cols])
        .sort_values("datetime")
        .reset_index(drop=True)[["datetime", *cols]]
    )
    return df, cols


# --- Aggregation, calibration, splitting --------------------------------

def longest_gap_free_segment(df_raw: pd.DataFrame, y_cols: list[str], bin: str = "30min") -> pd.DataFrame:
    """Resample to a regular grid and return the longest run of consecutively filled bins.

    An empty bin is a gap longer than one bin, which splits the grid into segments whose
    neighbours are exactly one bin apart -- what the discrete models assume. A bin counts as
    filled only when every requested column is non-NaN, so the returned segment is complete
    across all of `y_cols`.
    """
    binned = df_raw.set_index("datetime")[y_cols].resample(bin).mean()

    filled = binned.notna().all(axis=1).values
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


def calibrate_baseline_co2(
    seg_df: pd.DataFrame, baseline: float = 400.0, window: str = "28D", quantile: float = 0.02,
) -> tuple[pd.DataFrame, pd.Series]:
    """Remove the CO2 sensor's drifting zero: `co2_cal = co2_raw - floor(t) + baseline`.

    The NDIR sensors drift, so the level at which a room reads "empty" wanders by ~100 ppm
    over a few months. Subtracting a smoothly varying floor takes that out while leaving the
    short-run dynamics -- the peaks, decays and gaps the HMM actually models -- untouched.
    Only the `co2` column is touched; any other y columns pass through unchanged.

    Must run on the *aggregated* series, not the raw readings: a 30min bin averages several
    readings, so calibrating the raw series leaves the binned floor above `baseline`.

    Returns the frame with `co2` calibrated and the floor that was subtracted (saved alongside
    the data so the transform can be undone).
    """
    floor = sensor_floor(seg_df["co2"], window=window, quantile=quantile)
    if floor.isna().any():
        raise ValueError(
            f"sensor floor undefined for {int(floor.isna().sum())} of {len(floor)} bins -- "
            f"the segment is too short for a {window} window"
        )
    seg_df = seg_df.copy()
    seg_df["co2"] = seg_df["co2"] - floor + baseline
    return seg_df, floor


# --- Humidity calibration -------------------------------------------------

P_ATM_HPA = 1013.0


def saturation_vapour_pressure(temp_c):
    """Magnus formula over water (Sonntag 1990), in hPa; `temp_c` in °C, valid ~-45 to 60 °C."""
    return 6.112 * np.exp(17.62 * temp_c / (243.12 + temp_c))


def vapour_pressure(rh, temp_c):
    """Actual vapour pressure in hPa from relative humidity `rh` (%) and temperature (°C)."""
    return rh / 100.0 * saturation_vapour_pressure(temp_c)


def humidity_ratio(rh, temp_c, p: float = P_ATM_HPA):
    """Humidity ratio in g water vapour per kg dry air; `p` is total air pressure in hPa."""
    e = vapour_pressure(rh, temp_c)
    return 622.0 * e / (p - e)


def outdoor_humidity_path() -> Path:
    return Path(load_base_data_path()) / "raw" / "dtu" / "humidity.csv"


def load_outdoor_humidity() -> pd.Series:
    """Hourly outdoor humidity ratio `x_out` (g/kg), indexed by UTC `DateTo`."""
    df = pd.read_csv(outdoor_humidity_path())
    df["DateTo"] = pd.to_datetime(df["DateTo"], utc=True)
    df = df.dropna().sort_values("DateTo")
    x_out = humidity_ratio(df["mean_relative_hum"], df["mean_temp"])
    return pd.Series(x_out.values, index=pd.DatetimeIndex(df["DateTo"]), name="x_out")


def calibrate_humidity(seg_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Replace indoor RH with the moisture excess `Δx = x_in - x_out` (g/kg dry air).

    `x_in` uses the room's own `temperature`; `x_out` is the hourly DMI series matched to each
    local, tz-naive bin time (same convention as `drivers.utils.weather_features`). Raises if
    the segment runs outside the outdoor data's span rather than extrapolating.

    Returns the frame with `humidity` replaced and the `x_out` that was subtracted.
    """
    x_out_hourly = load_outdoor_humidity()

    local = pd.DatetimeIndex(seg_df.index).tz_localize(
        "Europe/Copenhagen", ambiguous="NaT", nonexistent="shift_forward"
    )
    utc = local.tz_convert("UTC")
    lo, hi = x_out_hourly.index[0], x_out_hourly.index[-1]
    tol = pd.Timedelta("1h")
    if utc.min() < lo - tol or utc.max() > hi + tol:
        raise ValueError(
            f"segment {utc.min()} -> {utc.max()} extends beyond outdoor humidity data {lo} -> {hi}"
        )

    left = pd.DataFrame({"t": utc, "order": np.arange(len(utc))}).sort_values("t")
    right = x_out_hourly.rename_axis("t").reset_index()
    merged = pd.merge_asof(left.dropna(subset=["t"]), right, on="t", direction="nearest")
    x_out = (
        merged.set_index("order")["x_out"]
        .reindex(np.arange(len(utc)))
        .ffill().bfill()        # bins dropped as ambiguous at the DST switch
        .to_numpy()
    )
    x_out = pd.Series(x_out, index=seg_df.index, name="x_out")

    x_in = humidity_ratio(seg_df["humidity"], seg_df["temperature"])
    seg_df = seg_df.copy()
    seg_df["humidity"] = x_in - x_out
    return seg_df, x_out


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


# --- IO -------------------------------------------------------------------

def save_csv(df: pd.DataFrame, data_name: str, tag: str, arr_name: str, arr_type: str) -> None:
    if len(df) == 0:
        print(f"Warning: {arr_name}_{arr_type} is empty. Not saving.")
        return

    base_path = load_base_data_path()
    save_path = os.path.join(base_path, f"{data_name}/{tag}/{arr_name}_{arr_type}.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)


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
    y_cols: tuple[str, ...] = ("co2", "humidity"),
) -> None:
    """Clean one DTU room and write headered y train/val/test CSVs to `{DATA_PATH}/<room>/<tag>/`."""
    calibrate_hum = "humidity" in y_cols
    drop_temperature = calibrate_hum and "temperature" not in y_cols
    load_cols = (*y_cols, "temperature") if drop_temperature else y_cols
    df_raw, y_cols = load_raw_series(room_name, y_cols=load_cols)
    calibrate_hum = calibrate_hum and "humidity" in y_cols and "temperature" in y_cols
    print(f"{room_name}: {len(df_raw)} cleaned raw observations, columns {y_cols}")
    print(f"  from {df_raw['datetime'].iloc[0]} to {df_raw['datetime'].iloc[-1]}")

    seg_df = longest_gap_free_segment(df_raw, y_cols=y_cols, bin=bin)
    start, end = seg_df.index[0], seg_df.index[-1]
    print(f"  longest gap-free {bin} segment: {len(seg_df)} bins")
    print(f"  window {start} -> {end}  ({(end - start).total_seconds() / 3600:.1f} h)")

    seg_df, co2_floor = calibrate_baseline_co2(
        seg_df, baseline=baseline_ppm, window=baseline_window, quantile=baseline_quantile,
    )

    x_out = None
    if calibrate_hum:
        seg_df, x_out = calibrate_humidity(seg_df)
        dx = seg_df["humidity"]
        print(f"  humidity -> moisture excess x_in - x_out: "
              f"{dx.min():.2f} -> {dx.max():.2f} g/kg (mean {dx.mean():.2f})")
    if drop_temperature:
        seg_df = seg_df.drop(columns="temperature")
        y_cols = [c for c in y_cols if c != "temperature"]

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
    arrays = [("y", out_df), ("floor", floor_df)]
    if x_out is not None:
        arrays.append(("x_out", x_out.rename_axis("datetime").reset_index()))
    for arr_name, df in arrays:
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


def save_metadata(
    data_name, tag, room_name, bin, start, end, n, val_size, test_size, train_idx, val_idx,
    dates, baseline_ppm, floor, baseline_window, baseline_quantile, y_cols, meta,
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
        # co2_uncalibrated = co2 + floor - baseline_ppm. These are the knobs and its span.
        "baseline_window": baseline_window,
        "baseline_quantile": baseline_quantile,
        "baseline_floor_start": floor[0],
        "baseline_floor_end": floor[-1],
        "baseline_floor_drift": floor.max() - floor.min(),
        "y_cols": "|".join(y_cols),
        # humidity = x_in - x_out (g/kg dry air); x_out saved as x_out_{train,val,test}.csv
        "humidity_unit": "g/kg (x_in - x_out)" if "humidity" in y_cols else "",
    }
    record.update(meta)

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

    TAG = "30min"
    VAL_SIZE = 0.15
    TEST_SIZE = 0.20
    Y_COLS = ("co2", "humidity")

    for room_name in ROOM_LIST:
        clean_dtu_data(
            room_name=room_name,
            tag=TAG,
            val_size=VAL_SIZE,
            test_size=TEST_SIZE,
            y_cols=Y_COLS,
        )
