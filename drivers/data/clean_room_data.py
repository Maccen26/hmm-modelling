"""Clean a DTU room's raw CO2 series into train/val/test CSVs under `data/<room>/<tag>/`.

Port of the cleaning in `summary_statistics.ipynb`: drop duplicate / NaN readings, remove the
sensor's drifting zero with a trailing rolling low quantile on the raw readings, then resample to
30 min means and linearly interpolate the gaps. Only CO2 is handled -- no covariates.
"""

import os
from pathlib import Path

import pandas as pd

from drivers.data.clean_dtu_data import room_slug, save_csv, split_three_way
from drivers.utils import load_base_data_path

OUTSIDE_CO2 = 420.0
WINDOW = "8D"
QUANTILE = 0.015
BIN = "30min"


# --- Load and clean -----------------------------------------------------

def raw_path(room_name: str) -> Path:
    return Path(load_base_data_path()) / "raw" / "dtu" / "timeseries" / f"{room_name}.csv"


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.drop_duplicates()
        .drop_duplicates(subset="datetime")
        .dropna(subset=["co2"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )


def load_co2(room_name: str) -> pd.Series:
    """Cleaned raw CO2 readings indexed by datetime."""
    df = pd.read_csv(raw_path(room_name))
    df["datetime"] = pd.to_datetime(df["datetime"])
    return clean_df(df).set_index("datetime")["co2"]


# --- Calibrate and bin --------------------------------------------------

def rolling_floor(co2: pd.Series) -> pd.Series:
    """Trailing `WINDOW` low quantile: what the sensor reads with the room empty."""
    return co2.rolling(WINDOW).quantile(QUANTILE)


def calibrate(co2: pd.Series, floor: pd.Series) -> tuple[pd.Series, float]:
    """Subtract the floor and shift so the lowest reading sits at `OUTSIDE_CO2`."""
    excess = co2 - floor
    offset = OUTSIDE_CO2 - excess.min()
    return excess + offset, offset


def bin_and_fill(series: pd.Series) -> pd.Series:
    """Resample to `BIN` means and linearly interpolate the empty bins."""
    return series.resample(BIN).mean().interpolate(method="linear")


# --- Split and save -----------------------------------------------------

def split_frames(df: pd.DataFrame, val_size: float, test_size: float) -> dict[str, pd.DataFrame]:
    """Chronological train -> val -> test split."""
    train_idx, val_idx = split_three_way(len(df), val_size=val_size, test_size=test_size)
    return {"train": df.iloc[:train_idx], "val": df.iloc[train_idx:val_idx], "test": df.iloc[val_idx:]}


def save_split(frames: dict[str, pd.DataFrame], data_name: str, tag: str, arr_name: str) -> None:
    for arr_type, part in frames.items():
        save_csv(part, data_name=data_name, tag=tag, arr_name=arr_name, arr_type=arr_type)


def save_metadata(record: dict, data_name: str, tag: str) -> None:
    save_path = os.path.join(load_base_data_path(), data_name, tag, "metadata.csv")
    pd.DataFrame([record]).to_csv(save_path, index=False)
    print(f"  wrote {save_path}")


def build_metadata(room_name: str, tag: str, y: dict[str, pd.DataFrame], offset: float) -> dict:
    """One row recording the window, the split and how to undo the calibration:
    `co2_raw = co2 + floor - offset`."""
    return {
        "room_name": room_name,
        "tag": tag,
        "bin": BIN,
        "start": y["train"]["datetime"].iloc[0],
        "end": y["test"]["datetime"].iloc[-1],
        "n_bins": sum(len(part) for part in y.values()),
        **{f"n_{key}": len(part) for key, part in y.items()},
        "outside_ppm": OUTSIDE_CO2,
        "window": WINDOW,
        "quantile": QUANTILE,
        "offset": offset,
    }


# --- Driver -------------------------------------------------------------

def clean_room_data(
    room_name: str, tag: str = "calibrated", val_size: float = 0.15, test_size: float = 0.20,
) -> None:
    """Clean one room and write y/floor train/val/test CSVs plus metadata to `{DATA_PATH}/<room>/<tag>/`."""
    co2 = load_co2(room_name)
    floor = rolling_floor(co2)
    co2_cal, offset = calibrate(co2, floor)

    y = bin_and_fill(co2_cal).rename("co2").reset_index()
    floor_binned = bin_and_fill(floor).rename("floor").reset_index()

    data_name = room_slug(room_name)
    y_split = split_frames(y, val_size, test_size)
    save_split(y_split, data_name, tag, "y")
    save_split(split_frames(floor_binned, val_size, test_size), data_name, tag, "floor")
    save_metadata(build_metadata(room_name, tag, y_split, offset), data_name, tag)
    print(f"{room_name}: {len(y)} bins, split {[len(p) for p in y_split.values()]}")


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
        "Room 016",
        "Room 019",
    ]

    for room_name in ROOM_LIST:
        clean_room_data(room_name)
