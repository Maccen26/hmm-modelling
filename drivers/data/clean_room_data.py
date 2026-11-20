"""Shared helpers for cleaning a DTU room's raw CO2 series: load, de-duplicate, split and save.

The calibration itself lives in `clean_room_data_anchor`, which writes the `calibrated_anchor` tag.
"""

import os
from pathlib import Path

import pandas as pd

from drivers.utils import load_base_data_path

OUTSIDE_CO2 = 420.0
BIN = "30min"


def room_slug(room_name: str) -> str:
    """"Room 009" -> "room_009": the `data_name` the arrays are saved under."""
    return room_name.strip().lower().replace(" ", "_")


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


# --- Occupant feedback --------------------------------------------------

def feedback_path(room_name: str) -> Path:
    return Path(load_base_data_path()) / "raw" / "dtu" / "feedback" / f"{room_name}.csv"


def add_feedback(seg_df: pd.DataFrame, room_name: str, bin: str = BIN) -> pd.DataFrame:
    """Add a binary `feedback` column: 1 if any feedback was submitted within that bin, else 0.

    Feedback timestamps are floored onto the same left-labelled grid `resample` produces, so a
    response at 13:13 lands in the 13:00 bin. Rooms without a feedback file get all zeros.
    """
    seg_df = seg_df.copy()
    seg_df["feedback"] = 0

    path = feedback_path(room_name)
    if not path.exists():
        print(f"  {room_name}: no feedback file, feedback column is all zeros")
        return seg_df

    fb_bins = pd.to_datetime(pd.read_csv(path)["datetime"]).dt.floor(bin).unique()
    seg_df.loc[seg_df.index.isin(fb_bins), "feedback"] = 1
    print(f"  feedback: {int(seg_df['feedback'].sum())} of {len(fb_bins)} feedback bins fall in the segment")
    return seg_df


# --- Split and save -----------------------------------------------------

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


def split_frames(df: pd.DataFrame, val_size: float, test_size: float) -> dict[str, pd.DataFrame]:
    """Chronological train -> val -> test split."""
    train_idx, val_idx = split_three_way(len(df), val_size=val_size, test_size=test_size)
    return {"train": df.iloc[:train_idx], "val": df.iloc[train_idx:val_idx], "test": df.iloc[val_idx:]}


def save_csv(df: pd.DataFrame, data_name: str, tag: str, arr_name: str, arr_type: str) -> None:
    if len(df) == 0:
        print(f"Warning: {arr_name}_{arr_type} is empty. Not saving.")
        return

    save_path = os.path.join(load_base_data_path(), f"{data_name}/{tag}/{arr_name}_{arr_type}.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)


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
        "offset": offset,
    }
