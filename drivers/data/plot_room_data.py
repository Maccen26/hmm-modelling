"""Plot one room's raw and calibrated CO2 data, each figure saved separately to `plots/<room>/data/`.

Raw readings come from `{DATA_PATH}/raw/dtu/timeseries/<room>.csv` (after the NaN / range /
duplicate filter in `clean_dtu_data.load_raw_series`); the calibrated series is the saved
`y_{train,val,test}.csv` written by `clean_dtu_data`.

Figures: raw series, calibrated series, histogram and ACF of the calibrated series, and the
raw series' trailing 2.5% quantile against its trailing minimum (the calibration floor's window).

    uv run python -m drivers.data.plot_room_data --room 9
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf

from drivers.data.clean_dtu_data import load_raw_series, room_slug
from drivers.models.utils import PLOTS_DIR
from drivers.utils import load_base_data_path, load_y_df


def load_calibrated(data_name: str, tag: str) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    """Concatenated train/val/test y frame plus the two split boundaries."""
    parts = [load_y_df(data_name, tag, arr_type) for arr_type in ("train", "val", "test")]
    boundaries = [parts[1]["datetime"].iloc[0], parts[2]["datetime"].iloc[0]]
    return pd.concat(parts, ignore_index=True), boundaries


def load_metadata(data_name: str, tag: str) -> pd.Series:
    return pd.read_csv(os.path.join(load_base_data_path(), data_name, tag, "metadata.csv")).iloc[0]


def save(fig, save_dir: str, name: str) -> str:
    fig.tight_layout()
    path = os.path.join(save_dir, f"{name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_raw(df_raw, df_cal, room_name):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df_raw["datetime"], df_raw["co2"], lw=0.5, color="tab:gray")
    ax.axvspan(df_cal["datetime"].iloc[0], df_cal["datetime"].iloc[-1], color="tab:blue", alpha=0.08,
               label="calibrated segment")
    ax.set(title=f"{room_name}: raw CO2 readings", ylabel="CO2 (ppm)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    return fig


def plot_calibrated(df_cal, boundaries, baseline, room_name, tag):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df_cal["datetime"], df_cal["co2"], lw=0.7, color="tab:blue")
    ax.axhline(baseline, color="tab:red", ls="--", lw=1, label=f"baseline {baseline:.0f} ppm")
    for b, name in zip(boundaries, ("val", "test")):
        ax.axvline(b, color="k", ls=":", lw=1)
        ax.text(b, ax.get_ylim()[1], f" {name}", va="top", fontsize=9)
    ax.set(title=f"{room_name}: calibrated CO2 ({tag} bins)", ylabel="CO2 (ppm)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    return fig


def plot_histogram(df_cal, baseline, room_name, bins=80):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df_cal["co2"], bins=bins, color="tab:blue", edgecolor="white", density=True)
    ax.axvline(baseline, color="tab:red", ls="--", lw=1, label=f"baseline {baseline:.0f} ppm")
    ax.set(title=f"{room_name}: histogram of calibrated CO2", xlabel="CO2 (ppm)", ylabel="density")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    return fig


def plot_acf_calibrated(df_cal, room_name, lags=48 * 7):
    """ACF over one week of 30min lags, so the daily cycle (lag 48) is visible."""
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_acf(df_cal["co2"], lags=lags, ax=ax, marker=None, vlines_kwargs={"lw": 0.8})
    for day in range(48, lags + 1, 48):
        ax.axvline(day, color="tab:gray", ls=":", lw=0.8)
    ax.set(title=f"{room_name}: ACF of calibrated CO2 (dotted = whole days)", xlabel="lag (30min bins)")
    ax.grid(alpha=0.3)
    return fig


def plot_quantile_vs_min(df_raw, room_name, window="8D", quantile=0.025):
    """Trailing `window` quantile and minimum of the raw readings: how far a single-reading min
    would sit below the quantile-based sensor floor."""
    co2 = df_raw.set_index("datetime")["co2"]
    rolling = co2.rolling(window, min_periods=10)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(co2.index, co2, lw=0.4, color="tab:gray", alpha=0.5, label="raw CO2")
    ax.plot(co2.index, rolling.quantile(quantile), color="tab:blue", lw=1.2,
            label=f"trailing {window} {quantile:.1%} quantile")
    ax.plot(co2.index, rolling.min(), color="tab:red", lw=1.2, label=f"trailing {window} minimum")
    ax.set(title=f"{room_name}: raw {quantile:.1%} quantile vs minimum", ylabel="CO2 (ppm)",
           ylim=(co2.min() - 50, co2.quantile(0.5)))
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    return fig


def plot_room_data(room_name: str, tag: str = "30min") -> list[str]:
    data_name = room_slug(room_name)
    df_raw, _ = load_raw_series(room_name, y_cols=("co2",))
    df_cal, boundaries = load_calibrated(data_name, tag)
    meta = load_metadata(data_name, tag)
    baseline = meta["baseline_ppm"]

    save_dir = os.path.join(PLOTS_DIR, data_name, "data")
    os.makedirs(save_dir, exist_ok=True)
    prefix = f"{data_name}_{tag}"
    return [
        save(plot_raw(df_raw, df_cal, room_name), save_dir, f"{prefix}_raw"),
        save(plot_calibrated(df_cal, boundaries, baseline, room_name, tag), save_dir, f"{prefix}_calibrated"),
        save(plot_histogram(df_cal, baseline, room_name), save_dir, f"{prefix}_calibrated_hist"),
        save(plot_acf_calibrated(df_cal, room_name), save_dir, f"{prefix}_calibrated_acf"),
        save(plot_quantile_vs_min(df_raw, room_name, window=meta["baseline_window"]), save_dir,
             f"{prefix}_raw_quantile_vs_min"),
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--room", type=int, default=9, help="room number, e.g. 9 -> 'Room 009'")
    parser.add_argument("--tag", default="30min")
    args = parser.parse_args()

    for path in plot_room_data(f"Room {args.room:03d}", tag=args.tag):
        print(f"wrote {path}")
