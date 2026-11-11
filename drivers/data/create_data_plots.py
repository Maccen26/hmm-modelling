"""Data plots of one DTU room, ported from `summary_statistics.ipynb`, saved to `plots/<room>/summary/`.

Figures: missing bins and their linear interpolation, the trailing floor vs low quantile and the
calibration they give, log CO2, ACF / PACF, diurnal and weekly profiles (all days, working vs
off days, working hours), the series with off days marked, and the train / val / test split.

    uv run python -m drivers.data.create_data_plots --room 15
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from drivers.data.clean_dtu_data import room_slug
from drivers.data.clean_room_data import (
    OUTSIDE_CO2, QUANTILE, WINDOW, bin_and_fill, calibrate, clean_df, load_co2,
)
from drivers.data.create_summary_tables import STEP, gap_table, load_raw, raw_dir
from drivers.models.utils import PLOTS_DIR
from drivers.utils import load_y_df

ACF_LAGS = 48 * 3  # three days of 30 min bins
PACF_LAGS = 5
WORK_START, WORK_END = 8, 17
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SPLITS = [("train", "Train", "tab:blue"), ("val", "Validation", "tab:orange"), ("test", "Test", "tab:green")]


# --- Load ---------------------------------------------------------------

def load_holidays(year: int) -> pd.DatetimeIndex:
    df = pd.read_excel(raw_dir() / f"holidays_{year}.xlsx", skiprows=1)
    df.columns = df.columns.str.strip()
    return pd.DatetimeIndex(df["DATE"]).normalize()


def all_holidays() -> pd.DatetimeIndex:
    """DTU holidays, with the whole Christmas break (24/12 - 01/01) as off days."""
    holidays = load_holidays(2023).union(load_holidays(2024))
    return holidays.union(pd.date_range("2023-12-24", "2024-01-01", freq="D"))


def off_day_mask(index: pd.DatetimeIndex, holidays: pd.DatetimeIndex) -> np.ndarray:
    return (index.dayofweek >= 5) | index.normalize().isin(holidays)  # Saturday = 5, Sunday = 6


def normal_weeks(co2_bin: pd.Series, holidays: pd.DatetimeIndex) -> pd.Series:
    """Complete Mon-Sun weeks without holidays, so the weekend is the only off time."""
    week = co2_bin.index.to_period("W-SUN")
    complete = pd.Series(week).map(pd.Series(week).value_counts()).to_numpy() == 7 * 48
    keep = ~week.isin(holidays.to_period("W-SUN")) & complete
    print(f"  normal weeks: {week[keep].nunique()} of {week.nunique()} weeks")
    return co2_bin[keep]


# --- Plots --------------------------------------------------------------

def plot_gaps(binned, gaps, room_name):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(binned, lw=0.8, label="30 min mean CO$_2$")
    for i, (_, g) in enumerate(gaps.iterrows()):
        ax.axvspan(g["Start"], g["End"] + STEP, color="red", alpha=0.3, label="Missing" if i == 0 else None)
    ax.scatter(gaps["Start"] - STEP, gaps["CO2 before"], s=15, color="green", zorder=3, label="Before gap")
    ax.scatter(gaps["End"] + STEP, gaps["CO2 after"], s=15, color="black", zorder=3, label="After gap")
    ax.set(title=room_name, xlabel="Date", ylabel="CO$_2$ [ppm]")
    ax.legend()
    return fig


def plot_interpolation(binned, room_name):
    interpolated = binned.interpolate(method="linear")
    filled = binned.isna()
    near_gap = filled | filled.shift(1, fill_value=False) | filled.shift(-1, fill_value=False)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(binned, lw=0.8, label="30 min mean CO$_2$")
    ax.plot(interpolated.where(near_gap), lw=1.2, color="orange", label="Linear interpolation")
    ax.set(title=room_name, xlabel="Date", ylabel="CO$_2$ [ppm]")
    ax.legend()
    return fig


def plot_floor_vs_quantile(floor, low_q, room_name):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(floor, label=f"Rolling {WINDOW} floor (min)", lw=1)
    ax.plot(low_q, label=f"Rolling {WINDOW} {QUANTILE:.1%} quantile", lw=1)
    ax.set(title=room_name, xlabel="Date", ylabel="CO$_2$ [ppm]")
    ax.legend()
    return fig


def plot_calibrated_floors(co2, floor, low_q, room_name):
    """Trailing minimum of the series calibrated with each floor."""
    co2_floor = co2 - floor + OUTSIDE_CO2
    co2_quantile, _ = calibrate(co2, low_q)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(co2_floor.rolling(WINDOW).min(), label=f"Rolling {WINDOW} co2 floor (min)", lw=1)
    ax.plot(co2_quantile.rolling(WINDOW).min(), label=f"Rolling {WINDOW} co2 quantile (min)", lw=1)
    ax.set(title=room_name, xlabel="Date", ylabel="CO$_2$ [ppm]")
    ax.legend()
    return fig


def plot_log(co2_bin, room_name):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.log(co2_bin), lw=0.8, label="30 min log mean CO$_2$")
    ax.set(title=room_name, xlabel="Date", ylabel="log CO$_2$ [ppm]")
    ax.legend()
    return fig


def plot_acf_bins(co2_bin, room_name):
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_acf(co2_bin, lags=ACF_LAGS, ax=ax, title=f"{room_name}: ACF of calibrated CO$_2$")
    return fig


def plot_pacf_bins(co2_bin, room_name):
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_pacf(co2_bin, lags=PACF_LAGS, ax=ax, title=f"{room_name}: PACF of calibrated CO$_2$")
    return fig


def iqr_band(ax, grouped, color, label, lo=0.25, hi=0.75, fill=None):
    """Median line in `color` over the `lo`-`hi` quantile band in `fill` (defaults to `color`)."""
    median = grouped.median()
    ax.fill_between(median.index, grouped.quantile(lo), grouped.quantile(hi), color=fill or color, alpha=0.3,
                    label=f"{label} IQR ({lo:.0%} - {hi:.0%})")
    ax.plot(median.index, median, color=color, lw=2, label=f"{label} median")


def plot_diurnal(co2_bin):
    fig, ax = plt.subplots(figsize=(10, 5))
    iqr_band(ax, co2_bin.groupby(co2_bin.index.hour), "navy", "CO$_2$", fill="skyblue")
    ax.set(title="Diurnal Variation and IQR of CO$_2$ Levels", xlabel="Hour of Day (0 - 23)",
           ylabel="CO$_2$ [ppm]", xticks=range(0, 24))
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    return fig


def plot_diurnal_off_days(co2_bin, off_day):
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, mask, color in [("Working days", ~off_day, "navy"), ("Off days", off_day, "darkorange")]:
        iqr_band(ax, co2_bin[mask].groupby(co2_bin.index.hour[mask]), color, label)
    ax.set(title="Diurnal Variation and IQR of CO$_2$ Levels: Working vs Off Days",
           xlabel="Hour of Day (0 - 23)", ylabel="CO$_2$ [ppm]", xticks=range(0, 24))
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    return fig


def weekly_axes(ax, title):
    """Day boundaries as gridlines, day names centred at noon."""
    ax.set(title=title, xlabel="Day of week", ylabel="CO$_2$ [ppm]", xlim=(0, 7 * 24))
    ax.set_xticks(range(0, 7 * 24 + 1, 24), labels=[])
    ax.set_xticks(range(12, 7 * 24, 24), labels=DAY_NAMES, minor=True)
    ax.tick_params(axis="x", which="minor", length=0)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right")


def by_hour_of_week(normal):
    """Group by position in the week in hours (Monday 00:00 = 0), at 30 min resolution."""
    idx = normal.index
    return normal.groupby(idx.dayofweek * 24 + idx.hour + idx.minute / 60)


def plot_weekly(normal):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axvspan(5 * 24, 7 * 24, color="grey", alpha=0.15, label="Weekend")
    iqr_band(ax, by_hour_of_week(normal), "navy", "CO$_2$", lo=0.15, hi=0.85, fill="skyblue")
    weekly_axes(ax, "Weekly Variation of CO$_2$ Levels (weeks without holidays)")
    return fig


def plot_weekly_working_hours(normal):
    fig, ax = plt.subplots(figsize=(14, 5))
    for day in range(5):  # Monday to Friday
        ax.axvspan(day * 24 + WORK_START, day * 24 + WORK_END, color="lightgrey", alpha=0.6, lw=0,
                   label=f"Working hours ({WORK_START}-{WORK_END})" if day == 0 else None)
    iqr_band(ax, by_hour_of_week(normal), "navy", "CO$_2$", fill="skyblue")
    weekly_axes(ax, "Weekly Variation and IQR of CO$_2$ Levels with Working Hours (weeks without holidays)")
    return fig


def plot_off_days(co2_bin, off_day, room_name):
    fig, ax = plt.subplots(figsize=(14, 4))
    for i, d in enumerate(co2_bin.index.normalize()[off_day].unique()):
        ax.axvspan(d, d + pd.Timedelta("1D"), color="lightgrey", alpha=0.6, lw=0,
                   label="Off day" if i == 0 else None)
    ax.plot(co2_bin, color="navy", lw=0.8, label="30 min mean CO$_2$")
    ax.set(title=f"{room_name}: CO$_2$ with off days marked", xlabel="Date", ylabel="CO$_2$ [ppm]",
           xlim=(co2_bin.index[0], co2_bin.index[-1]))
    ax.legend(loc="upper right")
    return fig


def plot_split(data_name, tag, room_name):
    split = {key: load_y_df(data_name, tag, key).set_index("datetime")["co2"] for key, _, _ in SPLITS}
    n_total = sum(len(s) for s in split.values())
    fig, ax = plt.subplots(figsize=(14, 4))
    for key, label, color in SPLITS:
        s = split[key]
        share = len(s) / n_total
        ax.axvspan(s.index[0], s.index[-1] + STEP, color=color, alpha=0.12, lw=0)
        ax.plot(s, color=color, lw=0.8, label=f"{label}: {len(s)} bins ({share:.0%})")
        ax.text(s.index[0] + (s.index[-1] - s.index[0]) / 2, 0.95, f"{label}\n{share:.0%}",
                transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=11,
                fontweight="bold", color=color)
    ax.set(title=f"{room_name}: train / validation / test split (calibrated CO$_2$)",
           xlabel="Date", ylabel="CO$_2$ [ppm]", xlim=(split["train"].index[0], split["test"].index[-1] + STEP))
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.15), ncol=3, frameon=False)
    return fig


# --- Driver -------------------------------------------------------------

def save(fig, save_dir: str, name: str) -> str:
    fig.tight_layout()
    path = os.path.join(save_dir, f"{name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def create_data_plots(room_name: str, tag: str = "calibrated") -> list[str]:
    raw_clean = clean_df(load_raw(room_name))
    binned = raw_clean.set_index("datetime")["co2"].resample(STEP).mean()
    gaps, _ = gap_table(raw_clean, top=None)

    co2 = load_co2(room_name)
    rolling = co2.rolling(WINDOW)
    floor, low_q = rolling.min(), rolling.quantile(QUANTILE)
    co2_bin = bin_and_fill(calibrate(co2, low_q)[0])

    holidays = all_holidays()
    off_day = off_day_mask(co2_bin.index, holidays)
    print(f"  off days: {co2_bin.index.normalize()[off_day].nunique()} of {co2_bin.index.normalize().nunique()} days")
    normal = normal_weeks(co2_bin, holidays)

    data_name = room_slug(room_name)
    save_dir = os.path.join(PLOTS_DIR, data_name, "summary")
    os.makedirs(save_dir, exist_ok=True)
    figures = {
        "raw_gaps": plot_gaps(binned, gaps, room_name),
        "raw_interpolation": plot_interpolation(binned, room_name),
        "floor_vs_quantile": plot_floor_vs_quantile(floor, low_q, room_name),
        "calibrated_floor_vs_quantile": plot_calibrated_floors(co2, floor, low_q, room_name),
        "calibrated_log": plot_log(co2_bin, room_name),
        "calibrated_acf": plot_acf_bins(co2_bin, room_name),
        "calibrated_pacf": plot_pacf_bins(co2_bin, room_name),
        "diurnal": plot_diurnal(co2_bin),
        "diurnal_off_days": plot_diurnal_off_days(co2_bin, off_day),
        "weekly": plot_weekly(normal),
        "weekly_working_hours": plot_weekly_working_hours(normal),
        "off_days": plot_off_days(co2_bin, off_day, room_name),
        "split": plot_split(data_name, tag, room_name),
    }
    return [save(fig, save_dir, f"{data_name}_{name}") for name, fig in figures.items()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--room", type=int, default=15, help="room number, e.g. 15 -> 'Room 015'")
    parser.add_argument("--tag", default="calibrated", help="data tag of the saved train/val/test split")
    args = parser.parse_args()

    for path in create_data_plots(f"Room {args.room:03d}", tag=args.tag):
        print(f"wrote {path}")
