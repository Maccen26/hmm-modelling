"""Data plots of one DTU room, ported from `summary_statistics.ipynb`, saved to `plots/<room>/summary/`.

Figures: missing bins and their linear interpolation, the raw readings against the causal anchor
baseline, and, on the anchor-calibrated series (`calibrated_anchor` tag, written by
`clean_room_data_anchor`): log CO2, ACF / PACF, diurnal and weekly profiles (all days, working vs
off days, working hours), the series with off days marked, and the train / val / test split.

    uv run python -m drivers.data.create_data_plots --room 15 [--tag calibrated_anchor]
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from drivers.data.clean_room_data import clean_df, load_co2, room_slug
from drivers.data.create_summary_tables import STEP, gap_table, load_raw, raw_dir
from drivers.models.utils import PLOTS_DIR, TAG
from drivers.utils import load_base_data_path, load_y_df

ACF_LAGS = 48 * 7  # seven days of 30 min bins
PACF_LAGS = 5
WORK_START, WORK_END = 8, 17
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SPLITS = [("train", "Train", "tab:blue"), ("val", "Validation", "tab:orange"), ("test", "Test", "tab:green")]


# --- Load ---------------------------------------------------------------

def load_split_series(data_name: str, tag: str, arr_name: str) -> pd.Series:
    """The saved train, val and test `{arr_name}` CSVs joined into one series indexed by datetime."""
    parts = []
    for split, _, _ in SPLITS:
        df = pd.read_csv(os.path.join(load_base_data_path(), data_name, tag, f"{arr_name}_{split}.csv"),
                         parse_dates=["datetime"])
        parts.append(df.set_index("datetime").iloc[:, 0])
    return pd.concat(parts)


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


def plot_anchor_baseline(co2, floor, room_name):
    """Raw readings with the causal anchor baseline that `clean_room_data_anchor` subtracts."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(co2, lw=0.6, alpha=0.7, label="Raw CO$_2$")
    ax.plot(floor, lw=1.5, color="tab:red", label="Anchor baseline")
    ax.set(title=room_name, xlabel="Date", ylabel="CO$_2$ [ppm]", xlim=(floor.index[0], floor.index[-1]))
    ax.legend()
    return fig


def plot_log(co2_bin, room_name):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.log(co2_bin), lw=0.8, label="30 min log mean CO$_2$")
    ax.set(title=room_name, xlabel="Date", ylabel="log CO$_2$ [ppm]")
    ax.legend()
    return fig


def plot_acf_pacf(co2_bin, room_name):
    """ACF over seven days (dotted = whole days) above the PACF of the first few lags."""
    fig, (ax_acf, ax_pacf) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [3, 2]})
    style = {"color": "navy", "vlines_kwargs": {"colors": "navy", "lw": 1}}
    plot_acf(co2_bin, lags=ACF_LAGS, ax=ax_acf, marker=None, **style)
    plot_pacf(co2_bin, lags=PACF_LAGS, ax=ax_pacf, method="ywm", markersize=4, **style)
    bound = 1.96 / np.sqrt(len(co2_bin))  # 95% band under white noise, as shaded by plot_pacf
    ax_pacf.fill_between([0.5, PACF_LAGS + 0.5], -bound, bound, color="navy", alpha=0.15, lw=0)
    for sign in (-1, 1):
        ax_pacf.axhline(sign * bound, color="navy", ls="--", lw=0.8)
    for lag in range(48, ACF_LAGS + 1, 48):
        ax_acf.axvline(lag, color="grey", ls=":", lw=1)
    ax_acf.set(title="ACF", xlabel="Lag (30 min bins)", ylabel="Correlation", xticks=range(0, ACF_LAGS + 1, 48))
    ax_pacf.set(title="PACF", xlabel="Lag (30 min bins)", ylabel="Partial correlation", xticks=range(PACF_LAGS + 1))
    for ax in (ax_acf, ax_pacf):
        ax.set_ylim(-0.5, 1.05)
        ax.axhline(0, color="black", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{room_name}: autocorrelation of calibrated CO$_2$")
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
    """Working days and off days on one axis, grouped by time of day at 30 min resolution."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axvspan(WORK_START, WORK_END, color="lightgrey", alpha=0.6, lw=0,
               label=f"Working hours ({WORK_START}-{WORK_END})")
    time_of_day = co2_bin.index.hour + co2_bin.index.minute / 60
    for label, mask, color in [("Working days", ~off_day, "navy"), ("Off days", off_day, "darkorange")]:
        iqr_band(ax, co2_bin[mask].groupby(time_of_day[mask]), color, label)
    ax.set(title="Diurnal Variation and IQR of CO$_2$ Levels: Working vs Off Days",
           xlabel="Hour of Day (0 - 24)", ylabel="CO$_2$ [ppm]", xticks=range(0, 25, 2), xlim=(0, 24))
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


def create_data_plots(room_name: str, tag: str = TAG) -> list[str]:
    raw_clean = clean_df(load_raw(room_name))
    binned = raw_clean.set_index("datetime")["co2"].resample(STEP).mean()
    gaps, _ = gap_table(raw_clean, top=None)

    data_name = room_slug(room_name)
    co2_bin = load_split_series(data_name, tag, "y")
    floor = load_split_series(data_name, tag, "floor")

    holidays = all_holidays()
    off_day = off_day_mask(co2_bin.index, holidays)
    print(f"  off days: {co2_bin.index.normalize()[off_day].nunique()} of {co2_bin.index.normalize().nunique()} days")
    normal = normal_weeks(co2_bin, holidays)

    save_dir = os.path.join(PLOTS_DIR, data_name, "summary")
    os.makedirs(save_dir, exist_ok=True)
    figures = {
        "raw_gaps": plot_gaps(binned, gaps, room_name),
        "raw_interpolation": plot_interpolation(binned, room_name),
        "anchor_baseline": plot_anchor_baseline(load_co2(room_name), floor, room_name),
        "calibrated_log": plot_log(co2_bin, room_name),
        "calibrated_acf_pacf": plot_acf_pacf(co2_bin, room_name),
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
    parser.add_argument("--tag", default=TAG, help=f"data tag of the saved train/val/test split (default {TAG})")
    args = parser.parse_args()

    for path in create_data_plots(f"Room {args.room:03d}", tag=args.tag):
        print(f"wrote {path}")
