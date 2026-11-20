"""Clean a DTU room's raw CO2 series with a causal fresh-air baseline, written to the `calibrated_anchor` tag.

A trailing 8-day low quantile floor jumps whenever a well-aired weekend enters or leaves the window,
and those jumps end up in the CO2 series as level steps. Here the baseline at time t only uses
readings up to t, and a calibrated value is never revised, so the series can be produced and trained
on online:

- Rule A (down fast): if the trailing SMOOTH mean drops below the baseline, the baseline is too high
  (the room can't be below fresh air), so it drops to that mean at once and any ramp is cancelled.
- Rule B (up slowly): a day whose minimum was reached with CO2 levelled off (|slope| < MAX_SLOPE over
  LOOKBACK) is an anchor. Once that day is over, the baseline ramps towards it over RAMP, moving at
  most MAX_STEP per anchor.

The baseline starts at OUTSIDE_CO2 (trust the sensor until there is evidence), and empty bins are
forward-filled rather than interpolated.

    uv run python -m drivers.data.clean_room_data_anchor "Room 015" ["Room 009" ...]
"""

import os
import sys

import numpy as np
import pandas as pd

from drivers.data.clean_room_data import (BIN, OUTSIDE_CO2, build_metadata, load_co2, room_slug, save_metadata,
                                          save_split, split_frames)
from drivers.utils import load_base_data_path

TAG = "calibrated_anchor"
SMOOTH = "2h"         # trailing mean, so one noisy reading can't move the baseline
LOOKBACK = "3h"       # slope window going into a day's minimum
MAX_SLOPE = 3.0       # ppm/h; with a decay time of ~10 h this is within ~30 ppm of fresh air
MAX_STEP = 30.0       # ppm an anchor may raise or lower the baseline
RAMP = "7D"           # time to move from the current baseline to an anchor's target


# --- Anchors ------------------------------------------------------------

def daily_minima(smooth: pd.Series) -> pd.DataFrame:
    """Per day: the lowest smoothed reading and the slope going into it. A day's anchor is only
    known once the day is over (`known_at`), so it is applied from then on."""
    hours = pd.Timedelta(LOOKBACK) / pd.Timedelta("1h")
    rows = []
    for day, s in smooth.groupby(smooth.index.normalize()):
        t = s.idxmin()
        before = smooth[t - pd.Timedelta(LOOKBACK):t]
        slope = (before.iloc[-1] - before.iloc[0]) / hours if len(before) > 1 else np.nan
        rows.append({"day": day, "datetime": t, "minimum": s.min(), "slope": slope})
    days = pd.DataFrame(rows)
    days["valid"] = days["slope"].abs() < MAX_SLOPE
    days["known_at"] = days["day"] + pd.Timedelta("1D")
    return days


# --- Baseline -----------------------------------------------------------

def causal_baseline(co2: pd.Series) -> tuple[pd.Series, pd.DataFrame, float]:
    """Baseline at every reading from readings up to that time only.
    Returns the baseline, the daily minima with the target each valid anchor set, and the share of
    readings where rule A set the baseline."""
    smooth = co2.rolling(SMOOTH).mean()
    days = daily_minima(smooth)
    anchors = list(days[days["valid"]].itertuples())
    ramp_len = pd.Timedelta(RAMP)

    base, ramp, i, rule_a = OUTSIDE_CO2, None, 0, 0
    targets, baseline = {}, np.empty(len(smooth))
    for n, (t, s) in enumerate(smooth.items()):
        while i < len(anchors) and anchors[i].known_at <= t:  # rule B: an anchor's day just ended
            target = base + np.clip(anchors[i].minimum - base, -MAX_STEP, MAX_STEP)
            ramp, targets[anchors[i].Index] = (t, base, target), target
            i += 1
        if ramp is not None:
            t0, start, target = ramp
            base = start + (target - start) * min((t - t0) / ramp_len, 1.0)
        if s < base:                                          # rule A: never above what the room reads
            base, ramp, rule_a = s, None, rule_a + 1
        baseline[n] = base

    days["target"] = pd.Series(targets)
    return pd.Series(baseline, index=co2.index, name="floor"), days, rule_a / len(smooth)


def bin_causal(series: pd.Series) -> pd.Series:
    """Resample to `BIN` means and forward-fill empty bins (interpolating would use the next reading)."""
    return series.resample(BIN).mean().ffill()


# --- Driver -------------------------------------------------------------

def save_anchors(days: pd.DataFrame, data_name: str, tag: str) -> None:
    path = os.path.join(load_base_data_path(), data_name, tag, "anchors.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    days.to_csv(path, index=False)


def clean_room_data_anchor(room_name: str, tag: str = TAG, val_size: float = 0.15, test_size: float = 0.20) -> None:
    """Clean one room and write y/floor train/val/test CSVs, anchors.csv and metadata to `{DATA_PATH}/<room>/<tag>/`.
    `co2_raw = co2 + floor - offset`, as for `clean_room_data`, with offset = OUTSIDE_CO2."""
    co2 = load_co2(room_name)
    floor, days, rule_a_share = causal_baseline(co2)
    co2_cal = co2 - floor + OUTSIDE_CO2

    y = bin_causal(co2_cal).rename("co2").reset_index()
    floor_binned = bin_causal(floor).rename("floor").reset_index()

    data_name = room_slug(room_name)
    y_split = split_frames(y, val_size, test_size)
    save_split(y_split, data_name, tag, "y")
    save_split(split_frames(floor_binned, val_size, test_size), data_name, tag, "floor")
    save_anchors(days, data_name, tag)
    metadata = build_metadata(room_name, tag, y_split, OUTSIDE_CO2)
    metadata.update(method="anchor_causal", smooth=SMOOTH, lookback=LOOKBACK,
                    max_slope=MAX_SLOPE, max_step=MAX_STEP, ramp=RAMP, n_anchors=int(days["valid"].sum()),
                    rule_a_share=rule_a_share)
    save_metadata(metadata, data_name, tag)
    print(f"{room_name}: {len(y)} bins, split {[len(p) for p in y_split.values()]}, "
          f"{int(days['valid'].sum())} anchors, rule A active {rule_a_share:.1%}, "
          f"floor {floor.min():.0f}-{floor.max():.0f} ppm")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit('usage: python -m drivers.data.clean_room_data_anchor "Room 015" ["Room 009" ...]')
    for room_name in sys.argv[1:]:
        clean_room_data_anchor(room_name)
