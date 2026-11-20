"""State analysis from the paths saved by `sample_paths`.

For each state s, over the sampled paths:

    drift       mean of y_t - y_{t-1} over the steps t >= 2 the path spends in s, i.e. how CO2 moves in s
    time share  share of all time steps the path spends in s

Each is computed per path; the table gives the mean over paths and the 2.5-97.5 % range across paths.
The daily profiles plot P(z_t = s | y_1:T) (the share of paths in s at t) averaged over the days by
time of day, once for working days (working hours 08-17 shaded) and once for off days (weekends and
Danish holidays, the model's off_day covariate).

Table in results/{room}/{tag}/states_{K}/simulation/, plots in plots/{room}/{tag}/states_{K}/simulation/.

    uv run python -m drivers.simulation.state_analysis ar1_hmm_beta_1 --room 15 -K 4
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from drivers.models.utils import PLOTS_DIR, TAG, room_name
from drivers.simulation.sample_paths import paths_file

WORK_START, WORK_END = 8, 17
CHUNK = 5_000  # paths per chunk, so the (paths, T) boolean masks stay small


def per_path_stats(paths: np.ndarray, ys: np.ndarray, num_states: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(P, K) drift and time share per path, and the (T, K) share of paths in each state at each step."""
    dy = np.diff(ys).astype(np.float32)
    drift = np.empty((len(paths), num_states))
    share = np.empty((len(paths), num_states))
    occupancy = np.zeros((paths.shape[1], num_states))
    for start in range(0, len(paths), CHUNK):
        chunk = paths[start:start + CHUNK]
        for s in range(num_states):
            in_s = chunk == s
            steps = in_s[:, 1:].sum(axis=1)
            with np.errstate(invalid="ignore"):
                drift[start:start + len(chunk), s] = (in_s[:, 1:] @ dy) / steps  # NaN if the path never visits s
            share[start:start + len(chunk), s] = in_s.mean(axis=1)
            occupancy[:, s] += in_s.sum(axis=0)
    return drift, share, occupancy / len(paths)


def summary_table(drift: np.ndarray, share: np.ndarray) -> pd.DataFrame:
    d_lo, d_hi = np.nanpercentile(drift, [2.5, 97.5], axis=0)
    s_lo, s_hi = np.percentile(share, [2.5, 97.5], axis=0)
    return pd.DataFrame({"drift_mean": np.nanmean(drift, axis=0), "drift_lo_2.5": d_lo, "drift_hi_97.5": d_hi,
                         "time_share_mean": share.mean(axis=0), "time_share_lo_2.5": s_lo,
                         "time_share_hi_97.5": s_hi},
                        index=pd.Index(np.arange(1, drift.shape[1] + 1), name="state"))


def daily_profile(occupancy: np.ndarray, dt: pd.DatetimeIndex, days: np.ndarray) -> pd.DataFrame:
    """Mean P(z_t = s) by time of day over the selected steps; index is hours since midnight."""
    hour = dt.hour + dt.minute / 60
    return pd.DataFrame(occupancy[days]).groupby(np.asarray(hour[days])).mean()


def plot_daily(profile: pd.DataFrame, n_days: int, title: str, path: str, working_hours: bool):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    if working_hours:
        ax.axvspan(WORK_START, WORK_END, color="lightgrey", alpha=0.5, lw=0, label="Working hours")
    for s in profile.columns:
        ax.plot(profile.index, profile[s], lw=1.8, label=f"State {s + 1}")
    ax.set(title=f"{title} ({n_days} days)", xlabel="Time of day [h]", ylabel="P(state | y$_{1:T}$)",
           xlim=(0, 24), ylim=(0, 1), xticks=range(0, 25, 2))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", nargs="?", default="ar1_hmm_beta_1", help="slug of the saved fit")
    parser.add_argument("--room", "-r", type=int, required=True, help="room number, e.g. 15 for room_015")
    parser.add_argument("--states", "-K", type=int, required=True)
    parser.add_argument("--tag", default=TAG, help=f"data tag (default {TAG})")
    args = parser.parse_args()

    room = room_name(args.room)
    src = paths_file(room, args.states, args.model, args.tag)
    saved = np.load(src)
    paths, ys, off_day = saved["paths"], saved["ys"], saved["off_day"].astype(bool)
    dt = pd.DatetimeIndex(saved["datetime"])
    print(f"{args.model} — {room}, K={args.states}: {len(paths)} paths over {paths.shape[1]} bins from {src}")

    drift, share, occupancy = per_path_stats(paths, ys, args.states)
    table = summary_table(drift, share)
    print(table.round(3).to_string())
    out = os.path.dirname(src)
    table.to_csv(os.path.join(out, f"{args.model}_state_summary.csv"))

    plot_dir = os.path.join(PLOTS_DIR, room, args.tag, f"states_{args.states}", "simulation")
    title = f"{args.model} — {room}, K={args.states}"
    for name, days, label in (("working_days", ~off_day, "working days"), ("off_days", off_day, "off days")):
        path = os.path.join(plot_dir, f"{args.model}_daily_states_{name}.png")
        plot_daily(daily_profile(occupancy, dt, days), dt[days].normalize().nunique(),
                   f"{title}: mean state probability over the day, {label}", path, working_hours=name == "working_days")
        print(f"Saved {path}")
    print(f"Saved {os.path.join(out, f'{args.model}_state_summary.csv')}")
