"""State analysis from the paths saved by `sample_paths`.

For each state s, over the sampled paths:

    co2         mean observed CO2 y_t over the steps the path spends in s
    drift       mean of y_t - y_{t-1} over the steps t >= 2 the path spends in s, i.e. how CO2 moves in s
    time share  share of all time steps the path spends in s

Each is computed per path; the CSV gives the mean over paths and the 2.5-97.5 % range across paths. The
drift quantiles are instead taken over all drifts pooled into one series: every step t >= 2 of every path that
is in s contributes its y_t - y_{t-1}. The thesis table (`Thesis/Tables/state_summary_<room>_<tag>_<model>_K<K>.tex`)
gives the mean CO2, the mean drift (the mean over paths of each path's mean drift), the 25 % and 75 % quantiles
of the pooled drifts and the time share.
The daily profiles plot P(z_t = s | y_1:T) (the share of paths in s at t) averaged over the days by
time of day, once for working days (working hours 08-17 shaded) and once for off days (weekends and
Danish holidays, the model's off_day covariate).

CSV in results/{room}/{tag}/states_{K}/simulation/, plots in plots/{room}/{tag}/states_{K}/simulation/.

    uv run python -m drivers.simulation.state_analysis ar1_hmm_beta_1 --room 15 -K 4
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from drivers.models.random_search import CHAIN
from drivers.models.utils import PLOTS_DIR, TAG, room_name
from drivers.simulation.sample_paths import paths_file
from drivers.utils import save_thesis_table

WORK_START, WORK_END = 8, 17
CHUNK = 5_000  # paths per chunk, so the (paths, T) boolean masks stay small
DRIFT_QUANTILES = (0.25, 0.75)


def per_path_stats(paths: np.ndarray, ys: np.ndarray, num_states: int) -> tuple[np.ndarray, ...]:
    """(P, K) mean CO2, drift and time share per path, and the (T, K) share of paths in each state at each step."""
    y, dy = ys.astype(np.float32), np.diff(ys).astype(np.float32)
    co2 = np.empty((len(paths), num_states))
    drift = np.empty((len(paths), num_states))
    share = np.empty((len(paths), num_states))
    occupancy = np.zeros((paths.shape[1], num_states))
    for start in range(0, len(paths), CHUNK):
        chunk = paths[start:start + CHUNK]
        for s in range(num_states):
            in_s = chunk == s
            steps = in_s[:, 1:].sum(axis=1)
            with np.errstate(invalid="ignore"):  # NaN if the path never visits s
                co2[start:start + len(chunk), s] = (in_s @ y) / in_s.sum(axis=1)
                drift[start:start + len(chunk), s] = (in_s[:, 1:] @ dy) / steps
            share[start:start + len(chunk), s] = in_s.mean(axis=1)
            occupancy[:, s] += in_s.sum(axis=0)
    return co2, drift, share, occupancy / len(paths)


def summary_table(stats: dict[str, np.ndarray]) -> pd.DataFrame:
    """Mean and 2.5-97.5 % range across paths of each (P, K) statistic; one row per state."""
    cols = {}
    for name, v in stats.items():
        lo, hi = np.nanpercentile(v, [2.5, 97.5], axis=0)
        cols |= {f"{name}_mean": np.nanmean(v, axis=0), f"{name}_lo_2.5": lo, f"{name}_hi_97.5": hi}
    k = next(iter(stats.values())).shape[1]
    return pd.DataFrame(cols, index=pd.Index(np.arange(1, k + 1), name="state"))


def pooled_drift_quantiles(ys: np.ndarray, occupancy: np.ndarray, n_paths: int) -> pd.DataFrame:
    """(K, Q) quantiles of all drifts y_t - y_{t-1} (t >= 2) pooled over the paths, one row per state.

    The pooled series for s holds y_t - y_{t-1} once for every path in s at t, so it is the drifts weighted by
    the path counts; the quantile is the smallest drift whose cumulative count reaches q of the total
    (np.quantile's "inverted_cdf" on the expanded series).
    """
    dy = np.diff(ys)
    order = np.argsort(dy)
    counts = np.rint(occupancy[1:] * n_paths)[order]  # (T-1, K) paths in s at each step, in drift order
    cum = np.cumsum(counts, axis=0)
    rows = [dy[order][np.searchsorted(cum[:, s], np.asarray(DRIFT_QUANTILES) * cum[-1, s])]
            for s in range(occupancy.shape[1])]
    return pd.DataFrame(rows, columns=[f"drift_q{round(100 * q)}" for q in DRIFT_QUANTILES],
                        index=pd.Index(np.arange(1, len(rows) + 1), name="state"))


def thesis_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Path means and pooled drift quantiles, for the thesis; drifts as math strings to get a proper minus."""
    drift = lambda col: [f"${d:.2f}$" for d in summary[col]]
    return pd.DataFrame({"State": summary.index.astype(str),
                         "Mean CO$_2$ (ppm)": summary["co2_mean"].to_numpy(),
                         "Mean drift (ppm / 30 min)": drift("drift_mean"),
                         **{f"Drift $Q_{{{q:g}}}$ (ppm / 30 min)": drift(f"drift_q{round(100 * q)}")
                            for q in DRIFT_QUANTILES},
                         "Time in state (%)": 100 * summary["time_share_mean"].to_numpy()})


def model_label(slug: str) -> str:
    """e.g. ar1_hmm_beta_1 -> AR(1)-HMM($\\beta$,1), for plot titles."""
    name = {s: n for s, n, *_ in CHAIN}.get(slug, slug)
    return name.replace("(b,", "($\\beta$,")


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

    co2, drift, share, occupancy = per_path_stats(paths, ys, args.states)
    summary = summary_table({"co2": co2, "drift": drift, "time_share": share}) \
        .join(pooled_drift_quantiles(ys, occupancy, len(paths)))
    print(summary.round(3).to_string())
    out = os.path.join(os.path.dirname(src), f"{args.model}_state_summary.csv")
    summary.to_csv(out)
    print(f"Saved {out}")

    room_title = room.replace("_", " ").title()  # room_015 -> Room 015
    name = f"state_summary_{room}_{args.tag}_{args.model}_K{args.states}"
    n_paths = f"{len(paths):,}".replace(",", "{,}")  # 100{,}000: a bare comma in math-free text gets odd spacing
    print("Saved", save_thesis_table(
        thesis_table(summary), name=name, label=f"tab:{name}",
        caption=f"{room_title}: mean observed CO$_2$, drift $y_t - y_{{t-1}}$ and share of time spent in each state "
                f"of the {model_label(args.model)} with $K={args.states}$ states, from {n_paths} state paths sampled "
                f"from $P(z_{{1:T}} \\mid y_{{1:T}})$ over the whole series. The mean CO$_2$, mean drift and time "
                f"share are the mean over paths of each path's value. The quantiles $Q_{{0.25}}$ and $Q_{{0.75}}$ "
                f"are taken over the drifts of all paths pooled into one series.",
        max_width=True))

    plot_dir = os.path.join(PLOTS_DIR, room, args.tag, f"states_{args.states}", "simulation")
    title = f"{room_title}, {model_label(args.model)}, $K={args.states}$"
    for name, days, label in (("working_days", ~off_day, "working days"), ("off_days", off_day, "off days")):
        path = os.path.join(plot_dir, f"{args.model}_daily_states_{name}.png")
        plot_daily(daily_profile(occupancy, dt, days), dt[days].normalize().nunique(),
                   f"{title}: mean state probability over the day, {label}", path, working_hours=name == "working_days")
        print(f"Saved {path}")
