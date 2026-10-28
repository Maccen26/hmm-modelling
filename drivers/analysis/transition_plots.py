"""Transition probabilities out of one state over a normal week (Mon-Fri working days + weekend).

The week is a 30-min grid starting on a Monday; its covariates are built exactly as for fitting, so the
curves are Gamma_t[from_state, :] as the fitted dynamic transition sees it. By default the week is the
first one in the training span without a holiday on Mon-Fri. The stay probability is drawn on its own
panel since it is usually close to 1 and would flatten the leaving probabilities.

    uv run python -m drivers.analysis.transition_plots ar1_hmm_beta_1 --room 9 -K 3 --state 1
    uv run python -m drivers.analysis.transition_plots ar1_hmm_beta_1 --room 9 -K 4 --state 2 --week 2023-11-27
"""
import jax
jax.config.update("jax_enable_x64", True)

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from drivers.analysis.state_confusion import load_saved
from drivers.models.utils import PLOTS_DIR, covariates

STEPS_PER_WEEK = 7 * 48  # 30-min bins


def week_grid(start: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start.normalize(), periods=STEPS_PER_WEEK, freq="30min")


def is_normal_week(dt: pd.DatetimeIndex) -> bool:
    """Mon-Fri are all working days and Sat-Sun are off days."""
    off_day = np.asarray(covariates(dt, ("off_day",)))[:, 0]
    return bool(np.array_equal(off_day, (dt.dayofweek >= 5).astype(float)))


def first_normal_week(span: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """First Monday-started week inside the span without a holiday."""
    monday = span[0].normalize() + pd.Timedelta(days=(7 - span[0].dayofweek) % 7)
    while monday + pd.Timedelta(days=7) <= span[-1]:
        dt = week_grid(monday)
        if is_normal_week(dt):
            return dt
        monday += pd.Timedelta(days=7)
    raise SystemExit("No holiday-free week in the training span; pass --week")


def plot(title: str, dt: pd.DatetimeIndex, probs: np.ndarray, from_state: int, path: str):
    """probs: (T, K) row `from_state` of Gamma_t over the week."""
    fig, (ax_stay, ax_leave) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, height_ratios=[1, 2])
    ax_stay.plot(dt, probs[:, from_state], color="black", lw=1.4)
    ax_stay.set_ylabel(f"P(s{from_state} → s{from_state})")
    for j in range(probs.shape[1]):
        if j != from_state:
            ax_leave.plot(dt, probs[:, j], lw=1.4, label=f"s{from_state} → s{j}")
    ax_leave.set_ylabel("Transition probability")
    ax_leave.legend(loc="upper right")

    weekend = dt.dayofweek >= 5
    for ax in (ax_stay, ax_leave):
        ax.fill_between(dt, 0, 1, where=weekend, transform=ax.get_xaxis_transform(),
                        color="grey", alpha=0.15, lw=0, label="_weekend")
        ax.grid(alpha=0.3)
    ax_leave.xaxis.set_major_locator(mdates.DayLocator())
    ax_leave.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    ax_leave.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))

    fig.suptitle(f"{title}\nshaded: weekend")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="slug of a saved β fit, e.g. ar1_hmm_beta_1")
    parser.add_argument("--room", "-r", type=int, required=True, help="room number, e.g. 9 for room_009")
    parser.add_argument("--states", "-K", type=int, default=6)
    parser.add_argument("--state", "-s", type=int, required=True, help="state the transitions are from (0-indexed)")
    parser.add_argument("--loss", help="loss folder, e.g. ridge_beta_lam1 (default: the driver's own loss)")
    parser.add_argument("--week", help="Monday the week starts on, e.g. 2023-11-20 (default: first normal training week)")
    args = parser.parse_args()

    params, cfg, train = load_saved(args.model, args.room, args.states, args.loss)
    if not hasattr(params.transition, "beta"):
        raise SystemExit(f"{args.model} has a static transition; there is nothing to plot over the week")
    if not 0 <= args.state < cfg.states:
        raise SystemExit(f"--state must be in 0..{cfg.states - 1}")

    if args.week:
        dt = week_grid(pd.Timestamp(args.week))
        if dt[0].dayofweek != 0:
            raise SystemExit(f"--week {args.week} is not a Monday")
        if not is_normal_week(dt):
            print(f"Warning: the week of {args.week} contains a holiday")
    else:
        dt = first_normal_week(train.datetime)

    xs = covariates(dt, cfg.covariate_cols)
    Gammas = np.asarray(params.transition_matrices(N=len(dt), xs=xs))   # (T, K, K)
    probs = Gammas[:, args.state, :]

    summary = pd.DataFrame(probs, columns=[f"s{args.state}->s{j}" for j in range(cfg.states)])
    summary["weekend"] = dt.dayofweek >= 5
    print(f"Week of {dt[0]:%Y-%m-%d}: mean / min / max transition probabilities out of s{args.state}")
    print(summary.groupby("weekend").agg(["mean", "min", "max"]).T.round(4).to_string())

    out = os.path.join(cfg.run_dir(PLOTS_DIR), f"{args.model}_transitions_from_s{args.state}.png")
    plot(f"{args.model} — {cfg.room}, K={cfg.states}, loss={cfg.loss.slug}\n"
         f"Transition probabilities out of s{args.state}, week of {dt[0]:%Y-%m-%d}",
         dt, probs, args.state, out)
    print(f"Saved {out}")
