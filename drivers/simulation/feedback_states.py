"""P(feedback | state) from the paths saved by `sample_paths`.

The occupant feedback is binned onto the series' timestamps (drivers.data.clean_room_data.add_feedback), so
f_t = 1 if any feedback was given in bin t. For each state s and each sampled path:

    feedback events   n_s = sum_t f_t 1[z_t = s]
    P(feedback | s)   n_s / sum_t 1[z_t = s], the share of the path's steps in s that received feedback

The CSV and the plot give the mean over paths and the 2.5-97.5 % range across paths. The dashed line in the
plot is the overall feedback rate sum_t f_t / T.

CSV in results/{room}/{tag}/states_{K}/simulation/, plot in plots/{room}/{tag}/states_{K}/simulation/.

    uv run python -m drivers.simulation.feedback_states ar1_hmm_beta_1 --room 15 -K 4
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from drivers.data.clean_room_data import add_feedback
from drivers.models.utils import PLOTS_DIR, TAG, room_name
from drivers.simulation.sample_paths import paths_file
from drivers.simulation.state_analysis import CHUNK, model_label, summary_table


def feedback_on(dt: pd.DatetimeIndex, room: str) -> np.ndarray:
    """(T,) 1 where feedback was received in the bin, else 0."""
    raw_room = room.replace("_", " ").title()  # room_015 -> Room 015, the raw feedback file name
    return add_feedback(pd.DataFrame(index=dt), raw_room)["feedback"].to_numpy(dtype=np.float32)


def feedback_per_state(paths: np.ndarray, feedback: np.ndarray, num_states: int) -> tuple[np.ndarray, np.ndarray]:
    """(P, K) feedback events and P(feedback | state) per path (NaN if the path never visits the state)."""
    events = np.empty((len(paths), num_states))
    steps = np.empty((len(paths), num_states))
    for start in range(0, len(paths), CHUNK):
        chunk = paths[start:start + CHUNK]
        for s in range(num_states):
            in_s = chunk == s
            events[start:start + len(chunk), s] = in_s @ feedback
            steps[start:start + len(chunk), s] = in_s.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return events, events / steps


def plot_feedback(summary: pd.DataFrame, overall: float, title: str, path: str):
    states = summary.index.to_numpy()
    p, lo, hi = (summary[f"p_feedback_{c}"].to_numpy() for c in ("mean", "lo_2.5", "hi_97.5"))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]  # same state colours as the daily profiles

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(states, p, width=0.6, color=[colors[(s - 1) % len(colors)] for s in states],
           yerr=np.clip([p - lo, hi - p], 0, None), capsize=5)
    ax.axhline(overall, color="black", ls="--", lw=1, label=f"Overall feedback rate ({100 * overall:.2f} %)")
    for s, top, ps, (n, n_lo, n_hi) in zip(states, hi, p, summary[["events_mean", "events_lo_2.5", "events_hi_97.5"]].to_numpy()):
        ax.annotate(f"P = {100 * ps:.2f} %\n{n:.1f} feedbacks\n[{n_lo:.0f}, {n_hi:.0f}]", (s, top), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1))  # keeps the dashed line off the text
    ax.set_xticks(states, [f"State {s}" for s in states])
    ax.set(title=title, ylabel="P(feedback | state)", ylim=(0, 1.4 * hi.max()))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=1))
    ax.legend(loc="upper left", frameon=False)
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
    paths, dt = saved["paths"], pd.DatetimeIndex(saved["datetime"])
    feedback = feedback_on(dt, room)
    n_feedback = int(feedback.sum())
    print(f"{args.model} — {room}, K={args.states}: {n_feedback} feedback events, {len(paths)} paths from {src}")

    events, p_feedback = feedback_per_state(paths, feedback, args.states)
    summary = summary_table({"events": events, "p_feedback": p_feedback})
    print(summary.round(4).to_string())
    out = os.path.join(os.path.dirname(src), f"{args.model}_feedback_given_state.csv")
    summary.to_csv(out)
    print(f"Saved {out}")

    room_title = room.replace("_", " ").title()  # room_015 -> Room 015
    path = os.path.join(PLOTS_DIR, room, args.tag, f"states_{args.states}", "simulation",
                        f"{args.model}_feedback_given_state.png")
    plot_feedback(summary, n_feedback / len(feedback),
                  f"{room_title}, {model_label(args.model)}, $K={args.states}$: "
                  f"P(feedback | state), {n_feedback} feedbacks", path)
    print(f"Saved {path}")
