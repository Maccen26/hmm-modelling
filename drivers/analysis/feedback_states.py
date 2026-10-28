"""Histogram of the hidden states the occupant feedback was received in.

State paths are drawn with BackwardSampler (forward filtering, backward sampling) as in week_7.ipynb.
For each path the feedback events are counted per state, giving one estimate of P(z_t = s | feedback_t = 1)
per path; the bars are the mean over paths and the error bars the 2.5-97.5 % range across paths.
The grey bars are the share of all time steps spent in each state; the exact value from the
forward-backward smoothed probabilities is printed as a check. Validation/test splits are filtered through the training set first.

    uv run python -m drivers.analysis.feedback_states ar1_hmm_beta_1 --room 9 -K 6
    uv run python -m drivers.analysis.feedback_states ar1_hmm_beta_1 --room 9 -K 6 --split val --paths 2000
"""
import jax
jax.config.update("jax_enable_x64", True)

import argparse
import os

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from drivers.analysis.state_confusion import load_saved
from drivers.models.evaluate import _covariates
from drivers.models.utils import PLOTS_DIR, load_data
from drivers.utils import load_y_df
from src.api.v5 import BackwardSampler, ForwardAlgorithm
from src.api.v5.inference.forward_backward import ForwardBackwardAlgorithm


def load_feedback(room: str, tag: str, split: str) -> jnp.ndarray:
    """(T,) 1 where feedback was received, else 0; ordered like load_data."""
    return jnp.asarray(load_y_df(room, tag, split).sort_values("datetime")["feedback"].to_numpy(), dtype=float)


def state_counts(paths: jnp.ndarray, weights: jnp.ndarray, num_states: int) -> np.ndarray:
    """(P, K) weighted number of time steps each path spends in each state."""
    count = lambda path: jnp.bincount(path, weights=weights, length=num_states)
    return np.asarray(jax.vmap(count)(paths))


def plot(title: str, per_path: np.ndarray, time_share: np.ndarray, n_feedback: int, path: str):
    states = np.arange(per_path.shape[1])
    pooled = per_path.mean(axis=0)
    lo, hi = np.percentile(per_path, [2.5, 97.5], axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(states, time_share, width=0.8, color="lightgrey")
    ax.bar(states, pooled, width=0.5, color="tab:blue",
           yerr=np.clip([pooled - lo, hi - pooled], 0, None), capsize=4)
    ax.set_xticks(states, [f"s{s}" for s in states])
    ax.set(xlabel="State", ylabel="P(state | feedback)", ylim=(0, max(1e-3, hi.max(), time_share.max()) * 1.15))
    ax.secondary_yaxis("right", functions=(lambda p: p * n_feedback, lambda c: c / n_feedback)) \
      .set_ylabel("Feedback events")
    ax.legend(handles=[
        Patch(facecolor="lightgrey", label="Share of all time steps"),
        Patch(facecolor="tab:blue", label="Share of feedback (mean of sampled paths; error bar: 2.5–97.5 % quantiles)"),
    ], loc="upper left", fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="slug of the saved fit, e.g. ar1_hmm_beta_1")
    parser.add_argument("--room", "-r", type=int, required=True, help="room number, e.g. 9 for room_009")
    parser.add_argument("--states", "-K", type=int, default=6)
    parser.add_argument("--loss", help="loss folder, e.g. ridge_beta_lam1 (default: the driver's own loss)")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--paths", type=int, default=10_000, help="number of backward-sampled state paths")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    params, cfg, train = load_saved(args.model, args.room, args.states, args.loss)
    ys, xs, offset = train.ys, _covariates(params, train), 0
    if args.split != "train":
        split = load_data(cfg, args.split)
        offset = len(train.ys)
        ys = jnp.concatenate([train.ys, split.ys])
        xs = None if xs is None else jnp.concatenate([xs, _covariates(params, split)])
    feedback = load_feedback(cfg.room, cfg.tag, args.split)
    if len(feedback) != len(ys) - offset:
        raise SystemExit(f"Feedback has {len(feedback)} rows but the {args.split} split has {len(ys) - offset}")
    n_feedback = int(feedback.sum())
    if n_feedback == 0:
        raise SystemExit(f"No feedback events in the {args.split} split of {cfg.room}")

    utt, _, _ = ForwardAlgorithm.run(params=params, initial_dist=train.init_dist, ys=ys, xs=xs)
    Gammas = params.transition_matrices(N=len(ys), xs=xs)
    paths = BackwardSampler.run(args.paths, Gammas, utt, seed=args.seed)[:, offset:]   # (P, T_split)

    per_path = state_counts(paths, feedback, cfg.states) / n_feedback
    time_share = state_counts(paths, jnp.ones_like(feedback), cfg.states).mean(axis=0) / len(feedback)
    smoothed = np.asarray(ForwardBackwardAlgorithm.run(params=params, initial_dist=train.init_dist, ys=ys, xs=xs))
    exact = np.asarray(feedback) @ smoothed[offset:] / n_feedback

    lo, hi = np.percentile(per_path, [2.5, 97.5], axis=0)
    table = pd.DataFrame({"share": per_path.mean(axis=0), "lo_2.5": lo, "hi_97.5": hi,
                          "forward_backward": exact, "time_share": time_share},
                         index=pd.Index(range(cfg.states), name="state"))
    print(f"{n_feedback} feedback events in the {args.split} split ({args.paths} sampled paths)")
    print(table.round(3).to_string())

    out = os.path.join(cfg.run_dir(PLOTS_DIR), f"{args.model}_feedback_states_{args.split}.png")
    plot(f"{args.model} — {cfg.room}, K={cfg.states}, loss={cfg.loss.slug}\n"
         f"States of the {n_feedback} feedback events, {args.split} split",
         per_path, time_share, n_feedback, out)
    print(f"Saved {out}")
