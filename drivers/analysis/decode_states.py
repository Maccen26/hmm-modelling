"""Plot the decoded hidden states of a saved fit over a window of the data.

Smoothed state probabilities P(z_t = s | y_1:T) are estimated as the share of paths drawn
with BackwardSampler that are in state s at time t; the top panel colours each observation by
its most probable state. The validation split is filtered through the training set first.

    uv run python -m drivers.analysis.decode_states ar1_hmm_beta_1 --room 9 -K 4
    uv run python -m drivers.analysis.decode_states ar1_hmm_beta_1 --room 9 -K 4 --split train --start-day 14 --days 7
"""
import jax
jax.config.update("jax_enable_x64", True)

import argparse
import os

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from drivers.models.evaluate import _covariates
from drivers.models.utils import PLOTS_DIR, fit_path, load_data, room_name, working_hours
from drivers.utils import load_model
from src.api.v5 import BackwardSampler, ForwardAlgorithm, Params


def smoothed_probs(params: Params, ys, xs, init_dist, num_paths: int, seed: int = 0) -> np.ndarray:
    """(T, K) Monte Carlo estimate of P(z_t | y_1:T) from backward-sampled paths."""
    utt, _, _ = ForwardAlgorithm.run(params=params, initial_dist=init_dist, ys=ys, xs=xs)
    Gammas = params.transition_matrices(N=len(ys), xs=xs)
    paths = np.asarray(BackwardSampler.run(num_paths, Gammas, utt, seed=seed))   # (P, T)
    K = Gammas.shape[1]
    return np.stack([(paths == s).mean(axis=0) for s in range(K)], axis=1)


def state_labels(params: Params, ys) -> list[str]:
    em = params.emission
    mu = np.asarray(em.mu_vals(0, ys) if hasattr(em, "mu_vals") else em.mu(0, ys))
    sigma = np.asarray(em.sigma(0, ys))
    phi = np.asarray(em.phi())[0] if hasattr(em, "phi") else None
    return [f"s{s}: μ={mu[s]:.0f}, σ={sigma[s]:.0f}" + (f", φ={phi[s]:.2f}" if phi is not None else "")
            for s in range(len(mu))]


def shade_working_hours(ax, dt: pd.DatetimeIndex):
    on = working_hours(dt).astype(bool)
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    for start, stop in zip(edges[::2], edges[1::2]):
        ax.axvspan(dt[start], dt[stop - 1], color="grey", alpha=0.12, lw=0)


def plot(title: str, dt: pd.DatetimeIndex, y: np.ndarray, probs: np.ndarray, labels: list[str], path: str):
    colors = plt.cm.tab10(np.arange(probs.shape[1]))
    fig, (ax_y, ax_p) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    ax_y.plot(dt, y, color="black", lw=0.8, zorder=1)
    ax_y.scatter(dt, y, c=colors[probs.argmax(axis=1)], s=12, zorder=2)
    for s, label in enumerate(labels):
        ax_y.scatter([], [], color=colors[s], label=label)
    ax_y.set(ylabel="CO₂ [ppm]", title="Observations coloured by most probable state (grey: working hours)")
    ax_y.legend(loc="upper left", fontsize=8)

    ax_p.stackplot(dt, probs.T, colors=colors, step="post")
    ax_p.set(ylabel="P(state | all data)", ylim=(0, 1), title="Smoothed state probabilities")

    for ax in (ax_y, ax_p):
        shade_working_hours(ax, dt)
    fig.suptitle(title)
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
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--start-day", type=float, default=0, help="window start, days after the split begins")
    parser.add_argument("--days", type=float, default=7, help="window length in days")
    parser.add_argument("--paths", type=int, default=500, help="backward-sampled paths for the smoothed probabilities")
    args = parser.parse_args()

    path = fit_path(room_name(args.room), args.states, args.model, args.loss)
    if not os.path.exists(path):
        raise SystemExit(f"No saved fit at {path}")
    saved = load_model(path)
    params, cfg, train = saved["result"].params, saved["config"], saved["data"]

    ys, xs, dt, offset = train.ys, _covariates(params, train), train.datetime, 0
    if args.split == "val":
        val = load_data(cfg, "val")
        offset = len(train.ys)
        ys, dt = jnp.concatenate([train.ys, val.ys]), train.datetime.append(val.datetime)
        xs = None if xs is None else jnp.concatenate([xs, _covariates(params, val)])
    probs = smoothed_probs(params, ys, xs, train.init_dist, args.paths)

    start = dt[offset] + pd.Timedelta(days=args.start_day)
    window = np.flatnonzero((dt >= start) & (dt < start + pd.Timedelta(days=args.days)) & (np.arange(len(dt)) >= offset))
    if window.size == 0:
        raise SystemExit("Window is empty; lower --start-day")
    y = np.asarray(ys)
    y = y if y.ndim == 1 else y[:, 0]

    out = os.path.join(cfg.run_dir(PLOTS_DIR), f"{args.model}_states_{args.split}.png")
    plot(f"{args.model} — {cfg.room}, K={cfg.states}, loss={cfg.loss.slug}: decoded states, "
         f"{args.split} split from {start:%Y-%m-%d}",
         dt[window], y[window], probs[window], state_labels(params, train.ys), out)
    print(f"Saved {out}")
