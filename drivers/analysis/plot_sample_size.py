"""Plots of a saved `sample_size` run, read from `summary.csv` and `p_hat.csv`, so nothing is refitted:

    p_hat_vs_N_simple.png    the HMM and known-states prediction intervals, the mean p_hat and the true p
    p_hat_distribution.png   per N, the distribution of p_hat over the datasets and a normal QQ plot

    uv run python -m drivers.analysis.plot_sample_size --room 9 -K 5
    uv run python -m drivers.analysis.plot_sample_size --room 9 -K 5 --p 0.05 --R 100 --seed 0
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from drivers.models.utils import PLOTS_DIR, RESULTS_DIR, TAG, driver_loss, room_name


def plot_p_hat(summary: pd.DataFrame, p: float, title: str, path: str):
    Ns = summary["n_complaints"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(Ns, summary["pi_lo"], summary["pi_hi"], color="tab:blue", alpha=0.2,
                    label="95% prediction interval, HMM")
    ax.plot(Ns, summary["mean"], color="tab:blue", lw=1.5, marker="o", ms=3, label=r"mean $\hat{p}$")
    ax.plot(Ns, summary["known_lo"], "k--", lw=1, label="95% prediction interval, known states")
    ax.plot(Ns, summary["known_hi"], "k--", lw=1)
    ax.axhline(p, color="tab:red", lw=1, label=f"true p = {p:g}")
    ax.set_ylim(0, 3 * p)
    ax.set(xlabel="N, number of complaints the model was trained on", ylabel=r"$\hat{p}$")
    ax.set_title(title, loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_distribution(p_hats: pd.DataFrame, p: float, title: str, path: str):
    """Per N: histogram of p_hat over the datasets (top) and a normal QQ plot (bottom)."""
    groups = list(p_hats.groupby("n_complaints")["p_hat"])
    fig, axes = plt.subplots(2, len(groups), figsize=(3.2 * len(groups), 6.5), squeeze=False)
    for (n, p_hat), top, bottom in zip(groups, axes[0], axes[1]):
        bins = np.linspace(0, 3 * p, 31)
        top.hist(p_hat.clip(upper=bins[-1]), bins=bins, density=True, color="tab:blue", alpha=0.5,
                 label=rf"HMM $\hat{{p}}$ (R={len(p_hat)})")
        top.axvline(p, color="tab:red", lw=1, label=f"true p = {p:g}")
        top.set_title(f"N = {n}", loc="left", fontsize=10)
        top.set_xlabel(r"$\hat{p}$")
        stats.probplot(p_hat, dist="norm", plot=bottom)
        bottom.set_title("")  # drop probplot's default title
        bottom.set_title("normal QQ", loc="left", fontsize=10)
        bottom.get_lines()[0].set(markersize=3, color="tab:blue")
        bottom.get_lines()[1].set(color="k", lw=1)
    axes[0, 0].set_ylabel("density")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--room", "-r", type=int, required=True, help="room number, e.g. 9 for room_009")
    parser.add_argument("--states", "-K", type=int, default=5)
    parser.add_argument("--model", default="ar1_hmm_beta_1", help="slug of the model the sample_size run used")
    parser.add_argument("--tag", default=TAG, help=f"data tag (default {TAG})")
    parser.add_argument("--p", type=float, default=0.05, help="true P(complaint | occupied state) of the run")
    parser.add_argument("--R", type=int, default=100, help="number of simulated datasets in the run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-noise", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    room, loss = room_name(args.room), driver_loss(args.model).slug
    name = f"{args.model}_p{args.p:g}_R{args.R}_seed{args.seed}" + (f"_noise{args.init_noise:g}" if args.init_noise else "")
    run = os.path.join(room, args.tag, f"states_{args.states}", "sample_size", name)

    summary = pd.read_csv(os.path.join(RESULTS_DIR, run, "summary.csv"))
    plot_dir = os.path.join(PLOTS_DIR, run)
    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, "p_hat_vs_N_simple.png")
    title = f"{args.model}, {room}, K={args.states}, loss={loss}, p={args.p:g}, R={args.R}"
    plot_p_hat(summary, args.p, title, path)
    p_hats = pd.read_csv(os.path.join(RESULTS_DIR, run, "p_hat.csv"))
    plot_distribution(p_hats, args.p, title, os.path.join(plot_dir, "p_hat_distribution.png"))
    print(f"Saved plots to {plot_dir}")
