"""Plots of a saved `sample_size` run, read from its CSVs and run.json, so nothing is refitted:

    p_feedback_vs_N.png     p_hat_s against N per state: per-dataset lines, mean, 95 % CI of the mean, the 2.5-97.5 %
                            quantiles over the datasets and the truth
    cutpoints_vs_N.png      the same for each cut-point c_k of the ordered-logit comfort curve

    uv run python -m drivers.simulation.plot_sample_size --room 15 -K 4
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from drivers.simulation.sample_size import CAT_LABELS, CUT_NAMES, add_run_args, run_dirs

BANDS = ["mean", "ci_lo", "ci_hi", "q_lo", "q_hi"]


def plot_bands(ax, estimates: pd.DataFrame, summary: pd.DataFrame, param: str, est: str, color: str, label: str):
    s = summary[(summary["parameter"] == param) & (summary["estimator"] == est)]
    wide = estimates[(estimates["parameter"] == param) & (estimates["estimator"] == est)] \
        .pivot(index="n_votes", columns="dataset", values="estimate")
    ax.plot(wide.index, wide.to_numpy(), color="0.85", lw=0.5, zorder=1)
    ax.fill_between(s["n_votes"], s["q_lo"], s["q_hi"], color=color, alpha=0.15, label=f"95% quantiles, {label}")
    ax.fill_between(s["n_votes"], s["ci_lo"], s["ci_hi"], color=color, alpha=0.4, label=f"95% CI of mean, {label}")
    ax.plot(s["n_votes"], s["mean"], color=color, lw=1.5, label=f"mean, {label}")


def panels(n: int):
    fig, axes = plt.subplots(-(-n // 2), 2, figsize=(10, 3.6 * -(-n // 2)), sharex=True, squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes.flat[max(n - 2, 0):n]:  # the bottom panel of each column, also when n is odd
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.set_xlabel("N, number of votes")
    return fig, axes


def finish(fig, axes, run: dict, path: str):
    axes.flat[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle(run["title"], fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_p(estimates: pd.DataFrame, summary: pd.DataFrame, run: dict, path: str):
    """One panel per state: the HMM-weighted p_hat_s per dataset, their CI/quantiles against N and the true p_s."""
    K = run["states"]
    fig, axes = panels(K)
    for s, ax in enumerate(axes.flat[:K]):
        name = f"p_{s}"
        rows = summary[(summary["parameter"] == name) & (summary["estimator"] == "hmm")]
        true = rows["true"].iloc[0]
        plot_bands(ax, estimates, summary, name, "hmm", "tab:blue", "HMM")
        ax.axhline(true, color="tab:red", lw=1, label="true")
        ax.set_ylim(0, max(2 * true, 1.5 * rows.loc[rows["n_votes"] >= 25, "q_hi"].max()))
        ax.set_title(f"$p_{s}$ = P(feedback | state {s})", loc="left", fontsize=10)
    finish(fig, axes, run, path)


def plot_cutpoints(estimates: pd.DataFrame, summary: pd.DataFrame, run: dict, path: str, half_width: float = 2.5):
    """One panel per cut-point: c_hat_k per dataset, their CI/quantiles against N and the true c_k.

    An N where fewer than half the datasets had a usable logit fit gets no mean or bands, as they rest on a few fits.
    """
    summary = summary.copy()
    summary.loc[summary["R"] < run["R"] / 2, BANDS] = np.nan
    fig, axes = panels(len(CUT_NAMES))
    for k, (name, ax) in enumerate(zip(CUT_NAMES, axes.flat)):
        true = run["cutpoints"][k]
        plot_bands(ax, estimates, summary, name, "logit", "tab:blue", "ordered logit")
        ax.axhline(true, color="tab:red", lw=1, label="true")
        ax.set_ylim(true - half_width, true + half_width)
        ax.set_title(f"$c_{k + 1}$ = {true:g} °C ({CAT_LABELS[k]} | {CAT_LABELS[k + 1]})", loc="left", fontsize=10)
        ax.set_ylabel("cut-point [°C]")
    finish(fig, axes, run, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args()
    out_dir, plot_dir = run_dirs(args)
    os.makedirs(plot_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run.json")) as f:
        run = json.load(f)
    read = lambda name: pd.read_csv(os.path.join(out_dir, f"{name}.csv"))
    estimates, summary = read("estimates"), read("summary")

    plot_p(estimates, summary, run, os.path.join(plot_dir, "p_feedback_vs_N.png"))
    plot_cutpoints(estimates, summary, run, os.path.join(plot_dir, "cutpoints_vs_N.png"))
    print(f"Saved plots to {plot_dir}")
