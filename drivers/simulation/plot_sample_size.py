"""Plots of a saved `sample_size` run, read from its CSVs and run.json, so nothing is refitted:

    p_feedback_vs_N.png     p_hat_s against N per state: per-dataset lines, 95 % CI of the mean, 95 % PI and the truth
    cutpoint_se_vs_N.png    standard error of each cut-point c_k against N (log-log), with the target SE and the
                            N from which every cut-point stays below it

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
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

from drivers.simulation.sample_size import CAT_LABELS, CUT_NAMES, add_run_args, run_dirs


def plot_bands(ax, estimates: pd.DataFrame, summary: pd.DataFrame, param: str, est: str, color: str, label: str):
    s = summary[(summary["parameter"] == param) & (summary["estimator"] == est)]
    wide = estimates[(estimates["parameter"] == param) & (estimates["estimator"] == est)] \
        .pivot(index="n_votes", columns="dataset", values="estimate")
    ax.plot(wide.index, wide.to_numpy(), color="0.85", lw=0.5, zorder=1)
    ax.fill_between(s["n_votes"], s["pi_lo"], s["pi_hi"], color=color, alpha=0.15, label=f"95% PI, {label}")
    ax.fill_between(s["n_votes"], s["ci_lo"], s["ci_hi"], color=color, alpha=0.4, label=f"95% CI of mean, {label}")
    ax.plot(s["n_votes"], s["mean"], color=color, lw=1.5, label=f"mean, {label}")


def plot_p(estimates: pd.DataFrame, summary: pd.DataFrame, run: dict, path: str):
    """One panel per state: the HMM-weighted p_hat_s per dataset, their CI/PI against N and the true p_s."""
    K = run["states"]
    fig, axes = plt.subplots(-(-K // 2), 2, figsize=(10, 3.6 * -(-K // 2)), sharex=True, squeeze=False)
    for s, ax in enumerate(axes.flat[:K]):
        name = f"p_{s}"
        true = summary.loc[summary["parameter"] == name, "true"].iloc[0]
        plot_bands(ax, estimates, summary, name, "hmm", "tab:blue", "HMM")
        ax.axhline(true, color="tab:red", lw=1, label="true")
        ax.set_ylim(0, 3 * max(true, run["p_bar"]))
        ax.set_title(f"$p_{s}$ = P(feedback | state {s})", loc="left", fontsize=10)
    for ax in axes.flat[K:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("N, number of votes")
    axes.flat[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle(run["title"], fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def n_below(n: np.ndarray, se: np.ndarray, target: float) -> int | None:
    """First N from which se stays <= target (NaN counts as above); None if it never does."""
    above = np.flatnonzero(~(se <= target))
    i = above[-1] + 1 if len(above) else 0
    return int(n[i]) if i < len(n) else None


def plot_cut_se(summary: pd.DataFrame, run: dict, target: float, path: str) -> dict[str, int | None]:
    """SE of each cut-point (sd of c_hat_k over the datasets) against N, on log axes.

    An N where fewer than half the datasets had a usable logit fit is left out, as its sd rests on a few fits.
    """
    fig, ax = plt.subplots(figsize=(8, 4.8))
    needed = {}
    for k, name in enumerate(CUT_NAMES):
        s = summary[(summary["parameter"] == name) & (summary["estimator"] == "logit")]
        n, se = s["n_votes"].to_numpy(), s["sd"].where(s["R"] >= run["R"] / 2).to_numpy()
        line, = ax.plot(n, se, marker="o", ms=3, lw=1.5,
                        label=f"{run['cutpoints'][k]:g} °C ({CAT_LABELS[k]} | {CAT_LABELS[k + 1]})")
        needed[name] = n_below(n, se, target)
        if needed[name]:
            ax.axvline(needed[name], color=line.get_color(), ls=":", lw=1)
    ax.axhline(target, color="k", ls="--", lw=1)
    if all(needed.values()):
        n_all = max(needed.values())
        ax.annotate(f"N ≈ {n_all} for all cut-points\nwithin SE ≤ {target:g} °C", xy=(n_all, target),
                    xytext=(n_all * 1.4, target * 2), fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [n for n in (5, 10, 25, 50, 100, 200, 300, 400, 800, 1600) if n <= run["n_max"]]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator([0.1, 0.15, 0.25, 0.5, 1, 1.5, 2]))
    ax.set(xlabel="N, number of votes", ylabel="standard error of cut-point [°C]",
           title="Precision of the category cut-points vs. N")
    ax.grid(alpha=0.3)
    ax.legend(title="Cut-point", frameon=True, fontsize=8, title_fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return needed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    parser.add_argument("--se-target", type=float, default=0.25,
                        help="cut-point standard error [°C] counted as precise enough (0.25 ≈ ±0.5 °C at 95 %%)")
    args = parser.parse_args()
    out_dir, plot_dir = run_dirs(args)
    os.makedirs(plot_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run.json")) as f:
        run = json.load(f)
    read = lambda name: pd.read_csv(os.path.join(out_dir, f"{name}.csv"))
    estimates = read("estimates")
    estimates = estimates[estimates["kind"] == "alt"]
    summary = read("summary")

    plot_p(estimates, summary, run, os.path.join(plot_dir, "p_feedback_vs_N.png"))
    needed = plot_cut_se(summary, run, args.se_target, os.path.join(plot_dir, "cutpoint_se_vs_N.png"))
    for name, n in needed.items():
        print(f"{name}: SE <= {args.se_target:g} °C from N = {n if n else '> %d' % run['n_max']}")
    print(f"Saved plots to {plot_dir}")
