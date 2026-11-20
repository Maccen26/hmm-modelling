"""LaTeX table comparing the four models of a random search across numbers of states.

For each model and K it reports the number of free parameters, the training log-likelihood (the
unpenalised NLL of the chosen fit, so the ridge-penalised beta models are comparable), AIC, BIC and the
forecast RMSE / MAPE at one horizon. Every model but HMM(1) also gets the likelihood-ratio statistics
2(l_full - l_reduced) and their chi^2 p-values against its nested models (`NESTED`): HMM(b,1) and
AR(1)-HMM(1) against HMM(1), and AR(1)-HMM(b,1) against AR(1)-HMM(1) (beta = 0) and HMM(b,1) (phi = 0).
Writes `Thesis/Tables/<name>.tex`.

    uv run python -m drivers.analysis.model_comparison_table \
        results/room_015/calibrated_anchor/random_search --states 3 4 5 6 --horizon 6 [--split val|test]
"""
import jax
jax.config.update("jax_enable_x64", True)

import argparse
import os

import equinox as eqx
import numpy as np
import pandas as pd
from scipy.stats import chi2

from drivers.analysis.predictive_metrics_across_states_table import SPLIT_NAMES, load_metrics
from drivers.utils import THESIS_TABLES_DIR, load_model

# slug -> row label, in table order
MODELS = {
    "hmm_1": "HMM",
    "hmm_beta_1": "HMM($\\beta$)",
    "ar1_hmm_1": "AR(1)-HMM",
    "ar1_hmm_beta_1": "AR(1)-Covariates HMM",
}
# full slug -> the nested models it is tested against, in the order shown in its LRT / p-value cells
NESTED = {
    "hmm_beta_1": ("hmm_1",),               # beta = 0
    "ar1_hmm_1": ("hmm_1",),                # phi = 0
    "ar1_hmm_beta_1": ("ar1_hmm_1", "hmm_beta_1"),
}


def n_params(params) -> int:
    """Number of free (trainable) parameters in the fitted pytree."""
    return sum(leaf.size for leaf in jax.tree_util.tree_leaves(eqx.filter(params, eqx.is_inexact_array)))


def fit_summary(results_dir: str, states: list[int]) -> pd.DataFrame:
    """One row per (K, slug): p, LL, AIC, BIC and the training-set size n."""
    rows = []
    for k in states:
        path = os.path.join(results_dir, f"states_{k}")
        cands = pd.read_csv(os.path.join(path, "candidates.csv"))
        chosen = cands[cands["chosen"]].set_index("slug")
        for slug in MODELS:
            model = load_model(os.path.join(path, f"{slug}.pkl"))
            p, n, ll = n_params(model["result"].params), len(model["data"].ys), -chosen.loc[slug, "nll"]
            rows.append({"K": k, "slug": slug, "p": p, "n": n, "LL": ll,
                         "AIC": 2 * p - 2 * ll, "BIC": p * np.log(n) - 2 * ll})
    return pd.DataFrame(rows)


def add_lrt(df: pd.DataFrame) -> pd.DataFrame:
    """LRT statistic, df and p-value of each full model against each of its nested models, per K."""
    rows = []
    for k, group in df.groupby("K"):
        g = group.set_index("slug")
        for full, nested in NESTED.items():
            for reduced in nested:
                stat = 2 * (g.loc[full, "LL"] - g.loc[reduced, "LL"])
                dof = g.loc[full, "p"] - g.loc[reduced, "p"]
                rows.append({"K": k, "full": full, "reduced": reduced, "stat": stat, "df": dof,
                             "p_value": chi2.sf(stat, dof)})
    return pd.DataFrame(rows)


def num(x: float, decimals: int = 1) -> str:
    """Number with a proper LaTeX minus sign."""
    s = f"{x:.{decimals}f}"
    return f"${s}$" if s.startswith("-") else s


def p_value(p: float) -> str:
    return "$<0.001$" if p < 1e-3 else f"{p:.3f}"


def to_latex(df: pd.DataFrame, lrt: pd.DataFrame, horizon: int, split: str, name: str) -> str:
    header = ("Model Specification & States ($K$) & Parameters ($p$) & Log-Likelihood ($\\ell$) & AIC & BIC "
              "& RMSE & MAPE (\\%) & LRT & $p$-value \\\\")
    blocks = []
    for slug, label in MODELS.items():
        lines = []
        for i, r in enumerate(df[df["slug"] == slug].sort_values("K").itertuples()):
            tests = lrt[(lrt["K"] == r.K) & (lrt["full"] == slug)].set_index("reduced")
            lrt_cell = " / ".join(num(tests.loc[s, "stat"]) for s in NESTED.get(slug, ())) or "--"
            p_cell = " / ".join(p_value(tests.loc[s, "p_value"]) for s in NESTED.get(slug, ())) or "--"
            cells = [label if i == 0 else "", str(r.K), str(r.p), num(r.LL), num(r.AIC), num(r.BIC),
                     num(r.RMSE), num(r.MAPE, 2), lrt_cell, p_cell]
            lines.append(" & ".join(cells) + " \\\\")
        blocks.append("\n".join(lines))

    n = int(df["n"].iloc[0])
    caption = (f"Model comparison for $K = {df['K'].min()}, \\dots, {df['K'].max()}$ states. "
               f"$\\ell$ is the log-likelihood on the training set ($n = {n}$; for the covariate models without the "
               f"ridge penalty), AIC $= 2p - 2\\ell$ and BIC $= p\\ln n - 2\\ell$. "
               f"RMSE (ppm) and MAPE are for the {horizon}-steps-ahead forecasts on the "
               f"{SPLIT_NAMES[split]} set. LRT gives $2(\\ell_{{\\mathrm{{full}}}} - \\ell_{{\\mathrm{{reduced}}}})$ "
               f"with its $\\chi^2$ $p$-value: HMM($\\beta$) and AR(1)-HMM against the HMM "
               f"($2K(K-1)$ and $K$ degrees of freedom), and the AR(1)-Covariates HMM against the AR(1)-HMM / "
               f"HMM($\\beta$) ($2K(K-1)$ / $K$ degrees of freedom).")
    return (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\begin{adjustbox}{max width=\\textwidth}\n"
        "\\begin{tabular}{llrrrrrrcc}\n"
        "\\hline\n"
        f"{header}\n"
        "\\hline\n"
        + "\n\\hline\n".join(blocks) + "\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{adjustbox}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{tab:{name}}}\n"
        "\\end{table}\n"
    )


def default_name(results_dir: str, split: str, horizon: int) -> str:
    """e.g. `model_comparison_room_015_calibrated_anchor_random_search_val_h6`."""
    parts = os.path.normpath(results_dir).split(os.sep)
    parts = parts[parts.index("results") + 1:] if "results" in parts else parts[-3:]
    return "_".join(["model_comparison", *parts, split, f"h{horizon}"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_dir", help="folder holding the states_{K}/ subfolders")
    parser.add_argument("--states", "-K", type=int, nargs="+", default=[3, 4, 5, 6])
    parser.add_argument("--horizon", "-H", type=int, default=6, help="forecast horizon in steps (default 6)")
    parser.add_argument("--split", choices=SPLIT_NAMES, default="val")
    parser.add_argument("--name", help="table file name in Thesis/Tables/ (default: from the folder, split and horizon)")
    args = parser.parse_args()

    metrics = load_metrics(args.results_dir, args.split)
    metrics = metrics[metrics["horizon"] == args.horizon][["K", "slug", "RMSE", "MAPE"]]
    table = fit_summary(args.results_dir, args.states).merge(metrics, on=["K", "slug"], how="left")
    lrt = add_lrt(table)
    print(table.drop(columns="n").to_string(index=False))
    print(lrt.to_string(index=False))

    name = args.name or default_name(args.results_dir, args.split, args.horizon)
    path = THESIS_TABLES_DIR / f"{name}.tex"
    path.write_text(to_latex(table, lrt, args.horizon, args.split, name))
    print(f"wrote {path}")
