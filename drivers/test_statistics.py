import os

import jax.numpy as jnp
import pandas as pd

from drivers.utils import load_model, write_latex_table
from src.api.v4 import HMM
from src.api.v4.utils import load_y_data


def lrt(model0: HMM, model1: HMM):
    ll0 = model0.log_likelihood()
    ll1 = model1.log_likelihood()
    test_statistic = 2 * (ll1 - ll0)
    return test_statistic


def aic(model: HMM):
    k = len(model.params)
    ll = model.log_likelihood()
    return 2 * k - 2 * ll


def bic(model: HMM, lag: int = 0):
    num_samples = len(load_y_data())
    num_samples = num_samples - lag if lag > 0 else num_samples
    k = len(model.params)
    ll = model.log_likelihood()
    return float(k * jnp.log(num_samples) - 2 * ll)


def p_value(test_statistic: float, df: int):
    from scipy.stats import chi2
    return float(1 - chi2.cdf(test_statistic, df))


MODEL_LABELS = {
    "ordinary_hmm": "HMM(1)",
    "ar_hmm": "AR(1), HMM(1)",
    "ar_2_hmm": "AR(2), HMM(1)",
    "second_order_hmm": "AR(1), HMM(2)",
    "ar_2_second_order_hmm": "AR(2), HMM(2)",
    "covariate_hmm": "HMM(1) + covariates",
}


def build_model_stats_df(models):
    rows = []
    for model_name, lag in models:
        model = load_model(f"results/models/{model_name}.pkl")
        rows.append({
            "Model": MODEL_LABELS.get(model_name, model_name),
            "#Params": int(len(model.params)),
            "LogLik": float(model.log_likelihood()),
            "AIC": float(aic(model)),
            "BIC": float(bic(model, lag=lag)),
        })
    return pd.DataFrame(rows)


def build_lrt_comparison_df(edges):
    rows = []
    for base_name, expanded_name, lag in edges:
        base = load_model(f"results/models/{base_name}.pkl")
        expanded = load_model(f"results/models/{expanded_name}.pkl")
        df = int(len(expanded.params) - len(base.params))
        test_stat = float(lrt(base, expanded))
        pval = p_value(test_stat, df) if df > 0 else float("nan")
        d_aic = float(aic(expanded) - aic(base))
        d_bic = float(bic(expanded, lag=lag) - bic(base, lag=lag))
        rows.append({
            "Base Model": MODEL_LABELS.get(base_name, base_name),
            "Expanded Model": MODEL_LABELS.get(expanded_name, expanded_name),
            "LRT": test_stat,
            "df": df,
            "P-val": pval,
            "ΔAIC": d_aic,
            "ΔBIC": d_bic,
        })
    return pd.DataFrame(rows)


def main_test_statistics():
    models = [
        ("ordinary_hmm", 0),
        ("ar_hmm", 1),
        ("ar_2_hmm", 2),
        ("second_order_hmm", 1),
        ("ar_2_second_order_hmm", 2),
        ("covariate_hmm", 0),
    ]
    # Edges follow the hierarchy diagram (docs/diagrams/06_model_hierarchy.puml).
    # Lag for ΔBIC is the larger of the two so both BIC values are computed on
    # the same sample size.
    #
    # covariate_hmm nests the ordinary HMM (setting beta=0 recovers the static
    # transition), so it is a valid LRT against ordinary_hmm. Both use a plain
    # Gaussian emission (no AR lag), so lag=0.
    edges = [
        ("ordinary_hmm", "ar_hmm", 1),
        ("ar_hmm", "ar_2_hmm", 2),
        ("ar_hmm", "second_order_hmm", 1),
        ("ar_2_hmm", "ar_2_second_order_hmm", 2),
        ("second_order_hmm", "ar_2_second_order_hmm", 2),
        ("ordinary_hmm", "covariate_hmm", 0),
    ]

    stats_df = build_model_stats_df(models)
    lrt_df = build_lrt_comparison_df(edges)

    print(stats_df.to_string(index=False))
    print()
    print(lrt_df.to_string(index=False))

    os.makedirs("results/test_statistics", exist_ok=True)
    stats_df.to_csv("results/test_statistics/model_stats.csv", index=False)
    lrt_df.to_csv("results/test_statistics/lrt_comparison.csv", index=False)

    write_latex_table(
        stats_df,
        "report/model_results/comparison/model_stats.tex",
        caption="Per-model log-likelihood, AIC and BIC for the fitted HMMs.",
        label="tab:model_stats",
    )
    write_latex_table(
        lrt_df,
        "report/model_results/comparison/lrt_comparison.tex",
        caption=(
            "Likelihood-ratio tests for each nested pair from the model "
            "hierarchy (Figure~\\ref{fig:diagram:model_hierarchy}). "
            "Negative $\\Delta$AIC/$\\Delta$BIC favour the expanded model."
        ),
        label="tab:lrt_comparison",
        float_cols_4dp=["P-val"],
    )


if __name__ == "__main__":
    main_test_statistics()
