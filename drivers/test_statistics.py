"""Compute model comparison statistics for a fitted model family.

For a given ``data_name`` / ``tag`` this loads every fitted model from
``results/models/{data_name}/{tag}`` and writes two CSVs to
``results/test_statistics/{data_name}/{tag}``:

* ``model_stats.csv`` -- per-model #params, log-likelihood, AIC and BIC.
* ``lrt_comparison.csv`` -- likelihood-ratio tests for each nested pair.

LaTeX rendering lives in ``drivers.latex``.

Run with ``python -m drivers.test_statistics``.
"""

import os

import jax.numpy as jnp
import pandas as pd

from drivers.utils import load_model, load_train_data
from src.api.v4 import HMM


def lrt(model0: HMM, model1: HMM):
    ll0 = model0.log_likelihood()
    ll1 = model1.log_likelihood()
    test_statistic = 2 * (ll1 - ll0)
    return test_statistic


def aic(model: HMM):
    k = (model.no_of_free_params)
    ll = model.log_likelihood()
    return 2 * k - 2 * ll


def bic(model: HMM, num_samples: int, lag: int = 0):
    num_samples = num_samples - lag if lag > 0 else num_samples
    k = (model.no_of_free_params)
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
    "covariate_hmm": "Covariates-HMM(1)",
    "ar_1_covariate_hmm": "AR(1), Covariates-HMM(1)",
    "ar_2_covariate_hmm": "AR(2), Covariates-HMM(1)",
}


def _model_path(data_name: str, tag: str, model_name: str) -> str:
    return f"results/models/{data_name}/{tag}/{model_name}.pkl"


def _results_dir(data_name: str, tag: str) -> str:
    return f"results/test_statistics/{data_name}/{tag}"


def build_model_stats_df(models, data_name: str, tag: str, num_samples: int):
    rows = []
    for model_name, lag in models:
        path = _model_path(data_name, tag, model_name)
        if not os.path.exists(path):
            print(f"skipping {model_name}: {path} not found")
            continue
        model = load_model(path)
        rows.append({
            "Model": MODEL_LABELS.get(model_name, model_name),
            "#Params": int((model.no_of_free_params)),
            "LogLik": float(model.log_likelihood()),
            "AIC": float(aic(model)),
            "BIC": float(bic(model, num_samples, lag=lag)),
        })
    return pd.DataFrame(rows)


def build_lrt_comparison_df(edges, data_name: str, tag: str, num_samples: int):
    rows = []
    for base_name, expanded_name, lag in edges:
        base_path = _model_path(data_name, tag, base_name)
        expanded_path = _model_path(data_name, tag, expanded_name)
        if not (os.path.exists(base_path) and os.path.exists(expanded_path)):
            print(f"skipping LRT {base_name} -> {expanded_name}: missing model")
            continue
        base = load_model(base_path)
        expanded = load_model(expanded_path)
        df = int(expanded.no_of_free_params - base.no_of_free_params)
        test_stat = float(lrt(base, expanded))
        pval = p_value(test_stat, df) if df > 0 else float("nan")
        d_aic = float(aic(expanded) - aic(base))
        d_bic = float(bic(expanded, num_samples, lag=lag) - bic(base, num_samples, lag=lag))
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


def main_test_statistics(data_name: str, tag: str):
    models = [
        ("ordinary_hmm", 0),
        ("ar_hmm", 1),
        ("ar_2_hmm", 2),
        ("second_order_hmm", 1),
        ("ar_2_second_order_hmm", 2),
        ("covariate_hmm", 0),
        ("ar_1_covariate_hmm", 1),
        ("ar_2_covariate_hmm", 2),
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
        ("covariate_hmm", "ar_1_covariate_hmm", 1),
        ("ar_1_covariate_hmm", "ar_2_covariate_hmm", 2),
        ("ar_hmm", "ar_1_covariate_hmm", 1),  # AR(1) HMM nests AR(1) Covariate HMM
        ("ar_2_hmm", "ar_2_covariate_hmm", 2),  # AR(2) HMM nests AR(2) Covariate HMM
    ]

    ys, _ = load_train_data(data_name=data_name, tag=tag)
    num_samples = len(ys)

    stats_df = build_model_stats_df(models, data_name, tag, num_samples)
    lrt_df = build_lrt_comparison_df(edges, data_name, tag, num_samples)

    print(stats_df.to_string(index=False))
    print()
    print(lrt_df.to_string(index=False))

    out_dir = _results_dir(data_name, tag)
    os.makedirs(out_dir, exist_ok=True)
    stats_df.to_csv(os.path.join(out_dir, "model_stats.csv"), index=False)
    lrt_df.to_csv(os.path.join(out_dir, "lrt_comparison.csv"), index=False)
    print(f"wrote CSVs to {out_dir}")


if __name__ == "__main__":
    DATA_NAME = "b1"
    TAG = "jans-split"
    main_test_statistics(data_name=DATA_NAME, tag=TAG)
