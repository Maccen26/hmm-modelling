"""LaTeX table of the test-set forecast metrics of chosen (model, K) fits across forecast horizons.

Reads `states_{K}/test_metrics.csv` under one results folder and writes one row per (model, K) and one
`RMSE / MAPE` column per horizon to `Thesis/Tables/<name>.tex`. Rows are given as `slug:K`.

    uv run python -m drivers.analysis.test_metrics_table results/room_015/calibrated_anchor \
        --rows hmm_1:4 hmm_beta_1:4 ar1_hmm_1:4 ar1_hmm_beta_1:3 ar1_hmm_beta_1:4
"""
import argparse
import os

import pandas as pd

from drivers.analysis.predictive_metrics_across_states_table import default_name, load_metrics
from drivers.utils import save_thesis_table


def parse_rows(rows: list[str]) -> list[tuple[str, int]]:
    """`["hmm_1:4", ...]` -> `[("hmm_1", 4), ...]`."""
    return [(slug, int(k)) for slug, k in (r.split(":") for r in rows)]


def metrics_table(df: pd.DataFrame, rows: list[tuple[str, int]]) -> pd.DataFrame:
    """Rows are (model, K) in the given order, columns are horizons, cells are `RMSE / MAPE`."""
    out = []
    for slug, k in rows:
        sub = df[(df["slug"] == slug) & (df["K"] == k)].sort_values("horizon")
        if sub.empty:
            raise SystemExit(f"No test metrics for {slug} with K={k}")
        cells = {f"$h={h}$": f"{r:.2f} / {m:.2f}" for h, r, m in zip(sub["horizon"], sub["RMSE"], sub["MAPE"])}
        out.append({"Model": sub["model"].iloc[0].replace("b", "$\\beta$"), "States": k, **cells})
    return pd.DataFrame(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_dir", help="folder holding the states_{K}/ subfolders")
    parser.add_argument("--rows", nargs="+", required=True, help="fits to show, as slug:K")
    parser.add_argument("--name", help="table file name in Thesis/Tables/ (default: from the folder)")
    args = parser.parse_args()

    table = metrics_table(load_metrics(args.results_dir, "test"), parse_rows(args.rows))
    print(table.to_string(index=False))

    name = args.name or default_name(args.results_dir, "test", 0).removesuffix("_test_h0").replace(
        "predictive_metrics", "test_metrics")
    path = save_thesis_table(
        table,
        name=name,
        caption="Test RMSE (ppm) / MAPE (\\%) of the $h$-steps-ahead forecasts (30-minute steps) for the nested "
                "models with $K=4$ states and the AR(1)-HMM($\\beta$,1) with $K=3$ and $K=4$.",
        label=f"tab:{name}",
        max_width=True,
    )
    print(f"wrote {path}")
