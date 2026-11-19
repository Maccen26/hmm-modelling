"""Out-of-sample evaluation of saved fits on the test split, like `evaluate` does for the validation split.

The fitted parameters are kept fixed. The filter runs through the training and then the validation
split, so the test forecasts start from the state the model would be in online at the start of the
test period; nothing is refitted.

    uv run python -m drivers.models.evaluate_test ar1_hmm_beta_1 --room 15 --states 3 4 5 --tag calibrated_anchor
"""
import jax
jax.config.update("jax_enable_x64", True)

import argparse
import importlib
import os

import jax.numpy as jnp
import numpy as np
import pandas as pd

from drivers.models.evaluate import (_columns, metrics_table, plot_diagnostics, plot_first_week, pseudo_residuals,
                                     rolling_predictions, run_label, save_metrics)
from drivers.models.utils import HORIZONS, PLOT_HORIZONS, PLOTS_DIR, TAG, Data, fit_path, load_data, room_name
from drivers.utils import load_model

METRICS_FILE = "test_metrics.csv"


def history(train: Data, val: Data) -> Data:
    """Train followed by validation: everything the model has seen before the test split starts."""
    xs = None if train.xs is None or val.xs is None else jnp.concatenate([train.xs, val.xs])
    return Data(ys=jnp.concatenate([train.ys, val.ys]), xs=xs, init_dist=train.init_dist,
                datetime=train.datetime.append(val.datetime))


def evaluate_test(slug: str, room: str, states: int, tag: str = TAG, horizons=HORIZONS) -> pd.DataFrame:
    """Rolling k-step forecasts of the test split from a saved fit: save R2/RMSE/MAPE to CSV and the plots."""
    path = fit_path(room, states, slug, tag)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved fit at {path}; fit it with `uv run python -m drivers.models.run` first.")
    saved = load_model(path)
    params, cfg, train = saved["result"].params, saved["config"], saved["data"]
    name = importlib.import_module(f"drivers.models.{slug}").NAME

    seen, test = history(train, load_data(cfg, "val")), load_data(cfg, "test")
    y_test = _columns(np.asarray(test.ys))
    preds = rolling_predictions(params, seen, test, horizons)
    _, res = pseudo_residuals(name, params, seen, test)

    table = metrics_table(name, slug, cfg, y_test, preds).assign(n_test=len(y_test)).drop(columns="n_val")
    print(table.to_string(index=False))
    print(f"Saved metrics to {save_metrics(table, cfg, METRICS_FILE)}")

    plot_dir = cfg.run_dir(PLOTS_DIR)
    label = run_label(name, cfg)
    shown = {k: p for k, p in preds.items() if k in PLOT_HORIZONS}
    for d, col in enumerate(cfg.obs_cols):
        file = f"{slug}_test.png" if not cfg.multivariate else f"{slug}_{col}_test.png"
        plot_diagnostics(f"{label} [{col}]", y_test[:, d], {k: p[:, d] for k, p in shown.items()},
                         res[:, d], os.path.join(plot_dir, file), split="test")
        plot_first_week(label, col, test.datetime, y_test[:, d], {k: p[:, d] for k, p in shown.items()},
                        table, os.path.join(plot_dir, file.replace(".png", "_first_week.png")), split="test")
    print(f"Saved plots to {plot_dir}")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", help="slugs of saved fits, e.g. ar1_hmm_beta_1")
    parser.add_argument("--room", "-r", type=int, required=True, help="room number, e.g. 15 for room_015")
    parser.add_argument("--states", "-K", type=int, nargs="+", required=True, help="number(s) of hidden states")
    parser.add_argument("--tag", default=TAG, help=f"data tag (default {TAG})")
    args = parser.parse_args()

    for slug in args.models:
        for states in args.states:
            evaluate_test(slug, room_name(args.room), states, args.tag)
