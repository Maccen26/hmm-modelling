"""Diagnostic plots (validation forecasts, first week, training residuals) of the saved random_search fits.

Written to plots/{room}/{tag}/states_{K}/. Metrics are not re-saved (random_search already wrote
val_metrics.csv).

    uv run python -m drivers.analysis.replot --room 15 [--tag calibrated_anchor] [--states 3 4 5 6 7]
"""
import jax
jax.config.update("jax_enable_x64", True)

import argparse
import os

import numpy as np

from drivers.models.evaluate import (_columns, metrics_table, plot_diagnostics, plot_first_week,
                                     plot_residual_diagnostics, pseudo_residuals, rolling_predictions, run_label)
from drivers.models.random_search import CHAIN, run_dir
from drivers.models.utils import PLOT_HORIZONS, PLOTS_DIR, TAG, Config, load_data, room_name
from drivers.utils import load_model


def replot(slug: str, name: str, room: str, states: int, tag: str):
    saved = load_model(os.path.join(run_dir(Config(room, (), (), states, tag=tag)), f"{slug}.pkl"))
    params, cfg, train = saved["result"].params, saved["config"], saved["data"]
    val = load_data(cfg, "val")
    name = f"{name} [{saved['init']}]"
    print(f"--- {name} ({cfg.room}, K={cfg.states})")

    y_val = _columns(np.asarray(val.ys))
    preds = rolling_predictions(params, train, val)
    res_train, res = pseudo_residuals(name, params, train, val)
    table = metrics_table(name, slug, cfg, y_val, preds)

    plot_dir = cfg.run_dir(PLOTS_DIR)
    label = run_label(name, cfg)
    shown = {k: p for k, p in preds.items() if k in PLOT_HORIZONS}
    for d, col in enumerate(cfg.obs_cols):
        file = os.path.join(plot_dir, f"{slug}.png")
        plot_diagnostics(f"{label} [{col}]", y_val[:, d], {k: p[:, d] for k, p in shown.items()}, res[:, d], file)
        plot_residual_diagnostics(f"{label} [{col}]", res_train[:, d], "training",
                                  file.replace(".png", "_train_residuals.png"))
        plot_first_week(label, col, val.datetime, y_val[:, d], {k: p[:, d] for k, p in shown.items()},
                        table, file.replace(".png", "_first_week.png"))
    print(f"Saved plots to {plot_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--room", "-r", type=int, required=True)
    parser.add_argument("--tag", default=TAG, help=f"data tag (default {TAG})")
    parser.add_argument("--states", "-K", type=int, nargs="+", default=[3, 4, 5, 6, 7])
    args = parser.parse_args()
    for states in args.states:
        for slug, name, *_ in CHAIN:
            replot(slug, name, room_name(args.room), states, args.tag)
