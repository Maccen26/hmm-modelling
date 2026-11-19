"""The replot diagnostics (validation forecasts, first week, training residuals) for the random_search fits.

Same plots as drivers.analysis.replot, written to plots/{room}/{tag}/random_search/states_{K}/ so the
per-model-driver plots in plots/{room}/{tag}/states_{K}/ are untouched. Metrics are not re-saved
(random_search already wrote val_metrics.csv).

    uv run python -m drivers.analysis.replot_random_search --room 15 --tag calibrated_anchor --states 3 4 5 6 7
"""
import jax
jax.config.update("jax_enable_x64", True)

import argparse
import os

import numpy as np

from drivers.models.evaluate import (PLOT_HORIZONS, _columns, metrics_table, plot_diagnostics, plot_first_week,
                                     plot_residual_diagnostics, pseudo_residuals, rolling_predictions, run_label)
from drivers.models.random_search import CHAIN, run_dir
from drivers.models.utils import PLOTS_DIR, Config, load_data, room_name
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

    plot_dir = os.path.join(PLOTS_DIR, cfg.room, cfg.tag, "random_search", f"states_{cfg.states}")
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
    parser.add_argument("--tag", required=True)
    parser.add_argument("--states", "-K", type=int, nargs="+", default=[3, 4, 5, 6, 7])
    args = parser.parse_args()
    for states in args.states:
        for slug, name, *_ in CHAIN:
            replot(slug, name, room_name(args.room), states, args.tag)
