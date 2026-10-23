"""Fit, save and evaluate the per-model drivers for one room.

    uv run python -m drivers.models.run --room 9 --states 6                    # every model, in fit order
    uv run python -m drivers.models.run --room 9 --states 6 --sigma-lam 1e-4   # same, plus a ridge on σ²
    uv run python -m drivers.models.run --room 9 --states 6 --models hmm_1 ar1_hmm_1
"""
import jax
jax.config.update("jax_enable_x64", True)

import argparse
import importlib
from dataclasses import replace
from typing import Callable

from drivers.models.evaluate import evaluate
from drivers.models.utils import (COVARIATE_COLS, OBS_COLS, Config, Loss, add_run_args, load_data, parse_args,
                                  room_name, save_result)

# Seeds come before the models that load them (see README.md).
FIT_ORDER = ("hmm_1", "hmm_beta_1", "hmm_beta_1_shared", "ar1_hmm_1", "ar1_hmm_beta_1", "ar1_hmm_beta_1_shared")


def fit_model(name: str, slug: str, loss: Loss, fit: Callable, dynamic: bool, room: str, states: int):
    """Fit one model on the room's training split, pickle it and evaluate it on the validation split."""
    cfg = Config(room, OBS_COLS, COVARIATE_COLS, states, loss=loss)
    data = load_data(cfg)
    result = fit(cfg, data)
    save_result(result, cfg, slug, data if dynamic else data.without_covariates())
    return evaluate(name, slug, result.params, cfg, data, load_data(cfg, "val"))


def main(name: str, slug: str, loss: Loss, fit: Callable, dynamic: bool, default_states: int):
    """Entry point of a single driver: `uv run python -m drivers.models.<slug> --room 9 --states K [--sigma-lam λ]`."""
    args = parse_args(default_states)
    fit_model(name, slug, replace(loss, sigma_lam=args.sigma_lam), fit, dynamic, room_name(args.room), args.states)


def fit_driver(slug: str, room: str, states: int, sigma_lam: float = 0.0):
    driver = importlib.import_module(f"drivers.models.{slug}")
    loss = replace(driver.LOSS, sigma_lam=sigma_lam)
    return fit_model(driver.NAME, slug, loss, driver.fit, driver.DYNAMIC, room, states)


if __name__ == "__main__":
    parser = add_run_args(argparse.ArgumentParser())
    parser.add_argument("--models", nargs="+", choices=FIT_ORDER, help="subset to fit (default: all, in fit order)")
    args = parser.parse_args()

    for slug in FIT_ORDER:
        if args.models is None or slug in args.models:
            fit_driver(slug, room_name(args.room), args.states, args.sigma_lam)
