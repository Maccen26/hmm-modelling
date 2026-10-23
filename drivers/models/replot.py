"""Re-evaluate saved fits on the validation split (plots + metrics CSV) without refitting.

    uv run python -m drivers.models.replot                       # every saved model
    uv run python -m drivers.models.replot --room room_009 --tag 30min --states 6 --models ar1_hmm_beta_1
"""
import jax
jax.config.update("jax_enable_x64", True)

import argparse
import glob
import importlib
import os

from drivers.models.evaluate import evaluate
from drivers.models.utils import RESULTS_DIR, load_data
from drivers.utils import load_model


def saved_fits(room: str, tag: str, states: str, models: list[str] | None) -> list[str]:
    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, room, tag, f"states_{states}", "*.pkl")))
    return [p for p in paths if models is None or os.path.basename(p)[:-4] in models]


def replot(path: str):
    slug = os.path.basename(path)[:-4]
    name = importlib.import_module(f"drivers.models.{slug}").NAME
    saved = load_model(path)
    cfg = saved["config"]
    print(f"--- {name} ({cfg.room}/{cfg.tag}, K={cfg.states})")
    evaluate(name, slug, saved["result"].params, cfg, saved["data"], load_data(cfg, "val"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--room", default="*")
    parser.add_argument("--tag", default="*")
    parser.add_argument("--states", "-K", default="*")
    parser.add_argument("--models", nargs="+", help="slugs to replot (default: all saved)")
    args = parser.parse_args()

    paths = saved_fits(args.room, args.tag, args.states, args.models)
    if not paths:
        raise SystemExit(f"No saved fits match under {RESULTS_DIR}")
    for path in paths:
        replot(path)
