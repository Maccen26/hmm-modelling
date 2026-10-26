"""Print the natural and working (unconstrained, optimised) parameters of a saved fit.

    uv run python -m drivers.models.show_params ar1_hmm_beta_1 --room 9 -K 6
    uv run python -m drivers.models.show_params hmm_1 --room 9 -K 4 --loss nll
"""
import jax
jax.config.update("jax_enable_x64", True)

import argparse
import os
from dataclasses import fields

import numpy as np

from drivers.models.utils import fit_path, room_name
from drivers.utils import load_model
from src.api.v5 import Params


def working(module) -> dict[str, np.ndarray]:
    """The module's array leaves, i.e. what the solver optimises."""
    return {f.name: np.asarray(getattr(module, f.name)) for f in fields(module)
            if hasattr(getattr(module, f.name), "shape")}


def natural(params: Params, ys) -> dict[str, np.ndarray]:
    em, tr = params.emission, params.transition
    mu = em.mu_vals(0, ys) if hasattr(em, "mu_vals") else em.mu(0, ys)  # AR: means before the AR correction
    out = {"mu": mu, "sigma": em.sigma(0, ys)}
    if hasattr(em, "phi"):
        out["phi"] = em.phi()
    if hasattr(tr, "base_transition_matrix"):
        out["Gamma (base, x = 0)"] = tr.base_transition_matrix()
        out["beta"] = tr.beta
    else:
        out["Gamma"] = tr.transition_matrix()
    return {k: np.asarray(v) for k, v in out.items()}


def show(title: str, values: dict[str, np.ndarray]):
    print(f"\n=== {title} ===")
    for name, v in values.items():
        print(f"{name}  {v.shape}:\n{v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="slug of the saved fit, e.g. ar1_hmm_beta_1")
    parser.add_argument("--room", "-r", type=int, required=True, help="room number, e.g. 9 for room_009")
    parser.add_argument("--states", "-K", type=int, default=6)
    parser.add_argument("--loss", help="loss folder, e.g. ridge_beta_lam1 (default: the driver's own loss)")
    args = parser.parse_args()

    path = fit_path(room_name(args.room), args.states, args.model, args.loss)
    if not os.path.exists(path):
        raise SystemExit(f"No saved fit at {path}")
    saved = load_model(path)
    params = saved["result"].params

    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    print(f"{args.model}: {type(params.emission).__name__} + {type(params.transition).__name__} "
          f"({saved['config'].room}, K={args.states}, loss={saved['config'].loss.slug}, obs={list(saved['config'].obs_cols)})")
    if "covariate_cols" in saved and hasattr(params.transition, "beta"):
        print(f"beta covariates (axis 0): {saved['covariate_cols']}")
    show("Natural parameters", natural(params, saved["data"].ys))
    show("Working parameters: emission", working(params.emission))
    show("Working parameters: transition", working(params.transition))
