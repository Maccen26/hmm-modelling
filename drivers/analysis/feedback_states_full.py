"""feedback_states over the whole series (train + val + test) for a random_search fit.

Forward filtering over the whole series, then BackwardSampler draws state paths; for each path the feedback
events are counted per state. The calibrated tags carry no feedback column, so feedback is re-binned from the
raw feedback file onto the series' timestamps (drivers.data.clean_dtu_data.add_feedback).
One plot for the train split and one for the whole series (the train plot slices the whole-series paths)
in plots/{room}/{tag}/random_search/states_{K}/, table in results/{room}/{tag}/random_search/states_{K}/.

    uv run python -m drivers.analysis.feedback_states_full ar1_hmm_beta_1 --room 15 -K 3 --tag calibrated_anchor
"""
import jax
jax.config.update("jax_enable_x64", True)

import argparse
import os

import jax.numpy as jnp
import numpy as np
import pandas as pd

from drivers.analysis.feedback_states import plot, state_counts
from drivers.data.clean_dtu_data import add_feedback
from drivers.models.evaluate import _covariates
from drivers.models.random_search import emission_summary, run_dir
from drivers.models.utils import PLOTS_DIR, Config, load_data, room_name
from drivers.utils import load_model
from src.api.v5 import BackwardSampler, ForwardAlgorithm


def feedback_on(dt: pd.DatetimeIndex, room: str) -> jnp.ndarray:
    raw_room = room.replace("_", " ").title()  # room_015 -> Room 015, the raw feedback file name
    return jnp.asarray(add_feedback(pd.DataFrame(index=dt), raw_room)["feedback"].to_numpy(), dtype=float)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--room", "-r", type=int, required=True)
    parser.add_argument("--states", "-K", type=int, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--paths", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = run_dir(Config(room_name(args.room), (), (), args.states, tag=args.tag))
    saved = load_model(os.path.join(out, f"{args.model}.pkl"))
    params, cfg = saved["result"].params, saved["config"]
    parts = [saved["data"], load_data(cfg, "val"), load_data(cfg, "test")]

    ys = jnp.concatenate([p.ys for p in parts])
    covs = [_covariates(params, p) for p in parts]
    xs = None if covs[0] is None else jnp.concatenate(covs)
    feedback = feedback_on(parts[0].datetime.append([p.datetime for p in parts[1:]]), cfg.room)
    n_feedback = int(feedback.sum())

    utt, _, _ = ForwardAlgorithm.run(params=params, initial_dist=parts[0].init_dist, ys=ys, xs=xs)
    Gammas = params.transition_matrices(N=len(ys), xs=xs)
    paths = BackwardSampler.run(args.paths, Gammas, utt, seed=args.seed)   # (P, T)

    mu, sigma, phi = emission_summary(params)
    tables = []
    for split, end, where in (("train", len(parts[0].ys), "train split"), ("all", len(ys), "whole series")):
        fb, n = feedback[:end], int(feedback[:end].sum())   # paths come from the whole series, sliced to the split
        per_path = state_counts(paths[:, :end], fb, cfg.states) / n
        time_share = state_counts(paths[:, :end], jnp.ones_like(fb), cfg.states).mean(axis=0) / end
        lo, hi = np.percentile(per_path, [2.5, 97.5], axis=0)
        # mean observed CO2 while in each state, per sampled path (NaN if a path never visits the state)
        steps = state_counts(paths[:, :end], jnp.ones_like(fb), cfg.states)
        co2 = state_counts(paths[:, :end], ys[:end], cfg.states) / np.where(steps > 0, steps, np.nan)
        co2_lo, co2_hi = np.nanpercentile(co2, [2.5, 97.5], axis=0)
        # mean previous observation y_{t-1} and mean realised change y_t - y_{t-1} while in each state (t >= 1)
        lag_steps = state_counts(paths[:, 1:end], jnp.ones(end - 1), cfg.states)
        lag_steps = np.where(lag_steps > 0, lag_steps, np.nan)
        lag = np.nanmean(state_counts(paths[:, 1:end], ys[:end - 1], cfg.states) / lag_steps, axis=0)
        change = np.nanmean(state_counts(paths[:, 1:end], jnp.diff(ys[:end]), cfg.states) / lag_steps, axis=0)
        table = pd.DataFrame({"split": split, "mu": mu.round(1), "sigma": sigma.round(1), "phi": phi.round(3),
                              "co2_mean": np.nanmean(co2, axis=0), "co2_lo_2.5": co2_lo, "co2_hi_97.5": co2_hi,
                              "lag_mean": lag, "drift": (1 - phi) * (mu - lag), "change_mean": change,
                              "share": per_path.mean(axis=0), "lo_2.5": lo, "hi_97.5": hi, "time_share": time_share},
                             index=pd.Index(range(cfg.states), name="state"))
        print(f"\n{split}: {n} feedback events in {end} steps ({args.paths} sampled paths)")
        print(table.drop(columns="split").round(3).to_string())
        tables.append(table)

        path = os.path.join(PLOTS_DIR, cfg.room, cfg.tag, "random_search", f"states_{cfg.states}",
                            f"{args.model}_feedback_states_{split}.png")
        plot(f"{args.model} — {cfg.room}, K={cfg.states}, loss={cfg.loss.slug}\n"
             f"States of the {n} feedback events, {where}", per_path, time_share, n, path)
        print(f"Saved {path}")
    pd.concat(tables).to_csv(os.path.join(out, f"{args.model}_feedback_states.csv"))
