"""Sample hidden-state paths of a saved fit given the room's whole observed CO2 series (train + val + test).

The forward filter runs over the whole series, then BackwardSampler draws state paths from
P(z_1:T | y_1:T) in batches. The paths are saved as uint8 together with the series they condition on,
so `state_analysis` can work from the file without refitting or resampling:

    paths       (P, T) sampled states, 0-based
    ys          (T,)   observed CO2
    datetime    (T,)   bin timestamps (datetime64[ns])
    off_day     (T,)   1 on weekends and Danish holidays (the model's off_day covariate)
    split_ends  (3,)   end index of train, val and test in the series

Written to results/{room}/{tag}/states_{K}/simulation/{model}_paths.npz.

    uv run python -m drivers.simulation.sample_paths ar1_hmm_beta_1 --room 15 -K 4
    uv run python -m drivers.simulation.sample_paths ar1_hmm_beta_1 --room 15 -K 4 --paths 1000   # smoke run
"""
import jax
jax.config.update("jax_enable_x64", True)

import argparse
import os

import jax.numpy as jnp
import numpy as np

from drivers.models.evaluate import _covariates
from drivers.models.utils import RESULTS_DIR, TAG, Data, covariates, fit_path, load_data, room_name
from drivers.utils import load_model
from src.api.v5 import BackwardSampler, ForwardAlgorithm, Params

SPLITS = ("train", "val", "test")


def paths_file(room: str, states: int, model: str, tag: str = TAG) -> str:
    return os.path.join(RESULTS_DIR, room, tag, f"states_{states}", "simulation", f"{model}_paths.npz")


def whole_series(cfg) -> tuple[Data, np.ndarray]:
    """Train, val and test joined into one Data, and the end index of each split."""
    parts = [load_data(cfg, split) for split in SPLITS]
    data = Data(ys=jnp.concatenate([p.ys for p in parts]), xs=jnp.concatenate([p.xs for p in parts]),
                init_dist=parts[0].init_dist, datetime=parts[0].datetime.append([p.datetime for p in parts[1:]]))
    return data, np.cumsum([len(p.ys) for p in parts])


def sample_paths(params: Params, data: Data, n_paths: int, batch: int, seed: int) -> np.ndarray:
    """(n_paths, T) uint8 paths, drawn `batch` at a time (one seed per batch) to bound memory."""
    xs = _covariates(params, data)
    utt, _, _ = ForwardAlgorithm.run(params=params, initial_dist=data.init_dist, ys=data.ys, xs=xs)
    Gammas = params.transition_matrices(N=len(data.ys), xs=xs)
    paths = np.empty((n_paths, len(data.ys)), dtype=np.uint8)
    for start in range(0, n_paths, batch):
        n = min(batch, n_paths - start)
        paths[start:start + n] = np.asarray(BackwardSampler.run(batch, Gammas, utt, seed=seed + start))[:n]
        print(f"  sampled {start + n}/{n_paths} paths")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", nargs="?", default="ar1_hmm_beta_1", help="slug of the saved fit")
    parser.add_argument("--room", "-r", type=int, required=True, help="room number, e.g. 15 for room_015")
    parser.add_argument("--states", "-K", type=int, required=True)
    parser.add_argument("--tag", default=TAG, help=f"data tag (default {TAG})")
    parser.add_argument("--paths", type=int, default=100_000, help="number of state paths to sample")
    parser.add_argument("--batch", type=int, default=5_000, help="paths per BackwardSampler call")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    room = room_name(args.room)
    saved = load_model(fit_path(room, args.states, args.model, args.tag))
    params, cfg = saved["result"].params, saved["config"]
    data, split_ends = whole_series(cfg)
    print(f"{args.model} — {room}, K={args.states}: sampling {args.paths} paths over {len(data.ys)} bins")

    paths = sample_paths(params, data, args.paths, min(args.batch, args.paths), args.seed)
    out = paths_file(room, args.states, args.model, args.tag)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, paths=paths, ys=np.asarray(data.ys), datetime=data.datetime.to_numpy(),
                        off_day=np.asarray(covariates(data.datetime, ("off_day",)))[:, 0],
                        split_ends=split_ends)
    print(f"Saved {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
