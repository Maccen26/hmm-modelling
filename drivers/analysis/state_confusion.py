"""Confusion matrix of the H-step predicted state against the smoothed (forward-backward) state.

For every point t of the split, the predicted state is the argmax of P(z_t | y_1:t-H) from a rolling
H-step forecast, and the inferred state is the argmax of P(z_t | y_1:T) from ForwardBackwardAlgorithm.
The split is filtered through the training set first.

    uv run python -m drivers.analysis.state_confusion ar1_hmm_beta_1 --room 9 -K 6 --horizon 4
"""
import jax
jax.config.update("jax_enable_x64", True)

import argparse
import os

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from drivers.models.evaluate import _covariates
from drivers.models.utils import PLOTS_DIR, Data, fit_path, load_data, room_name
from drivers.utils import load_model
from src.api.v5 import Params
from src.api.v5.analysis import Predictor
from src.api.v5.inference.forward_backward import ForwardBackwardAlgorithm


def load_saved(model: str, room: int, states: int, loss: str | None):
    path = fit_path(room_name(room), states, model, loss)
    if not os.path.exists(path):
        raise SystemExit(f"No saved fit at {path}")
    saved = load_model(path)
    return saved["result"].params, saved["config"], saved["data"]


def smoothed_states(params: Params, train: Data, split: Data) -> np.ndarray:
    """(N_split,) argmax of P(z_t | y_1:T) over train+split, sliced to the split."""
    ys = jnp.concatenate([train.ys, split.ys])
    xs = None if _covariates(params, train) is None else jnp.concatenate([train.xs, split.xs])
    probs = ForwardBackwardAlgorithm.run(params=params, initial_dist=train.init_dist, ys=ys, xs=xs)
    return np.asarray(probs)[len(train.ys):].argmax(axis=1)


def predicted_states(params: Params, train: Data, split: Data, horizon: int) -> np.ndarray:
    """(N_split,) argmax of the H-step-ahead predicted state distribution of each split point."""
    pred = Predictor.rolling(params, train.ys, split.ys, xs=_covariates(params, train),
                             xs_test=_covariates(params, split), horizon=horizon, initial_dist=train.init_dist)
    return np.asarray(pred.state_probs).argmax(axis=1)


def confusion_matrix(inferred: np.ndarray, predicted: np.ndarray, num_states: int) -> np.ndarray:
    """(K, K) counts; rows are the inferred state, columns the predicted state."""
    cm = np.zeros((num_states, num_states), dtype=int)
    np.add.at(cm, (inferred, predicted), 1)
    return cm


def plot(title: str, cm: np.ndarray, labels: list[str], path: str):
    rows = cm.sum(axis=1, keepdims=True)
    share = np.divide(cm, rows, out=np.zeros_like(cm, dtype=float), where=rows > 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, f"{cm[i, j]}\n({share[i, j]:.0%})", ha="center", va="center", fontsize=8,
                color="white" if share[i, j] > 0.5 else "black")
    ticks = np.arange(len(labels))
    ax.set_xticks(ticks, labels)
    ax.set_yticks(ticks, labels)
    ax.set(xlabel="Predicted state (H steps ahead)", ylabel="Inferred state (forward-backward)")
    fig.colorbar(im, ax=ax, label="Share of row")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="slug of the saved fit, e.g. ar1_hmm_beta_1")
    parser.add_argument("--room", "-r", type=int, required=True, help="room number, e.g. 9 for room_009")
    parser.add_argument("--states", "-K", type=int, default=6)
    parser.add_argument("--horizon", "-H", type=int, default=1, help="forecast horizon in steps")
    parser.add_argument("--loss", help="loss folder, e.g. ridge_beta_lam1 (default: the driver's own loss)")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()

    params, cfg, train = load_saved(args.model, args.room, args.states, args.loss)
    split = load_data(cfg, args.split)

    inferred = smoothed_states(params, train, split)
    predicted = predicted_states(params, train, split, args.horizon)
    cm = confusion_matrix(inferred, predicted, cfg.states)
    accuracy = np.trace(cm) / cm.sum()
    print(f"Accuracy (H={args.horizon}, {args.split}, n={cm.sum()}): {accuracy:.3f}")

    out = os.path.join(cfg.run_dir(PLOTS_DIR), f"{args.model}_state_confusion_h{args.horizon}_{args.split}.png")
    plot(f"{args.model} — {cfg.room}, K={cfg.states}, loss={cfg.loss.slug}\n"
         f"{args.horizon}-step predicted vs smoothed state, {args.split} split: accuracy {accuracy:.1%}",
         cm, [f"s{s}" for s in range(cfg.states)], out)
    print(f"Saved {out}")
