"""Extract per-model transition matrices and write them to CSV.

For a given ``data_name`` / ``tag`` this loads every fitted model from
``results/models/{data_name}/{tag}`` and writes the estimated transition matrix
to ``results/transition_params/{data_name}/{tag}/{model}.csv`` in long format.

Each row is one (from, to) entry. Columns:

* ``order``    -- Markov order (1 or 2); constant per file.
* ``k``        -- number of base states.
* ``from_idx`` -- source (augmented) state index, 0-based.
* ``to_idx``   -- target (augmented) state index, 0-based.
* ``prob``     -- transition probability.

For covariate (dynamic) transitions the covariate-free baseline matrix is
stored. For second-order models the full K^2 x K^2 augmented matrix is stored;
``drivers.latex`` reconstructs the reachable next-state layout.

Run with ``python -m drivers.transition_params``.
"""

import os

import numpy as np
import pandas as pd

from drivers.utils import load_model

MODEL_NAMES = [
    "ordinary_hmm",
    "ar_hmm",
    "ar_2_hmm",
    "second_order_hmm",
    "ar_2_second_order_hmm",
    "covariate_hmm",
    "ar_1_covariate_hmm",
    "ar_2_covariate_hmm",
]


def _model_path(data_name: str, tag: str, model_name: str) -> str:
    return f"results/models/{data_name}/{tag}/{model_name}.pkl"


def _results_dir(data_name: str, tag: str) -> str:
    return f"results/transition_params/{data_name}/{tag}"


# --------------------------------------------------------------------------- #
# Transition-matrix extraction
# --------------------------------------------------------------------------- #
def extract_transition_matrix(model):
    """Return (Gamma, order) for a fitted model.

    ``Gamma`` is the estimated transition matrix as a numpy array. For covariate
    (dynamic) transitions the covariate-free baseline matrix is returned. ``order``
    is the Markov order (order > 1 => second-order layout).
    """
    transition = model.transition
    order = int(getattr(transition, "order", 1))
    if hasattr(transition, "base_transition_matrix"):
        gamma = transition.base_transition_matrix()
    else:
        gamma = transition.transition_matrix()
    return np.asarray(gamma), order


def build_transition_df(model):
    """Long-format transition matrix; see module docstring for columns."""
    gamma, order = extract_transition_matrix(model)
    n = gamma.shape[0]
    k = int(round(n ** (1.0 / order))) if order > 1 else n

    rows = []
    for i in range(n):
        for j in range(n):
            rows.append({
                "order": order,
                "k": k,
                "from_idx": i,
                "to_idx": j,
                "prob": float(gamma[i, j]),
            })
    return pd.DataFrame(rows)


def main_transition_params(data_name: str, tag: str):
    out_dir = _results_dir(data_name, tag)
    os.makedirs(out_dir, exist_ok=True)
    for model_name in MODEL_NAMES:
        path = _model_path(data_name, tag, model_name)
        if not os.path.exists(path):
            print(f"skipping {model_name}: {path} not found")
            continue
        model = load_model(path)
        df = build_transition_df(model)
        out_path = os.path.join(out_dir, f"{model_name}.csv")
        df.to_csv(out_path, index=False)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    DATA_NAME = "b1"
    TAG = "jans-split"
    main_transition_params(data_name=DATA_NAME, tag=TAG)
