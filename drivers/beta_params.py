"""Extract covariate-coefficient (beta) tensors and write them to CSV.

Only the covariate (dynamic-transition) models carry covariate coefficients. For
a given ``data_name`` / ``tag`` this loads each of those fitted models from
``results/models/{data_name}/{tag}`` and writes its ``beta`` tensor to
``results/beta_params/{data_name}/{tag}/{model}.csv``.

``beta`` has shape (num_inputs, num_states, num_states - 1): the effect of each
input covariate on each off-diagonal transition logit. Each CSV row is one
off-diagonal transition. Columns:

* ``from_state`` -- source state (1-based).
* ``to_state``   -- target state (1-based).
* ``beta_1``, ``beta_2``, ... -- coefficient per input covariate (for the
  2-input case these are the cos/sin time-of-day harmonics).

LaTeX rendering lives in ``drivers.latex``.

Run with ``python -m drivers.beta_params``.
"""

import os

import numpy as np
import pandas as pd

from drivers.utils import load_model

MODEL_NAMES = [
    "covariate_hmm",
    "ar_1_covariate_hmm",
    "ar_2_covariate_hmm",
]


def _model_path(data_name: str, tag: str, model_name: str) -> str:
    return f"results/models/{data_name}/{tag}/{model_name}.pkl"


def _results_dir(data_name: str, tag: str) -> str:
    return f"results/beta_params/{data_name}/{tag}"


def build_beta_df(model):
    """One row per off-diagonal transition; see module docstring for columns."""
    beta = np.asarray(model.transition.beta)  # (num_inputs, num_states, num_states - 1)
    num_inputs, num_states, _ = beta.shape

    rows = []
    for i in range(num_states):
        # Off-diagonal targets for from-state i, in the same increasing order the
        # logits use (states j != i).
        targets = [j for j in range(num_states) if j != i]
        for k, j in enumerate(targets):
            row = {"from_state": i + 1, "to_state": j + 1}
            for c in range(num_inputs):
                row[f"beta_{c + 1}"] = float(beta[c, i, k])
            rows.append(row)
    return pd.DataFrame(rows)


def main_beta_params(data_name: str, tag: str):
    out_dir = _results_dir(data_name, tag)
    os.makedirs(out_dir, exist_ok=True)
    for model_name in MODEL_NAMES:
        path = _model_path(data_name, tag, model_name)
        if not os.path.exists(path):
            print(f"skipping {model_name}: {path} not found")
            continue
        model = load_model(path)
        if not hasattr(model.transition, "beta"):
            print(f"skipping {model_name}: transition has no beta coefficients")
            continue
        df = build_beta_df(model)
        out_path = os.path.join(out_dir, f"{model_name}.csv")
        df.to_csv(out_path, index=False)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    DATA_NAME = "b1"
    TAG = "jans-split"
    main_beta_params(data_name=DATA_NAME, tag=TAG)
