"""Extract per-model emission parameters and write them to CSV.

For a given ``data_name`` / ``tag`` this loads every fitted model from
``results/models/{data_name}/{tag}`` and, for each one, extracts the estimated
Gaussian emission parameters -- state means ``mu``, standard deviations
``sigma`` and (for the autoregressive models) the AR coefficients ``phi`` -- and
writes them to ``results/emission_params/{data_name}/{tag}/{model}.csv``.

Each CSV row is one emission state. Columns:

* ``order``   -- Markov order (1 or 2); constant per file.
* ``state``   -- flat emission-state index (1-based).
* ``s_prev``  -- previous state for second-order models (blank for first-order).
* ``s_curr``  -- current state.
* ``mu``, ``sigma`` -- Gaussian mean and standard deviation.
* ``phi_1``, ``phi_2``, ... -- AR coefficients (only for autoregressive models).

LaTeX rendering lives in ``drivers.latex``.

Run with ``python -m drivers.emission_params``.
"""

import os

import jax.numpy as jnp
import numpy as np
import pandas as pd

from drivers.utils import load_model

# Long-enough dummy observation series so AutoregressiveGaussEmission.mu can take
# its `k`-length lag slice at t=0 without going out of bounds. The returned means
# are the base state means (the AR term is dropped while t < k).
_DUMMY_YS = jnp.zeros(16)

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
    return f"results/emission_params/{data_name}/{tag}"


# --------------------------------------------------------------------------- #
# Parameter extraction
# --------------------------------------------------------------------------- #
def extract_emission_params(model):
    """Return (mu, sigma, phi) as numpy arrays for a fitted model.

    ``mu`` and ``sigma`` have shape (num_emission_states,). ``phi`` has shape
    (num_lags, num_emission_states) for autoregressive emissions, else None.
    """
    emission = model.emission
    mu = np.asarray(emission.mu(0, _DUMMY_YS, None)).ravel()
    sigma = np.asarray(emission.sigma(0, _DUMMY_YS, None)).ravel()
    phi = np.asarray(emission.phi()) if hasattr(emission, "phi") else None
    return mu, sigma, phi


def _num_lags(phi):
    return 0 if phi is None else phi.shape[0]


def _base_states(model, mu):
    """Number of base states and Markov order (order > 1 => second-order layout)."""
    order = int(getattr(model.transition, "order", 1))
    num_augmented = mu.shape[0]
    base = int(round(num_augmented ** (1.0 / order))) if order > 1 else num_augmented
    return base, order


def build_emission_df(model):
    """One row per emission state; see module docstring for columns."""
    mu, sigma, phi = extract_emission_params(model)
    base, order = _base_states(model, mu)
    num_lags = _num_lags(phi)

    rows = []
    for idx in range(mu.shape[0]):
        if order > 1:
            # Flat emission index is (s_prev - 1) * base + (s_curr - 1).
            s_prev = idx // base + 1
            s_curr = idx % base + 1
        else:
            s_prev = ""
            s_curr = idx + 1
        row = {
            "order": order,
            "state": idx + 1,
            "s_prev": s_prev,
            "s_curr": s_curr,
            "mu": float(mu[idx]),
            "sigma": float(sigma[idx]),
        }
        for lag in range(num_lags):
            row[f"phi_{lag + 1}"] = float(phi[lag, idx])
        rows.append(row)
    return pd.DataFrame(rows)


def main_emission_params(data_name: str, tag: str):
    out_dir = _results_dir(data_name, tag)
    os.makedirs(out_dir, exist_ok=True)
    for model_name in MODEL_NAMES:
        path = _model_path(data_name, tag, model_name)
        if not os.path.exists(path):
            print(f"skipping {model_name}: {path} not found")
            continue
        model = load_model(path)
        df = build_emission_df(model)
        out_path = os.path.join(out_dir, f"{model_name}.csv")
        df.to_csv(out_path, index=False)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    DATA_NAME = "b1"
    TAG = "jans-split"
    main_emission_params(data_name=DATA_NAME, tag=TAG)
