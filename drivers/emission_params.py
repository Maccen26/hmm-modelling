"""Build per-model emission-parameter tables and write them to LaTeX.

Loads every fitted model from ``results/models`` and, for each one, extracts the
estimated Gaussian emission parameters -- state means ``mu``, standard deviations
``sigma`` and (for the autoregressive models) the AR coefficients ``phi`` -- and
writes a ``\\begin{table}`` block to
``report/model_results/<model>/emission_params.tex``.

Two layouts are produced, mirroring the model structure:

* First-order models (4 emission states): one row per state.
* Second-order models (K^2 augmented states): one row per state pair
  ``(s_{t-1}, s_t)``, grouped by the current state ``s_t``.

Run with ``python -m drivers.emission_params``.
"""

import os

import jax.numpy as jnp
import numpy as np

from drivers.utils import load_model

# Long-enough dummy observation series so AutoregressiveGaussEmission.mu can take
# its `k`-length lag slice at t=0 without going out of bounds. The returned means
# are the base state means (the AR term is dropped while t < k).
_DUMMY_YS = jnp.zeros(16)

# Per-model caption and label. Keeping these explicit keeps the generated tables
# readable and matches the wording used elsewhere in the report.
MODEL_META = {
    "ordinary_hmm": (
        "Estimated emission parameters (Gaussian) for the ordinary 4-state HMM.",
        "tab:ordinary_hmm_emissions",
    ),
    "ar_hmm": (
        "Estimated emission parameters for the AR-HMM (Gaussian with AR(1) residuals).",
        "tab:ar_hmm_emissions",
    ),
    "ar_2_hmm": (
        "Estimated emission parameters for the AR(2)-HMM (Gaussian with AR(2) residuals).",
        "tab:ar_2_hmm_emissions",
    ),
    "second_order_hmm": (
        "Estimated emission parameters for the AR(1) second-order HMM. Each row is a "
        "state pair $(s_{t-1}, s_t)$; $\\hat{\\mu}$ and $\\hat{\\sigma}$ are the Gaussian "
        "mean and standard deviation of the residual and $\\hat{\\phi}$ is the AR(1) "
        "coefficient.",
        "tab:second_order_hmm_emissions",
    ),
    "ar_2_second_order_hmm": (
        "Estimated emission parameters for the AR(2) second-order HMM. $\\hat{\\sigma}$ "
        "and the AR(2) coefficients $\\hat{\\phi}_1$, $\\hat{\\phi}_2$ depend on "
        "$(s_{t-1}, s_t)$. A sum $\\hat{\\phi}_1 + \\hat{\\phi}_2 > 1$ indicates a "
        "non-stationary AR process for that state pair.",
        "tab:ar_2_second_order_hmm_emissions",
    ),
    "covariate_hmm": (
        "Estimated emission parameters (Gaussian) for the covariate HMM.",
        "tab:covariate_hmm_emissions",
    ),
    "ar_1_covariate_hmm": (
        "Estimated emission parameters for the AR(1) covariate HMM.",
        "tab:ar_1_covariate_hmm_emissions",
    ),
    "ar_2_covariate_hmm": (
        "Estimated emission parameters for the AR(2) covariate HMM.",
        "tab:ar_2_covariate_hmm_emissions",
    ),
}

MODEL_NAMES = list(MODEL_META)


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


# --------------------------------------------------------------------------- #
# LaTeX rendering
# --------------------------------------------------------------------------- #
def _fmt(value, dp):
    """Format a float, wrapping negatives in math mode (e.g. $-0.0781$)."""
    s = f"{abs(value):.{dp}f}"
    return f"$-{s}$" if value < 0 else s


def _phi_headers(num_lags, compact=False):
    """AR-coefficient column headers.

    ``compact`` drops the "AR coeff." prefix, matching the denser second-order
    tables where the state-pair columns already carry the context.
    """
    if num_lags == 0:
        return []
    if num_lags == 1:
        return ["$\\hat{\\phi}$"] if compact else ["AR coeff.\\ ($\\hat{\\phi}$)"]
    if compact:
        heads = [f"$\\hat{{\\phi}}_{i}$" for i in range(1, num_lags + 1)]
    else:
        heads = [f"AR coeff.\\ ($\\hat{{\\phi}}_{i}$)" for i in range(1, num_lags + 1)]
    heads.append("$\\hat{\\phi}_1 + \\hat{\\phi}_2$")
    return heads


def _phi_cells(phi, flat_idx, num_lags):
    if num_lags == 0:
        return []
    cells = [_fmt(float(phi[lag, flat_idx]), 4) for lag in range(num_lags)]
    if num_lags > 1:
        cells.append(_fmt(float(phi[:, flat_idx].sum()), 4))
    return cells


def _table(col_spec, header_cols, body_rows, caption, label, placement="ht"):
    header = " & ".join(header_cols) + " \\\\"
    body = "\n".join(body_rows)
    return (
        f"\\begin{{table}}[{placement}]\n"
        "\\centering\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\hline\n"
        f"{header}\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )


def _first_order_table(mu, sigma, phi, caption, label):
    num_lags = _num_lags(phi)
    header = ["State", "Mean ($\\hat{\\mu}$)", "Std.\\ Dev.\\ ($\\hat{\\sigma}$)"] + _phi_headers(num_lags)
    col_spec = "l" + "c" * (len(header) - 1)
    rows = []
    for i in range(mu.shape[0]):
        cells = [f"State {i + 1}", _fmt(float(mu[i]), 3), _fmt(float(sigma[i]), 3)]
        cells += _phi_cells(phi, i, num_lags)
        rows.append(" & ".join(cells) + " \\\\")
    return _table(col_spec, header, rows, caption, label, placement="ht")


def _second_order_table(mu, sigma, phi, base, caption, label):
    num_lags = _num_lags(phi)
    header = ["$s_{t-1}$", "$s_t$", "$\\hat{\\mu}$", "$\\hat{\\sigma}$"] + _phi_headers(num_lags, compact=True)
    col_spec = "c" * len(header)
    rows = []
    # Group by current state s_t (outer), previous state s_{t-1} (inner). The flat
    # emission index is (s_prev - 1) * base + (s_curr - 1).
    for s_curr in range(1, base + 1):
        for s_prev in range(1, base + 1):
            idx = (s_prev - 1) * base + (s_curr - 1)
            cells = [str(s_prev), str(s_curr), _fmt(float(mu[idx]), 3), _fmt(float(sigma[idx]), 3)]
            cells += _phi_cells(phi, idx, num_lags)
            rows.append(" & ".join(cells) + " \\\\")
        rows.append("\\hline")
    rows = rows[:-1]  # drop trailing separator; _table adds the closing \hline
    return _table(col_spec, header, rows, caption, label, placement="H")


def build_emission_table_tex(model_name, model):
    caption, label = MODEL_META[model_name]
    mu, sigma, phi = extract_emission_params(model)
    base, order = _base_states(model, mu)
    if order > 1:
        return _second_order_table(mu, sigma, phi, base, caption, label)
    return _first_order_table(mu, sigma, phi, caption, label)


# --------------------------------------------------------------------------- #
def main_emission_params():
    for model_name in MODEL_NAMES:
        model = load_model(f"results/models/{model_name}.pkl")
        tex = build_emission_table_tex(model_name, model)
        out_path = f"report/model_results/{model_name}/emission_params.tex"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write(tex)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main_emission_params()
