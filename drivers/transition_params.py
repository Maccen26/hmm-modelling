"""Build per-model transition tables and write them to LaTeX.

Loads every fitted model from ``results/models`` and, for each one, extracts the
estimated transition probabilities and writes a ``\\begin{table}`` block to
``report/model_results/<model>/transition_matrix.tex``.

Three layouts are produced, matching the transition structure:

* First-order static models: the full K x K transition matrix.
* Second-order models: for each augmented state $(s_{t-1}, s_t)$ the next-state
  distribution $P(s_{t+1} \\mid s_{t-1}, s_t)$ (a K^2 x K table), grouped by the
  previous state. Probabilities that round to zero are shown as a dash.
* Covariate (dynamic) models: the covariate-free baseline matrix
  ``base_transition_matrix()``; the time-of-day covariates modulate these
  probabilities at each step.

Run with ``python -m drivers.transition_params``.
"""

import os

import numpy as np

from drivers.utils import load_model

MODEL_META = {
    "ordinary_hmm": (
        "Estimated transition matrix for the ordinary 4-state HMM.",
        "tab:ordinary_hmm_transition",
    ),
    "ar_hmm": (
        "Estimated transition matrix for the AR-HMM.",
        "tab:ar_hmm_transition",
    ),
    "ar_2_hmm": (
        "Estimated transition matrix for the AR(2)-HMM.",
        "tab:ar_2_hmm_transition",
    ),
    "second_order_hmm": (
        "Estimated transition probabilities for the AR(1) second-order HMM. Each "
        "row is a state pair $(s_{t-1}, s_t)$; columns give the next-state "
        "probability $P(s_{t+1} \\mid s_{t-1}, s_t)$. A dash ($-$) indicates zero "
        "or negligible probability.",
        "tab:second_order_hmm_transition",
    ),
    "ar_2_second_order_hmm": (
        "Estimated transition probabilities for the AR(2) second-order HMM. Each "
        "row is a state pair $(s_{t-1}, s_t)$; columns give the next-state "
        "probability $P(s_{t+1} \\mid s_{t-1}, s_t)$. A dash ($-$) indicates zero "
        "or negligible probability.",
        "tab:ar_2_second_order_hmm_transition",
    ),
    "covariate_hmm": (
        "Estimated baseline transition matrix for the covariate HMM. This is the "
        "covariate-free intercept; the time-of-day covariates modulate these "
        "probabilities at each step.",
        "tab:covariate_hmm_transition",
    ),
    "ar_1_covariate_hmm": (
        "Estimated baseline transition matrix for the AR(1) covariate HMM. This is "
        "the covariate-free intercept; the time-of-day covariates modulate these "
        "probabilities at each step.",
        "tab:ar_1_covariate_hmm_transition",
    ),
    "ar_2_covariate_hmm": (
        "Estimated baseline transition matrix for the AR(2) covariate HMM. This is "
        "the covariate-free intercept; the time-of-day covariates modulate these "
        "probabilities at each step.",
        "tab:ar_2_covariate_hmm_transition",
    ),
}

MODEL_NAMES = list(MODEL_META)


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


# --------------------------------------------------------------------------- #
# LaTeX rendering
# --------------------------------------------------------------------------- #
def _prob(value):
    """Format a probability to 4 dp, or a dash when it rounds to zero."""
    return "$-$" if round(float(value), 4) == 0.0 else f"{float(value):.4f}"


def _table(col_spec, header_cols, body_rows, caption, label, placement):
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


def _first_order_table(gamma, caption, label):
    k = gamma.shape[0]
    header = [""] + [f"State {j + 1}" for j in range(k)]
    col_spec = "l" + "c" * k
    rows = []
    for i in range(k):
        cells = [f"State {i + 1}"] + [f"{float(gamma[i, j]):.4f}" for j in range(k)]
        rows.append(" & ".join(cells) + " \\\\")
    return _table(col_spec, header, rows, caption, label, placement="ht")


def _second_order_table(gamma, k, caption, label):
    header = ["From $(s_{t-1}, s_t)$"] + [f"$s_{{t+1}} = {j + 1}$" for j in range(k)]
    col_spec = "l" + "c" * k
    rows = []
    # Augmented index i = s_prev * k + s_curr (0-indexed). From (s_prev, s_curr) the
    # only reachable augmented states are (s_curr, s_next), i.e. column j = s_curr*k + s_next.
    for s_prev in range(k):
        for s_curr in range(k):
            i = s_prev * k + s_curr
            probs = [_prob(gamma[i, s_curr * k + s_next]) for s_next in range(k)]
            label_pair = f"$({s_prev + 1}, {s_curr + 1})$"
            rows.append(" & ".join([label_pair] + probs) + " \\\\")
        rows.append("\\hline")
    rows = rows[:-1]  # drop trailing separator; _table adds the closing \hline
    return _table(col_spec, header, rows, caption, label, placement="H")


def build_transition_table_tex(model_name, model):
    caption, label = MODEL_META[model_name]
    gamma, order = extract_transition_matrix(model)
    if order > 1:
        k = int(round(gamma.shape[0] ** (1.0 / order)))
        return _second_order_table(gamma, k, caption, label)
    return _first_order_table(gamma, caption, label)


# --------------------------------------------------------------------------- #
def main_transition_params():
    for model_name in MODEL_NAMES:
        model = load_model(f"results/models/{model_name}.pkl")
        tex = build_transition_table_tex(model_name, model)
        out_path = f"report/model_results/{model_name}/transition_matrix.tex"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write(tex)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main_transition_params()
