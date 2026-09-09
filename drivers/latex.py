"""Render LaTeX tables from the computed CSVs for a given data/tag.

Given a ``data_name`` and ``tag`` this reads the CSVs produced by

* ``drivers.test_statistics``  -> results/test_statistics/{data}/{tag}/
* ``drivers.emission_params``  -> results/emission_params/{data}/{tag}/
* ``drivers.transition_params``-> results/transition_params/{data}/{tag}/
* ``drivers.beta_params``      -> results/beta_params/{data}/{tag}/

and writes the report tables to ``report/model_results/`` (the paths the report
``\\input``s). Re-run for a different data/tag to regenerate the report tables
for that split.

Run with ``python -m drivers.latex``.
"""

import os

import numpy as np
import pandas as pd

from drivers.utils import write_latex_table

# --------------------------------------------------------------------------- #
# Captions / labels
# --------------------------------------------------------------------------- #
STATS_META = {
    "model_stats": (
        "Per-model log-likelihood, AIC and BIC for the fitted HMMs.",
        "tab:model_stats",
    ),
    "lrt_comparison": (
        "Likelihood-ratio tests for each nested pair from the model "
        "hierarchy (Figure~\\ref{fig:diagram:model_hierarchy}). "
        "Negative $\\Delta$AIC/$\\Delta$BIC favour the expanded model.",
        "tab:lrt_comparison",
    ),
}

EMISSION_META = {
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

TRANSITION_META = {
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

BETA_META = {
    "covariate_hmm": ("the covariate HMM", "tab:covariate_hmm_beta"),
    "ar_1_covariate_hmm": ("the AR(1) covariate HMM", "tab:ar_1_covariate_hmm_beta"),
    "ar_2_covariate_hmm": ("the AR(2) covariate HMM", "tab:ar_2_covariate_hmm_beta"),
}


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def _csv_dir(kind: str, data_name: str, tag: str) -> str:
    return f"results/{kind}/{data_name}/{tag}"


def _tex_path(model_name: str, filename: str) -> str:
    return f"report/model_results/{model_name}/{filename}"


def _write_tex(path: str, tex: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(tex)
    print(f"wrote {path}")


# --------------------------------------------------------------------------- #
# Shared LaTeX helpers
# --------------------------------------------------------------------------- #
def _fmt(value, dp):
    """Format a float, wrapping negatives in math mode (e.g. $-0.0781$)."""
    s = f"{abs(value):.{dp}f}"
    return f"$-{s}$" if value < 0 else s


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


# --------------------------------------------------------------------------- #
# Test statistics
# --------------------------------------------------------------------------- #
def latex_test_statistics(data_name: str, tag: str):
    csv_dir = _csv_dir("test_statistics", data_name, tag)

    stats_csv = os.path.join(csv_dir, "model_stats.csv")
    if os.path.exists(stats_csv):
        caption, label = STATS_META["model_stats"]
        write_latex_table(
            pd.read_csv(stats_csv),
            _tex_path("comparison", "model_stats.tex"),
            caption=caption,
            label=label,
        )
        print(_tex_path("comparison", "model_stats.tex"))
    else:
        print(f"skipping model_stats: {stats_csv} not found")

    lrt_csv = os.path.join(csv_dir, "lrt_comparison.csv")
    if os.path.exists(lrt_csv):
        caption, label = STATS_META["lrt_comparison"]
        write_latex_table(
            pd.read_csv(lrt_csv),
            _tex_path("comparison", "lrt_comparison.tex"),
            caption=caption,
            label=label,
            float_cols_4dp=["P-val"],
        )
        print(_tex_path("comparison", "lrt_comparison.tex"))
    else:
        print(f"skipping lrt_comparison: {lrt_csv} not found")


# --------------------------------------------------------------------------- #
# Emission parameters
# --------------------------------------------------------------------------- #
def _phi_headers(num_lags, compact=False):
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


def _phi_cells(phi_vals):
    """Format the AR-coefficient cells for one state (plus their sum if >1 lag)."""
    if not phi_vals:
        return []
    cells = [_fmt(float(v), 4) for v in phi_vals]
    if len(phi_vals) > 1:
        cells.append(_fmt(float(sum(phi_vals)), 4))
    return cells


def _emission_first_order(df, phi_cols, caption, label):
    num_lags = len(phi_cols)
    header = ["State", "Mean ($\\hat{\\mu}$)", "Std.\\ Dev.\\ ($\\hat{\\sigma}$)"] + _phi_headers(num_lags)
    col_spec = "l" + "c" * (len(header) - 1)
    rows = []
    for _, r in df.iterrows():
        cells = [f"State {int(r['state'])}", _fmt(float(r["mu"]), 3), _fmt(float(r["sigma"]), 3)]
        cells += _phi_cells([r[c] for c in phi_cols])
        rows.append(" & ".join(cells) + " \\\\")
    return _table(col_spec, header, rows, caption, label, placement="ht")


def _emission_second_order(df, phi_cols, base, caption, label):
    num_lags = len(phi_cols)
    header = ["$s_{t-1}$", "$s_t$", "$\\hat{\\mu}$", "$\\hat{\\sigma}$"] + _phi_headers(num_lags, compact=True)
    col_spec = "c" * len(header)
    lookup = {(int(r["s_prev"]), int(r["s_curr"])): r for _, r in df.iterrows()}
    rows = []
    # Group by current state s_t (outer), previous state s_{t-1} (inner).
    for s_curr in range(1, base + 1):
        for s_prev in range(1, base + 1):
            r = lookup[(s_prev, s_curr)]
            cells = [str(s_prev), str(s_curr), _fmt(float(r["mu"]), 3), _fmt(float(r["sigma"]), 3)]
            cells += _phi_cells([r[c] for c in phi_cols])
            rows.append(" & ".join(cells) + " \\\\")
        rows.append("\\hline")
    rows = rows[:-1]  # drop trailing separator; _table adds the closing \hline
    return _table(col_spec, header, rows, caption, label, placement="H")


def latex_emission_params(data_name: str, tag: str):
    csv_dir = _csv_dir("emission_params", data_name, tag)
    for model_name, (caption, label) in EMISSION_META.items():
        csv_path = os.path.join(csv_dir, f"{model_name}.csv")
        if not os.path.exists(csv_path):
            print(f"skipping {model_name} emission: {csv_path} not found")
            continue
        df = pd.read_csv(csv_path)
        phi_cols = [c for c in df.columns if c.startswith("phi_")]
        order = int(df["order"].iloc[0])
        if order > 1:
            base = int(round(len(df) ** (1.0 / order)))
            tex = _emission_second_order(df, phi_cols, base, caption, label)
        else:
            tex = _emission_first_order(df, phi_cols, caption, label)
        _write_tex(_tex_path(model_name, "emission_params.tex"), tex)


# --------------------------------------------------------------------------- #
# Transition matrices
# --------------------------------------------------------------------------- #
def _prob(value):
    """Format a probability to 4 dp, or a dash when it rounds to zero."""
    return "$-$" if round(float(value), 4) == 0.0 else f"{float(value):.4f}"


def _gamma_from_df(df):
    n = int(df["from_idx"].max()) + 1
    gamma = np.zeros((n, n))
    for _, r in df.iterrows():
        gamma[int(r["from_idx"]), int(r["to_idx"])] = float(r["prob"])
    return gamma


def _transition_first_order(gamma, caption, label):
    k = gamma.shape[0]
    header = [""] + [f"State {j + 1}" for j in range(k)]
    col_spec = "l" + "c" * k
    rows = []
    for i in range(k):
        cells = [f"State {i + 1}"] + [f"{float(gamma[i, j]):.4f}" for j in range(k)]
        rows.append(" & ".join(cells) + " \\\\")
    return _table(col_spec, header, rows, caption, label, placement="ht")


def _transition_second_order(gamma, k, caption, label):
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


def latex_transition_params(data_name: str, tag: str):
    csv_dir = _csv_dir("transition_params", data_name, tag)
    for model_name, (caption, label) in TRANSITION_META.items():
        csv_path = os.path.join(csv_dir, f"{model_name}.csv")
        if not os.path.exists(csv_path):
            print(f"skipping {model_name} transition: {csv_path} not found")
            continue
        df = pd.read_csv(csv_path)
        order = int(df["order"].iloc[0])
        gamma = _gamma_from_df(df)
        if order > 1:
            k = int(df["k"].iloc[0])
            tex = _transition_second_order(gamma, k, caption, label)
        else:
            tex = _transition_first_order(gamma, caption, label)
        _write_tex(_tex_path(model_name, "transition_matrix.tex"), tex)


# --------------------------------------------------------------------------- #
# Beta coefficients
# --------------------------------------------------------------------------- #
def _input_headers(num_inputs):
    """Column header per input covariate; the 2-input case is the cos/sin pair."""
    if num_inputs == 2:
        return ["$\\beta_{\\cos}$", "$\\beta_{\\sin}$"]
    return [f"$\\beta_{{{c + 1}}}$" for c in range(num_inputs)]


def _beta_table(df, beta_cols, desc, label):
    num_inputs = len(beta_cols)
    header = ["From", "To"] + _input_headers(num_inputs)
    col_spec = "cc" + "c" * num_inputs
    rows = []
    for from_state, group in df.groupby("from_state", sort=True):
        for _, r in group.iterrows():
            cells = [f"State {int(r['from_state'])}", f"State {int(r['to_state'])}"]
            cells += [_fmt(float(r[c]), 4) for c in beta_cols]
            rows.append(" & ".join(cells) + " \\\\")
        rows.append("\\hline")
    rows = rows[:-1]  # drop trailing separator; _table adds the closing \hline

    caption = (
        f"Estimated covariate coefficients $\\beta$ for {desc}. Each row is an "
        "off-diagonal transition State $i \\to$ State $j$; columns give the effect "
        "of the time-of-day covariates $x_{\\cos}$ and $x_{\\sin}$ on that "
        "transition's logit."
    )
    return _table(col_spec, header, rows, caption, label)


def latex_beta_params(data_name: str, tag: str):
    csv_dir = _csv_dir("beta_params", data_name, tag)
    for model_name, (desc, label) in BETA_META.items():
        csv_path = os.path.join(csv_dir, f"{model_name}.csv")
        if not os.path.exists(csv_path):
            print(f"skipping {model_name} beta: {csv_path} not found")
            continue
        df = pd.read_csv(csv_path)
        beta_cols = [c for c in df.columns if c.startswith("beta_")]
        tex = _beta_table(df, beta_cols, desc, label)
        _write_tex(_tex_path(model_name, "beta_coefficients.tex"), tex)


# --------------------------------------------------------------------------- #
def main_latex(data_name: str, tag: str):
    latex_test_statistics(data_name, tag)
    latex_emission_params(data_name, tag)
    latex_transition_params(data_name, tag)
    latex_beta_params(data_name, tag)


if __name__ == "__main__":
    DATA_NAME = "b1"
    TAG = "jans-split"
    main_latex(data_name=DATA_NAME, tag=TAG)
