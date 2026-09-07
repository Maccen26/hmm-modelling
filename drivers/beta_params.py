"""Build the covariate-coefficient (beta) tables and write them to LaTeX.

Only the covariate (dynamic-transition) models carry covariate coefficients. For
each of those models this loads the fitted ``DynamicTransition`` and writes its
``beta`` tensor to ``report/model_results/<model>/beta_coefficients.tex``.

``beta`` has shape (num_inputs, num_states, num_states - 1): the effect of each
input covariate on each off-diagonal transition logit. The transition logit from
state $i$ to state $j$ at time $t$ is ``transition_logits[i, k] +
sum_c beta[c, i, k] * x_c(t)``, where $k$ indexes the off-diagonal targets
$j \\neq i$ (increasing order). The inputs are the time-of-day harmonics
$x_{\\cos}$ and $x_{\\sin}$ built by ``drivers.utils.load_time_covariates``.

The table has one row per off-diagonal transition (State $i \\to$ State $j$) and
one column per input covariate.

Run with ``python -m drivers.beta_params``.
"""

import os

import numpy as np

from drivers.utils import load_model

MODEL_META = {
    "covariate_hmm": (
        "the covariate HMM",
        "tab:covariate_hmm_beta",
    ),
    "ar_1_covariate_hmm": (
        "the AR(1) covariate HMM",
        "tab:ar_1_covariate_hmm_beta",
    ),
    "ar_2_covariate_hmm": (
        "the AR(2) covariate HMM",
        "tab:ar_2_covariate_hmm_beta",
    ),
}

MODEL_NAMES = list(MODEL_META)


def _input_headers(num_inputs):
    """Column header per input covariate; the 2-input case is the cos/sin pair."""
    if num_inputs == 2:
        return ["$\\beta_{\\cos}$", "$\\beta_{\\sin}$"]
    return [f"$\\beta_{{{c + 1}}}$" for c in range(num_inputs)]


def _fmt(value, dp=4):
    """Format a coefficient, wrapping negatives in math mode (e.g. $-0.1781$)."""
    s = f"{abs(value):.{dp}f}"
    return f"$-{s}$" if value < 0 else s


def _table(col_spec, header_cols, body_rows, caption, label):
    header = " & ".join(header_cols) + " \\\\"
    body = "\n".join(body_rows)
    return (
        "\\begin{table}[ht]\n"
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


def build_beta_table_tex(model_name, model):
    desc, label = MODEL_META[model_name]
    beta = np.asarray(model.transition.beta)  # (num_inputs, num_states, num_states - 1)
    num_inputs, num_states, _ = beta.shape

    header = ["From", "To"] + _input_headers(num_inputs)
    col_spec = "cc" + "c" * num_inputs
    rows = []
    for i in range(num_states):
        # Off-diagonal targets for from-state i, in the same increasing order the
        # logits use (states j != i).
        targets = [j for j in range(num_states) if j != i]
        for k, j in enumerate(targets):
            cells = [f"State {i + 1}", f"State {j + 1}"]
            cells += [_fmt(float(beta[c, i, k])) for c in range(num_inputs)]
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


def main_beta_params():
    for model_name in MODEL_NAMES:
        model = load_model(f"results/models/{model_name}.pkl")
        if not hasattr(model.transition, "beta"):
            print(f"skipping {model_name}: transition has no beta coefficients")
            continue
        tex = build_beta_table_tex(model_name, model)
        out_path = f"report/model_results/{model_name}/beta_coefficients.tex"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write(tex)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main_beta_params()
