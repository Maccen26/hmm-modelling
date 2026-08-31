"""Render per-model transition_matrix.tex and emission_params.tex from the
fitted pickles under results/models/.  Layouts match the existing files in
report/model_results/<model>/, so the rendered output is a drop-in
replacement and the report's \\input / \\ref calls keep working.
"""
import os

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from drivers.utils import load_model
from src.api.v4.transitions.static_transition_higher_order import decode_possible_transitions
from src.api.v4.utils import load_y_data


REPORT_DIR = "report/model_results"
MODELS_DIR = "results/models"
K = 4  # number of latent states (all fitted models are 4-state)


CAPTIONS = {
    "ordinary_hmm": {
        "transition": (
            "Estimated transition matrix for the ordinary 4-state HMM.",
            "tab:ordinary_hmm_transition",
        ),
        "emission": (
            "Estimated emission parameters (Gaussian) for the ordinary 4-state HMM.",
            "tab:ordinary_hmm_emissions",
        ),
    },
    "ar_hmm": {
        "transition": (
            "Estimated transition matrix for the AR-HMM.",
            "tab:ar_hmm_transition",
        ),
        "emission": (
            "Estimated emission parameters for the AR-HMM (Gaussian with AR(1) residuals).",
            "tab:ar_hmm_emissions",
        ),
    },
    "ar_2_hmm": {
        "transition": (
            "Estimated transition matrix for the AR(2)-HMM.",
            "tab:ar_2_hmm_transition",
        ),
        "emission": (
            "Estimated emission parameters for the AR(2)-HMM (Gaussian with AR(2) residuals).",
            "tab:ar_2_hmm_emissions",
        ),
    },
    "second_order_hmm": {
        "transition": (
            "Estimated transition probabilities for the AR(1) second-order HMM. "
            "Each row is a state pair $(s_{t-1}, s_t)$; columns give the "
            "next-state probability $P(s_{t+1} \\mid s_{t-1}, s_t)$. "
            "A dash ($-$) indicates zero or negligible probability.",
            "tab:second_order_hmm_transition",
        ),
        "emission": (
            "Estimated emission parameters for the AR(1) second-order HMM. "
            "Each row is a state pair $(s_{t-1}, s_t)$; "
            "$\\hat{\\mu}$ and $\\hat{\\sigma}$ are the Gaussian mean and "
            "standard deviation of the residual and $\\hat{\\phi}$ is the AR(1) "
            "coefficient.",
            "tab:second_order_hmm_emissions",
        ),
    },
    "ar_2_second_order_hmm": {
        "transition": (
            "Estimated transition probabilities for the AR(2) second-order HMM. "
            "Each row is a state pair $(s_{t-1}, s_t)$; columns give the "
            "next-state probability $P(s_{t+1} \\mid s_{t-1}, s_t)$. "
            "A dash ($-$) indicates zero or negligible probability.",
            "tab:ar_2_second_order_hmm_transition",
        ),
        "emission": (
            "Estimated emission parameters for the AR(2) second-order HMM. "
            "$\\hat{\\sigma}$ and the AR(2) coefficients "
            "$\\hat{\\phi}_1$, $\\hat{\\phi}_2$ depend on $(s_{t-1}, s_t)$. "
            "A sum $\\hat{\\phi}_1 + \\hat{\\phi}_2 > 1$ indicates a "
            "non-stationary AR process for that state pair.",
            "tab:ar_2_second_order_hmm_emissions",
        ),
    },
}


def _wrap_table(body: str, caption: str, label: str) -> str:
    return (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        f"{body}"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )


def _fmt_signed(v: float, width_total: int = 8, decimals: int = 4) -> str:
    """Render a possibly-negative AR coefficient with TeX-friendly minus.

    Positive values render as ` 0.6235` (leading space for alignment with
    `$-0.0781$`).
    """
    if v < 0:
        return f"$-{abs(v):.{decimals}f}$"
    return f" {v:.{decimals}f}"


# ----- First-order transition (4x4) -----

def render_first_order_transition(model, caption: str, label: str) -> str:
    T = jnp.asarray(model.transition.transition_matrix())
    col_headers = [f"State {j + 1}" for j in range(K)]
    rows = []
    for i in range(K):
        cells = [f"State {i + 1}"] + [f"{float(T[i, j]):.4f}" for j in range(K)]
        rows.append(" & ".join(cells) + " \\\\")
    body = (
        "\\begin{tabular}{lcccc}\n"
        "\\hline\n"
        " & " + " & ".join(col_headers) + " \\\\\n"
        "\\hline\n"
        + "\n".join(rows) + "\n"
        "\\hline\n"
        "\\end{tabular}\n"
    )
    return _wrap_table(body, caption, label)


# ----- First-order emission -----

def render_first_order_emission(model, ys, ar_lag: int, caption: str, label: str) -> str:
    mu = jnp.asarray(model.emission.mu(0, ys))
    sigma = jnp.asarray(model.emission.sigma(0, ys))
    phi = None
    if ar_lag > 0:
        phi = jnp.asarray(model.emission.phi())  # (lag, K)

    headers = ["State", "Mean ($\\hat{\\mu}$)", "Std.\\ Dev.\\ ($\\hat{\\sigma}$)"]
    if ar_lag == 1:
        headers.append("AR coeff.\\ ($\\hat{\\phi}$)")
    elif ar_lag == 2:
        headers.append("AR coeff.\\ ($\\hat{\\phi}_1$)")
        headers.append("AR coeff.\\ ($\\hat{\\phi}_2$)")
        headers.append("$\\hat{\\phi}_1 + \\hat{\\phi}_2$")

    rows = []
    for s in range(K):
        cells = [
            f"State {s + 1}",
            f"{float(mu[s]):8.3f}",
            f"{float(sigma[s]):8.3f}",
        ]
        if ar_lag == 1:
            cells.append(_fmt_signed(float(phi[0, s])))
        elif ar_lag == 2:
            phi1 = float(phi[0, s])
            phi2 = float(phi[1, s])
            cells.append(_fmt_signed(phi1))
            cells.append(_fmt_signed(phi2))
            cells.append(_fmt_signed(phi1 + phi2))
        rows.append(" & ".join(cells) + " \\\\")

    col_spec = "l" + "c" * (len(headers) - 1)
    body = (
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\hline\n"
        + " & ".join(headers) + " \\\\\n"
        "\\hline\n"
        + "\n".join(rows) + "\n"
        "\\hline\n"
        "\\end{tabular}\n"
    )
    return _wrap_table(body, caption, label)


# ----- Second-order transition -----

def render_second_order_transition(model, ys, caption: str, label: str) -> str:
    Gamma = jnp.asarray(model.transition.transition_matrix(0, ys))
    decoded = decode_possible_transitions(Gamma, order=2)

    # Build a (16, 4) table indexed by (s_prev, s_t) → next-state probs.
    # Decoded values are keyed by state tuples; (next_s_prev, next_s_t).
    next_probs = {}
    for from_tuple, transitions in decoded.items():
        row = {}
        for to_tuple, prob in transitions:
            # In a valid second-order chain to_tuple = (s_t, s_{t+1}).
            s_next = int(to_tuple[1])
            row[s_next] = float(prob)
        next_probs[from_tuple] = row

    col_headers = [f"$s_{{t+1}} = {j + 1}$" for j in range(K)]
    rows = []
    for s_prev in range(K):
        for s_t in range(K):
            row_probs = next_probs.get((s_prev, s_t), {})
            cells = [f"$({s_prev + 1}, {s_t + 1})$"]
            for j in range(K):
                if j in row_probs and not jnp.isclose(row_probs[j], 0.0, atol=5e-5):
                    cells.append(f"{row_probs[j]:.4f}")
                else:
                    cells.append("$-$    ")
            rows.append(" & ".join(cells) + " \\\\")
        if s_prev < K - 1:
            rows.append("\\hline")

    body = (
        "\\begin{tabular}{lcccc}\n"
        "\\hline\n"
        "From $(s_{t-1}, s_t)$ & " + " & ".join(col_headers) + " \\\\\n"
        "\\hline\n"
        + "\n".join(rows) + "\n"
        "\\hline\n"
        "\\end{tabular}\n"
    )
    return _wrap_table(body, caption, label)


# ----- Second-order emission -----

def render_second_order_emission(model, ys, ar_lag: int, caption: str, label: str) -> str:
    mu = jnp.asarray(model.emission.mu_vals(0, ys))  # (16,)
    sigma = jnp.asarray(model.emission.sigma(0, ys))  # (16,)
    phi = None
    if ar_lag > 0:
        phi = jnp.asarray(model.emission.phi())  # (lag, 16)

    headers = ["$s_{t-1}$", "$s_t$", "$\\hat{\\mu}$", "$\\hat{\\sigma}$"]
    if ar_lag == 1:
        headers.append("$\\hat{\\phi}$")
    elif ar_lag == 2:
        headers.append("$\\hat{\\phi}_1$")
        headers.append("$\\hat{\\phi}_2$")
        headers.append("$\\hat{\\phi}_1 + \\hat{\\phi}_2$")

    # Existing files iterate s_t outer, s_prev inner. Augmented-state index is
    # s_prev * K + s_t (consistent with decode_possible_transitions).
    rows = []
    for s_t in range(K):
        for s_prev in range(K):
            idx = s_prev * K + s_t
            cells = [
                f"{s_prev + 1}",
                f"{s_t + 1}",
                f"{float(mu[idx]):8.3f}",
                f"{float(sigma[idx]):8.3f}",
            ]
            if ar_lag == 1:
                cells.append(_fmt_signed(float(phi[0, idx])))
            elif ar_lag == 2:
                phi1 = float(phi[0, idx])
                phi2 = float(phi[1, idx])
                cells.append(_fmt_signed(phi1))
                cells.append(_fmt_signed(phi2))
                cells.append(_fmt_signed(phi1 + phi2))
            rows.append(" & ".join(cells) + " \\\\")
        if s_t < K - 1:
            rows.append("\\hline")

    col_spec = "c" * len(headers)
    body = (
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\hline\n"
        + " & ".join(headers) + " \\\\\n"
        "\\hline\n"
        + "\n".join(rows) + "\n"
        "\\hline\n"
        "\\end{tabular}\n"
    )
    return _wrap_table(body, caption, label)


def render_all():
    ys = load_y_data()

    specs = [
        ("ordinary_hmm", False, 0),
        ("ar_hmm",       False, 1),
        ("ar_2_hmm",     False, 2),
        ("second_order_hmm",       True, 1),
        ("ar_2_second_order_hmm",  True, 2),
    ]

    for name, second_order, ar_lag in specs:
        model = load_model(f"{MODELS_DIR}/{name}.pkl")
        out_dir = f"{REPORT_DIR}/{name}"
        os.makedirs(out_dir, exist_ok=True)

        t_caption, t_label = CAPTIONS[name]["transition"]
        e_caption, e_label = CAPTIONS[name]["emission"]

        if second_order:
            t_tex = render_second_order_transition(model, ys, t_caption, t_label)
            e_tex = render_second_order_emission(model, ys, ar_lag, e_caption, e_label)
        else:
            t_tex = render_first_order_transition(model, t_caption, t_label)
            e_tex = render_first_order_emission(model, ys, ar_lag, e_caption, e_label)

        with open(f"{out_dir}/transition_matrix.tex", "w") as f:
            f.write(t_tex)
        with open(f"{out_dir}/emission_params.tex", "w") as f:
            f.write(e_tex)
        print(f"Wrote {out_dir}/transition_matrix.tex and emission_params.tex")


if __name__ == "__main__":
    render_all()
