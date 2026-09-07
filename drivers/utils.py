import os
import pickle

import jax.numpy as jnp
import numpy as np


def save_model(model, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(model, f)


def load_model(save_path: str):
    with open(save_path, "rb") as f:
        model = pickle.load(f)
    return model


def format_transition_matrix(matrix: jnp.ndarray) -> str:
    formatted = "\n".join(["\t" + " ".join(f"{val:.4f}" for val in row) for row in matrix])
    return f"Transition Matrix:\n{formatted}"


def load_time_covariates(period: int = 48) -> jnp.ndarray:
    """
    Build cyclic time-of-day covariates (cos/sin of HalfHour) for the covariate HMM.

    Returns a (T, 2) array aligned with load_y_data(), since both draw from
    load_and_aggregate_data() with default arguments.
    """
    from src.data import load_and_aggregate_data  # lazy import to avoid circular import

    data = load_and_aggregate_data()
    t = jnp.asarray(data["HalfHour"], dtype=float)
    cos = jnp.cos(2 * jnp.pi * t / period)
    sin = jnp.sin(2 * jnp.pi * t / period)
    return jnp.column_stack((cos, sin))


# --------------------------------------------------------------------------- #
# Diagnostics plotting
# --------------------------------------------------------------------------- #

def plot_hmm_diagnostics(model, save_path: str | None = None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import MaxNLocator
    from scipy import stats
    from statsmodels.graphics.tsaplots import plot_acf

    residuals = np.asarray(model.state_results.pseudo_residuals).ravel()
    residuals = residuals[np.isfinite(residuals)]

    sns.set_theme(style="whitegrid", context="notebook")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ll = np.asarray(model.ll_fits)
    iterations = np.arange(1, len(ll) + 1)
    sns.lineplot(x=iterations, y=ll, ax=axes[0], marker="o")
    axes[0].set_title("Log-likelihood per iteration")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("log L")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].get_lines()[0].set_color(sns.color_palette()[0])
    axes[1].get_lines()[1].set_color(sns.color_palette()[3])
    axes[1].set_title("Normal Q-Q of pseudo-residuals")

    lags = min(40, max(1, len(residuals) // 4))
    plot_acf(residuals, lags=lags, ax=axes[2])
    axes[2].set_title("ACF of pseudo-residuals")
    axes[2].set_ylim(-0.25, 1.05)

    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)

    return fig


# --------------------------------------------------------------------------- #
# LaTeX table rendering
# --------------------------------------------------------------------------- #

_LATEX_HEADER_MAP = {
    "#Params": "\\#Params",
    "ΔAIC": "$\\Delta$AIC",
    "ΔBIC": "$\\Delta$BIC",
    "P-val": "$p$-value",
}


def _fmt_cell(value, col, float_cols_4dp):
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}" if col in float_cols_4dp else f"{value:.2f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value) + 1)
    return str(value)


def write_latex_table(df, path: str, caption: str, label: str, float_cols_4dp=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    float_cols_4dp = set(float_cols_4dp or [])
    col_spec = "l" + "c" * (len(df.columns) - 1)
    header = " & ".join(_LATEX_HEADER_MAP.get(c, c) for c in df.columns) + " \\\\"
    body_rows = []
    for _, row in df.iterrows():
        cells = [_fmt_cell(row[col], col, float_cols_4dp) for col in df.columns]
        body_rows.append(" & ".join(cells) + " \\\\")
    tex = (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\hline\n"
        f"{header}\n"
        "\\hline\n"
        + "\n".join(body_rows) + "\n"
        "\\hline\n"
        "\\end{tabular}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )
    with open(path, "w") as f:
        f.write(tex)
