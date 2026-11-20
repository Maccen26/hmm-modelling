"""LaTeX tables of the estimated working and natural parameters of one saved fit.

Writes four tables to `Thesis/Tables/`:
  - `params_working_emission_<name>`: mu0 / log mean differences, log sigma and phi tilde per state.
  - `params_working_transition_<name>`: the base logits log(gamma_ij / gamma_ii) next to the covariate
    effects beta on them; the diagonal is the reference and has neither.
  - `params_emission_<name>`: state mean, standard deviation and AR coefficient per state.
  - `params_transition_<name>`: the transition matrix at the base (all covariates 0) and with each
    covariate switched on alone.

Standard errors are read from `{model}_standard_errors.csv` next to the fit (written by
`drivers.analysis.standard_errors`) and shown as 95% Wald intervals estimate +- 1.96 SE, clipped to the
parameter's range (sigma > 0, -1 < phi < 1, 0 <= gamma <= 1); without that file they are left out.

    uv run python -m drivers.analysis.params_table results/room_015/calibrated_anchor/random_search -K 4
"""
import jax
jax.config.update("jax_enable_x64", True)

import argparse
import os

import numpy as np
import pandas as pd

from drivers.analysis.show_params import natural
from drivers.utils import THESIS_TABLES_DIR, load_model, save_thesis_table

Z = 1.959964  # two-sided 95% normal quantile


def load_se(path: str) -> dict[str, float]:
    """`{parameter: se}` from the standard-errors CSV, or `{}` when it has not been computed."""
    if not os.path.exists(path):
        print(f"no standard errors at {path}; tables are written without them")
        return {}
    return dict(zip(*pd.read_csv(path)[["parameter", "se"]].to_numpy().T))


def se_array(se: dict[str, float], prefix: str, shape: tuple, offset: int = 0) -> np.ndarray:
    """The SEs of `prefix[i,j,...]` (indices shifted by `offset`) as an array; NaN where missing."""
    if not shape:
        return np.array(se.get(prefix, np.nan))
    keys = (f"{prefix}[{','.join(str(j + offset) for j in i)}]" for i in np.ndindex(shape))
    return np.array([se.get(k, np.nan) for k in keys]).reshape(shape)


def ci(value: float, se: float, fmt: str, bounds=(-np.inf, np.inf)) -> str:
    """`[lower, upper]` of the 95% Wald interval clipped to `bounds`, or `""` when the SE is missing."""
    if np.isnan(se):
        return ""
    lower, upper = np.clip([value - Z * se, value + Z * se], *bounds)
    return f"[{fmt.format(lower)}, {fmt.format(upper)}]"


def cell(value: float, se: float, fmt: str, bounds=(-np.inf, np.inf)) -> str:
    """`value [lower, upper]`, or just `value` when the SE is missing."""
    return f"{fmt.format(value)} {ci(value, se, fmt, bounds)}".rstrip()


def emission_table(nat: dict[str, np.ndarray], se: dict[str, float]) -> pd.DataFrame:
    k = len(nat["mu"])
    col = lambda name, v, fmt, bounds=(-np.inf, np.inf): [
        cell(a, b, fmt, bounds) for a, b in zip(v, se_array(se, name, (k,), offset=1))]
    table = pd.DataFrame({"State": np.arange(1, k + 1),
                          "$\\mu$ (ppm)": col("mu", nat["mu"], "{:.2f}"),
                          "$\\sigma$ (ppm)": col("sigma", nat["sigma"], "{:.2f}", (0, np.inf))})
    if "phi" in nat:
        table["$\\phi$"] = col("phi", nat["phi"][0], "{:.4f}", (-1, 1))
    return table


def working_emission_table(em, se: dict[str, float]) -> pd.DataFrame:
    """State 1 holds mu0, state k > 1 the log difference log(mu_k - mu_{k-1})."""
    k = len(np.asarray(em.log_sigma))
    col = lambda name, v, fmt, shape=(k,): [cell(a, b, fmt) for a, b in zip(v, se_array(se, name, shape).ravel())]
    mu = col("emission.mu0", [float(em.mu0)], "{:.2f}", ()) + col("emission.log_mu_diff", np.asarray(em.log_mu_diff),
                                                                  "{:.4f}", (k - 1,))
    return pd.DataFrame({"State": np.arange(1, k + 1),
                         "$\\mu_0$ / $\\log(\\mu_k - \\mu_{k-1})$": mu,
                         "$\\log \\sigma$": col("emission.log_sigma", np.asarray(em.log_sigma), "{:.4f}"),
                         "$\\tilde{\\phi}$": col("emission.phi_tilde", np.asarray(em.phi_tilde)[0], "{:.4f}",
                                                (1, k))})


def full_beta(beta: np.ndarray) -> np.ndarray:
    """(K, K-1) off-diagonal block -> (K, K) with NaN on the reference diagonal."""
    k = beta.shape[0]
    out = np.full((k, k), np.nan)
    out[~np.eye(k, dtype=bool)] = beta.ravel()
    return out


def transition_latex(blocks: dict[str, tuple[np.ndarray, np.ndarray, str]], caption: str, label: str,
                     bounds=(-np.inf, np.inf)) -> str:
    """One `From x To` block of columns per entry of `{title: (matrix, se, format)}`; NaN cells print as --.

    Each state row is followed by a row of the 95% intervals (left out when all SEs are missing).
    """
    k = next(iter(blocks.values()))[0].shape[0]
    states = " & ".join(f"{j}" for j in range(1, k + 1))
    header = " & ".join(f"\\multicolumn{{{k}}}{{c}}{{{title}}}" for title in blocks)
    rules = " ".join(f"\\cmidrule(lr){{{2 + n * k}-{1 + (n + 1) * k}}}" for n in range(len(blocks)))
    rows = []
    for i in range(k):
        cells = [("--" if np.isnan(m[i, j]) else fmt.format(m[i, j])) for m, _, fmt in blocks.values() for j in range(k)]
        rows.append(f"{i + 1} & " + " & ".join(cells) + " \\\\")
        cis = [ci(m[i, j], s[i, j], fmt.replace("+", ""), bounds) for m, s, fmt in blocks.values() for j in range(k)]
        if any(cis):
            rows.append(" & " + " & ".join(f"{{\\footnotesize {c}}}" if c else "" for c in cis) + " \\\\")
    col_spec = "l" + "|".join(["c" * k] * len(blocks))
    return "\n".join([
        "\\begin{table}[ht]", "\\centering", "\\begin{adjustbox}{max width=\\textwidth}",
        f"\\begin{{tabular}}{{{col_spec}}}", "\\hline",
        f" & {header} \\\\", rules,
        "From $\\backslash$ To & " + " & ".join([states] * len(blocks)) + " \\\\", "\\hline",
        *rows, "\\hline", "\\end{tabular}", "\\end{adjustbox}",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}", ""])


def default_name(results_dir: str, model: str, states: int) -> str:
    """e.g. `room_015_calibrated_anchor_random_search_ar1_hmm_beta_1_K4`."""
    parts = os.path.normpath(results_dir).split(os.sep)
    parts = parts[parts.index("results") + 1:] if "results" in parts else parts[-3:]
    return "_".join([*parts, model, f"K{states}"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_dir", help="folder holding the states_{K}/ subfolders")
    parser.add_argument("--states", "-K", type=int, required=True)
    parser.add_argument("--model", default="ar1_hmm_beta_1", help="slug of the saved fit")
    parser.add_argument("--label", default="AR(1)-HMM($\\beta$,1)", help="model name used in the captions")
    args = parser.parse_args()

    saved = load_model(os.path.join(args.results_dir, f"states_{args.states}", f"{args.model}.pkl"))
    nat = natural(saved["result"].params, saved["data"].ys)
    name = default_name(args.results_dir, args.model, args.states)
    room = saved["config"].room.replace("_", " ").capitalize()

    of = f"of the {args.label} with $K={args.states}$ states"
    rows_cols = "Rows are the current state $i$, columns the next state $j$"
    params = saved["result"].params
    se = load_se(os.path.join(args.results_dir, f"states_{args.states}", f"{args.model}_standard_errors.csv"))
    k = args.states
    with_se = " Brackets give 95\\% Wald confidence intervals (estimate $\\pm 1.96$ SE)." if se else ""

    def write_transition(kind: str, blocks: dict, caption: str, bounds=(-np.inf, np.inf)):
        path = str(THESIS_TABLES_DIR / f"params_{kind}_{name}.tex")
        with open(path, "w") as f:
            f.write(transition_latex(blocks, caption=caption, label=f"tab:params_{kind}_{name}", bounds=bounds))
        print("wrote", path)

    em_working = working_emission_table(params.emission, se)
    print(em_working.to_string(index=False))
    print("wrote", save_thesis_table(
        em_working, name=f"params_working_emission_{name}", label=f"tab:params_working_emission_{name}",
        caption=f"{room}: estimated working emission parameters {of}. The state means are $\\mu_1 = \\mu_0$ and "
                f"$\\mu_k = \\mu_{{k-1}} + \\exp(\\log(\\mu_k - \\mu_{{k-1}}))$, "
                f"$\\sigma = \\exp(\\log \\sigma)$ and $\\phi = 2\\,\\mathrm{{sigmoid}}(\\tilde{{\\phi}}) - 1$.{with_se}",
        max_width=True))

    raw_covariates = list(saved.get("covariate_cols", ()))
    covariates = [c.replace("_", " ").capitalize() for c in raw_covariates]
    blocks = {"Base logits": (full_beta(np.asarray(params.transition.transition_logits)),
                              full_beta(se_array(se, "transition.transition_logits", (k, k - 1))), "{:+.2f}")}
    if "beta" in nat:
        beta_se = se_array(se, "transition.beta", nat["beta"].shape)
        blocks |= {f"$\\beta$: {c}": (full_beta(b), full_beta(s), "{:+.2f}")
                   for c, b, s in zip(covariates, nat["beta"], beta_se)}
    write_transition(
        "working_transition", blocks,
        f"{room}: estimated base logits $\\log(\\gamma_{{ij}}/\\gamma_{{ii}})$ and the covariate effects "
        f"$\\beta$ added to them {of}. {rows_cols}; the diagonal is the reference category.{with_se}")

    em = emission_table(nat, se)
    print(em.to_string(index=False))
    print("wrote", save_thesis_table(
        em, name=f"params_emission_{name}", label=f"tab:params_emission_{name}",
        caption=f"{room}: estimated state means $\\mu$, standard deviations $\\sigma$ and AR(1) coefficients "
                f"$\\phi$ {of}.{with_se} Intervals for $\\sigma$ and $\\phi$ are clipped to their ranges.",
        max_width=True))

    gamma = nat.get("Gamma (base, x = 0)", nat.get("Gamma"))
    if "beta" in nat:
        blocks = {"$\\Gamma$ ($x_t = 0$)": (gamma, se_array(se, "Gamma (base)", (k, k), offset=1), "{:.4f}")}
        # One block per covariate switched on alone, the others held at 0.
        for c, raw, x in zip(covariates, raw_covariates, np.eye(len(covariates))):
            blocks[f"$\\Gamma$: {c}"] = (np.asarray(params.transition.transition_matrix(x)),
                                          se_array(se, f"Gamma ({raw})", (k, k), offset=1), "{:.4f}")
    else:
        blocks = {"$\\Gamma$": (gamma, se_array(se, "Gamma", (k, k), offset=1), "{:.4f}")}
    write_transition(
        "transition", blocks,
        f"{room}: estimated transition matrix $\\Gamma$ {of}, at the base ($x_t = 0$) and with each covariate "
        f"set to 1 while the others are 0. {rows_cols}.{with_se}" + (" Intervals are clipped to $[0, 1]$." if se else ""),
        bounds=(0, 1))
