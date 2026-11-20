"""Sandwich standard errors of the working and natural parameters of a saved (ridge-penalised) fit.

    H     = Hessian of the penalised NLL at the estimate
    I(n)  = H - Hessian of the ridge penalty            (observed Fisher information of the NLL)
    Cov   = H^-1 I(n) H^-1                               (working parameters)
    Cov_n = J Cov J^T,  J = d natural / d working        (delta method, natural parameters)

The penalty is beta_lam * sum(beta^2), so its Hessian is 2 * beta_lam on the beta entries and 0 elsewhere.
H is badly scaled (mu0 is in ppm, the logits are O(1)), so it is inverted after the diagonal scaling
D H D with D = diag(1 / sqrt(H_ii)); eigenvalues of the scaled Hessian below `--eig-tol` (relative to the
largest) are treated as unidentified directions: they are dropped from the inverse and the parameters
that load on them are flagged.

Writes `{model}_standard_errors.csv` next to the fit.

    uv run python -m drivers.analysis.standard_errors results/room_015/calibrated_anchor/random_search -K 4
"""
import jax
jax.config.update("jax_enable_x64", True)

import argparse
import os

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.flatten_util import ravel_pytree

from drivers.models.utils import negative_likelihood
from drivers.utils import load_model


def leaf_names(tree) -> list[str]:
    """One name per scalar in `ravel_pytree(tree)` order, e.g. `transition.beta[0,1,2]`."""
    names = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        base = "".join(str(p) for p in path).lstrip(".")
        names += [base] if leaf.ndim == 0 else [f"{base}[{','.join(map(str, i))}]" for i in np.ndindex(leaf.shape)]
    return names


def natural_vector(params, ys, xs_settings: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    """The natural parameters as jnp arrays (differentiable): mu, sigma, phi, Gamma per covariate setting, beta."""
    em, tr = params.emission, params.transition
    out = {"mu": em.mu_vals(0, ys) if hasattr(em, "mu_vals") else em.mu(0, ys), "sigma": em.sigma(0, ys)}
    if hasattr(em, "phi"):
        out["phi"] = em.phi()[0]
    if hasattr(tr, "beta"):
        out["Gamma (base)"] = tr.base_transition_matrix()
        out |= {f"Gamma ({name})": tr.transition_matrix(x) for name, x in xs_settings.items()}
        out["beta"] = tr.beta
    else:
        out["Gamma"] = tr.transition_matrix()
    return out


def flat_names(blocks: dict[str, jnp.ndarray]) -> list[str]:
    """`mu[0]`, `Gamma (base)[1,2]`, ... in the order of `jnp.concatenate([v.ravel() ...])` (1-based states)."""
    return [f"{name}[{','.join(str(j + 1) for j in i)}]" for name, v in blocks.items() for i in np.ndindex(v.shape)]


def scaled_inverse(h: np.ndarray, eig_tol: float):
    """H^-1 via the diagonally scaled Hessian; returns (H^-1, D, eigenvalues, dropped eigenvectors)."""
    d = 1.0 / np.sqrt(np.abs(np.diag(h)))
    hs = d[:, None] * h * d[None, :]
    eigval, eigvec = np.linalg.eigh(hs)
    keep = eigval > eig_tol * eigval.max()
    hs_inv = (eigvec[:, keep] / eigval[keep]) @ eigvec[:, keep].T
    return d[:, None] * hs_inv * d[None, :], d, eigval, eigvec[:, ~keep]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_dir", help="folder holding the states_{K}/ subfolders")
    parser.add_argument("--states", "-K", type=int, required=True)
    parser.add_argument("--model", default="ar1_hmm_beta_1", help="slug of the saved fit")
    parser.add_argument("--eig-tol", type=float, default=1e-10,
                        help="relative eigenvalue cutoff of the scaled Hessian (default 1e-10)")
    args = parser.parse_args()

    path = os.path.join(args.results_dir, f"states_{args.states}", f"{args.model}.pkl")
    saved = load_model(path)
    params, cfg, d = saved["result"].params, saved["config"], saved["data"]
    data = (d.ys, d.xs, d.init_dist)
    lam = cfg.loss.beta_lam

    trainable, static = eqx.partition(params, eqx.is_inexact_array)
    theta, unravel = ravel_pytree(trainable)
    names = leaf_names(trainable)
    rebuild = lambda th: eqx.combine(unravel(th), static)

    def penalty(th):
        p = rebuild(th)
        return lam * jnp.sum(p.transition.beta ** 2) if lam and hasattr(p.transition, "beta") else 0.0

    objective = lambda th: negative_likelihood(rebuild(th), data) + penalty(th)

    print(f"{args.model} ({cfg.room}, K={cfg.states}, loss={cfg.loss.slug}): {theta.size} working parameters")
    grad = np.asarray(jax.grad(objective)(theta))
    print(f"max |gradient| at the estimate: {np.abs(grad).max():.2e}")

    h = np.asarray(jax.hessian(objective)(theta))
    h = (h + h.T) / 2
    info = h - np.asarray(jax.hessian(penalty)(theta))

    raw = np.linalg.eigvalsh(h)
    print(f"raw Hessian eigenvalues: min {raw.min():.3e}, max {raw.max():.3e}")
    h_inv, scale, eigval, dropped = scaled_inverse(h, args.eig_tol)
    print(f"scaled Hessian eigenvalues: min {eigval.min():.3e}, max {eigval.max():.3e}, "
          f"condition number {eigval.max() / max(eigval.min(), 1e-300):.3e}")
    print(f"smallest 5: {np.array2string(eigval[:5], precision=3)}")
    if eigval.min() <= 0:
        print("WARNING: the Hessian is not positive definite -- the estimate may not be a local minimum.")
    flagged = np.zeros(theta.size, dtype=bool)
    for v in dropped.T:
        flagged |= np.abs(v) > 0.3
        print("dropped direction, loads on:", [names[i] for i in np.argsort(-np.abs(v))[:4]])

    cov = h_inv @ info @ h_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))

    covariates = list(saved.get("covariate_cols", ()))
    settings = {c: x for c, x in zip(covariates, jnp.eye(len(covariates)))}
    nat_fn = lambda th: jnp.concatenate(
        [v.ravel() for v in natural_vector(rebuild(th), d.ys, settings).values()])
    blocks = natural_vector(params, d.ys, settings)
    jac = np.asarray(jax.jacfwd(nat_fn)(theta))
    cov_nat = jac @ cov @ jac.T
    se_nat = np.sqrt(np.clip(np.diag(cov_nat), 0, None))
    flagged_nat = (np.abs(jac[:, flagged]) > 1e-8 * np.abs(jac).max()).any(axis=1)

    table = pd.concat([
        pd.DataFrame({"kind": "working", "parameter": names, "estimate": np.asarray(theta), "se": se,
                      "weakly_identified": flagged}),
        pd.DataFrame({"kind": "natural", "parameter": flat_names(blocks), "estimate": np.asarray(nat_fn(theta)),
                      "se": se_nat, "weakly_identified": flagged_nat}),
    ], ignore_index=True)
    with pd.option_context("display.max_rows", None, "display.width", 140, "display.float_format", "{:.4g}".format):
        print(table.to_string(index=False))

    out = os.path.join(os.path.dirname(path), f"{args.model}_standard_errors.csv")
    table.to_csv(out, index=False)
    print(f"wrote {out}")
