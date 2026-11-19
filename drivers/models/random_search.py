"""Random-restart search over the model chain HMM(1) -> HMM(b,1) -> AR(1)-HMM(1) -> AR(1)-HMM(b,1).

Every model gets N_INITS random starts plus one start seeded from the models already chosen earlier
in the chain (the same seeding as the per-model drivers; hmm_1's seed is the quantile start). Each
candidate is fitted, rejected if it has a collapsed state, and scored on the validation split. The
admissible candidate with the lowest mean validation RMSE over HORIZONS is kept and seeds the next
model. phi is not capped.

    uv run python -m drivers.models.random_search --room 15 --states 4 --tag calibrated_anchor

Writes to results/{room}/{tag}/random_search/states_{K}/: {slug}.pkl for each chosen model,
candidates.csv (every start: loss, validation metrics, parameters, rejection reason), and
val_metrics.csv / test_metrics.csv for the chosen models.
"""
import jax
jax.config.update("jax_enable_x64", True)

import argparse
import os

import jax.numpy as jnp
import numpy as np
import pandas as pd

from drivers.models.evaluate import _columns, metrics_table, rolling_predictions
from drivers.models.evaluate_test import history
from drivers.models.utils import (COVARIATE_COLS, HORIZONS, OBS_COLS, RESULTS_DIR, Config, Data, Loss,
                                  add_run_args, ar1_emission_from, fit_and_report, floored_transition_matrix,
                                  load_data, negative_likelihood, room_name, seed_emission, seed_transition)
from drivers.utils import save_model
from src.api.v5 import AutoregressiveGaussEmission, DynamicTransition, GaussEmission, Params, StaticTransition

N_INITS = 10
PHI_MAX = 0.999       # phi at or above this is a unit root (the state is a random walk)
SIGMA_MIN = 1.0       # ppm; below this a state has collapsed onto a point and the likelihood is unbounded
SIGMA_MAX = 1e4       # ppm; above this a state has blown up into a flat catch-all (CO2 never spans this)
MU_REL_TOL = 0.01     # two states whose means differ by < 1 % ...
SIGMA_RATIO_TOL = 1.5  # ... and whose sigmas are within a factor 1.5 are duplicates

# (slug, name, loss, dynamic, autoregressive), in fit order
CHAIN = (
    ("hmm_1", "HMM(1)", Loss(), False, False),
    ("hmm_beta_1", "HMM(b,1)", Loss(beta_lam=1e-2), True, False),
    ("ar1_hmm_1", "AR(1)-HMM(1)", Loss(), False, True),
    ("ar1_hmm_beta_1", "AR(1)-HMM(b,1)", Loss(beta_lam=1e-2), True, True),
)


# --- Starts -------------------------------------------------------------------------

def random_params(cfg: Config, data: Data, dynamic: bool, autoregressive: bool, rng) -> Params:
    """mu at random data quantiles, sigma a random fraction of the data std, phi in (0, 0.95),
    a diagonal-heavy random Gamma and, for b models, beta ~ N(0, 0.5^2)."""
    K, ys = cfg.states, np.asarray(data.ys)
    mu = np.sort(np.quantile(ys, np.sort(rng.uniform(0.02, 0.98, K)))) + np.arange(K) * 1e-3 * ys.std()
    sigma = ys.std() * rng.uniform(0.1, 1.0, K)
    gamma = jnp.asarray(rng.dirichlet(np.ones(K), K) * 0.2 + np.eye(K) * 0.8)
    if autoregressive:
        emission = AutoregressiveGaussEmission.from_params(mu=jnp.asarray(mu), sigma=jnp.asarray(sigma),
                                                           phi=jnp.asarray(rng.uniform(0.0, 0.95, K)))
    else:
        emission = GaussEmission.from_params(mu=jnp.asarray(mu), sigma=jnp.asarray(sigma))
    if dynamic:
        beta = jnp.asarray(rng.normal(0.0, 0.5, (len(cfg.covariate_cols), K, K - 1)))
        transition = DynamicTransition.from_params(transition_matrix=gamma, beta=beta)
    else:
        transition = StaticTransition.from_params(transition_matrix=gamma)
    return Params(emission=emission, transition=transition)


def seeded_params(slug: str, cfg: Config, data: Data, chosen: dict[str, Params]) -> Params:
    """The per-model drivers' start, built from the models chosen earlier in this search."""
    if slug == "hmm_1":
        return Params(emission=seed_emission(cfg, data.ys), transition=seed_transition(cfg.states))
    if slug == "hmm_beta_1":
        base = chosen["hmm_1"]
        beta = jnp.ones((len(cfg.covariate_cols), cfg.states, cfg.states - 1)) * 0.05
        return Params(emission=base.emission, transition=DynamicTransition.from_params(
            transition_matrix=floored_transition_matrix(base.transition), beta=beta))
    if slug == "ar1_hmm_1":
        base = chosen["hmm_1"]
        return Params(emission=ar1_emission_from(base.emission, cfg), transition=base.transition)
    return Params(emission=chosen["ar1_hmm_1"].emission, transition=chosen["hmm_beta_1"].transition)


# --- Checks and scores --------------------------------------------------------------

def emission_summary(params: Params) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    em = params.emission
    if hasattr(em, "phi_tilde"):  # the AR mean depends on the lag; mu_vals is the state level
        return np.asarray(em.mu_vals(0, 0)), np.asarray(em.sigma(0, 0)), np.asarray(em.phi()).ravel()
    return np.asarray(em.mu(0, 0)), np.asarray(em.sigma(0, 0)), None


def collapse_reason(params: Params) -> str:
    """Why a fit is inadmissible, or '' if it is fine."""
    mu, sigma, phi = emission_summary(params)
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))):
        return "non-finite parameters"
    if phi is not None and np.any(phi >= PHI_MAX):
        return f"phi=1 in state {int(np.argmax(phi)) + 1}"
    if sigma.min() < SIGMA_MIN:
        return f"sigma~0 in state {int(np.argmin(sigma)) + 1}"
    if sigma.max() > SIGMA_MAX:
        return f"sigma exploded in state {int(np.argmax(sigma)) + 1}"
    for i in range(len(mu)):
        for j in range(i + 1, len(mu)):
            same_mu = abs(mu[j] - mu[i]) < MU_REL_TOL * max(abs(mu[i]), abs(mu[j]))
            same_sigma = max(sigma[i], sigma[j]) / min(sigma[i], sigma[j]) < SIGMA_RATIO_TOL
            if same_mu and same_sigma:
                return f"states {i + 1} and {j + 1} duplicate (mu {mu[i]:.0f}/{mu[j]:.0f})"
    return ""


def score(table: pd.DataFrame) -> float:
    """Selection score: mean RMSE over the forecast horizons (lower is better)."""
    return float(table["RMSE"].mean())


def run_dir(cfg: Config) -> str:
    return os.path.join(RESULTS_DIR, cfg.room, cfg.tag, "random_search", f"states_{cfg.states}")


# --- Search -------------------------------------------------------------------------

def fit_candidate(slug: str, name: str, init: str, params: Params, cfg: Config, train: Data, val: Data,
                  dynamic: bool) -> dict:
    data = train.dynamic() if dynamic else train.static()
    row = {"K": cfg.states, "slug": slug, "init": init}
    try:
        result = fit_and_report(f"{name} [{init}]", params, data, cfg)
        row.update(loss=float(result.loss), nll=float(negative_likelihood(result.params, data)),
                   converged=bool(result.converged))
        mu, sigma, phi = emission_summary(result.params)
        row.update(mu=np.round(mu, 1).tolist(), sigma=np.round(sigma, 1).tolist(),
                   phi=None if phi is None else np.round(phi, 3).tolist())
        row["rejected"] = collapse_reason(result.params)
        table = metrics_table(name, slug, cfg, _columns(np.asarray(val.ys)),
                              rolling_predictions(result.params, train, val))
        if not np.all(np.isfinite(table["RMSE"])):
            row["rejected"] = row["rejected"] or "non-finite forecasts"
        row["val_score"] = score(table)
        row.update({f"val_RMSE_h{k}": v for k, v in zip(table["horizon"], table["RMSE"])})
        row.update({f"val_MAPE_h{k}": v for k, v in zip(table["horizon"], table["MAPE"])})
        row.update({f"val_R2_h{k}": v for k, v in zip(table["horizon"], table["R2"])})
        return row | {"_result": result, "_table": table}
    except Exception as e:  # a start that blows up is just a rejected candidate
        print(f"{name} [{init}] failed: {e!r}")
        return row | {"rejected": f"error: {type(e).__name__}", "val_score": np.inf}


def search(room: str, states: int, tag: str, n_inits: int = N_INITS):
    out = None
    chosen: dict[str, Params] = {}
    rows, val_tables, test_tables = [], [], []
    for m, (slug, name, loss, dynamic, autoregressive) in enumerate(CHAIN):
        cfg = Config(room, OBS_COLS, COVARIATE_COLS, states, loss=loss, tag=tag)
        out = run_dir(cfg)
        train, val, test = load_data(cfg), load_data(cfg, "val"), load_data(cfg, "test")

        starts = [("seeded", seeded_params(slug, cfg, train, chosen))]
        starts += [(f"random_{s}", random_params(cfg, train, dynamic, autoregressive,
                                                 np.random.default_rng([states, m, s]))) for s in range(n_inits)]
        cands = [fit_candidate(slug, name, init, p, cfg, train, val, dynamic) for init, p in starts]

        admissible = [c for c in cands if not c.get("rejected") and np.isfinite(c["val_score"])]
        if not admissible:
            raise RuntimeError(f"K={states} {slug}: every start collapsed or failed")
        best = min(admissible, key=lambda c: c["val_score"])
        for c in cands:
            c["chosen"] = c is best
        print(f"K={states} {slug}: chose {best['init']} (mean val RMSE {best['val_score']:.1f})")

        result = best["_result"]
        chosen[slug] = result.params
        os.makedirs(out, exist_ok=True)
        save_model({"result": result, "config": cfg, "covariate_cols": list(cfg.covariate_cols),
                    "data": train if dynamic else train.without_covariates(), "init": best["init"]},
                   os.path.join(out, f"{slug}.pkl"))

        val_tables.append(best["_table"].assign(init=best["init"], K=states))
        seen = history(train, val)
        test_tables.append(metrics_table(name, slug, cfg, _columns(np.asarray(test.ys)),
                                         rolling_predictions(result.params, seen, test))
                           .assign(init=best["init"], K=states, n_test=len(test.ys)).drop(columns="n_val"))

        rows += [{k: v for k, v in c.items() if not k.startswith("_")} for c in cands]
        pd.DataFrame(rows).to_csv(os.path.join(out, "candidates.csv"), index=False)
        pd.concat(val_tables).to_csv(os.path.join(out, "val_metrics.csv"), index=False)
        pd.concat(test_tables).to_csv(os.path.join(out, "test_metrics.csv"), index=False)
    print(f"Saved search results to {out}")


if __name__ == "__main__":
    parser = add_run_args(argparse.ArgumentParser())
    parser.add_argument("--inits", type=int, default=N_INITS, help=f"random starts per model (default {N_INITS})")
    args = parser.parse_args()
    search(room_name(args.room), args.states, args.tag, args.inits)
