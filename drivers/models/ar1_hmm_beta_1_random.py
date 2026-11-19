import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from drivers.models.utils import Config, Data, Loss, fit_and_report, load_data
from src.api.v5 import AutoregressiveGaussEmission, DynamicTransition, Params
from src.api.v5.solvers.base_solver import SolverResult

NAME = "AR(1)-HMM(b,1), random init"
SLUG = "ar1_hmm_beta_1_random"
LOSS = Loss(beta_lam=1e-2)  # same ridge as ar1_hmm_beta_1
DYNAMIC = True
SEEDS = (0, 1, 2, 3, 4)  # random restarts; the lowest final loss is kept
SIGMA_MIN = 1.0  # ppm; a state with sigma below this has collapsed onto a point and its likelihood is unbounded


def admissible(result: SolverResult) -> bool:
    """Finite loss and no collapsed state."""
    sigma = np.asarray(result.params.emission.sigma(0, 0))
    return bool(np.isfinite(float(result.loss)) and sigma.min() >= SIGMA_MIN)


def random_params(cfg: Config, data: Data, seed: int) -> Params:
    """Random start that uses no previous fit: mu at random data quantiles, phi in (0, 0.95)."""
    rng = np.random.default_rng(seed)
    K, C = cfg.states, len(cfg.covariate_cols)
    ys = np.asarray(data.ys)
    mu = np.sort(np.quantile(ys, np.sort(rng.uniform(0.02, 0.98, K))))
    mu = mu + np.arange(K) * 1e-3 * ys.std()  # keep mu strictly increasing for log_mu_diff
    sigma = ys.std() * rng.uniform(0.1, 1.0, K)
    phi = rng.uniform(0.0, 0.95, K)
    gamma = rng.dirichlet(np.ones(K), K) * 0.2 + np.eye(K) * 0.8
    beta = rng.normal(0.0, 0.5, (C, K, K - 1))
    emission = AutoregressiveGaussEmission.from_params(mu=jnp.asarray(mu), sigma=jnp.asarray(sigma), phi=jnp.asarray(phi))
    transition = DynamicTransition.from_params(transition_matrix=jnp.asarray(gamma), beta=jnp.asarray(beta))
    return Params(emission=emission, transition=transition)


def fit(cfg: Config, data: Data | None = None) -> SolverResult:
    data = load_data(cfg) if data is None else data
    results = []
    for seed in SEEDS:
        result = fit_and_report(f"{NAME}, seed {seed}", random_params(cfg, data, seed), data.dynamic(), cfg)
        em = result.params.emission
        print(f"seed {seed}: loss={float(result.loss):.2f}  mu={np.round(np.asarray(em.mu_vals(0, 0)), 1)}  "
              f"phi={np.round(np.asarray(em.phi()).ravel(), 3)}")
        results.append(result)
    best = min(results, key=lambda r: float(r.loss) if admissible(r) else np.inf)
    print(f"Best seed: {SEEDS[results.index(best)]} (loss {float(best.loss):.2f})")
    return best


if __name__ == "__main__":
    from drivers.models.run import main
    main(NAME, SLUG, LOSS, fit, DYNAMIC, default_states=4)
