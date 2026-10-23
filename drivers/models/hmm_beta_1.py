import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from drivers.models.utils import Config, Data, Loss, fit_and_report, floored_transition_matrix, load_data, load_fit
from src.api.v5 import DynamicTransition, Params
from src.api.v5.solvers.base_solver import SolverResult

NAME = "HMM(b,1)"
SLUG = "hmm_beta_1"
LOSS = Loss(beta_lam=1e-2)  # L2 penalty on beta; without it some betas diverge and the fit never converges
DYNAMIC = True


def build(cfg: Config, data: Data) -> Params:
    base = load_fit(cfg, "hmm_1")
    beta = jnp.ones((len(cfg.covariate_cols), cfg.states, cfg.states - 1)) * 0.05
    transition = DynamicTransition.from_params(transition_matrix=floored_transition_matrix(base.transition), beta=beta)
    return Params(emission=base.emission, transition=transition)


def fit(cfg: Config, data: Data | None = None) -> SolverResult:
    data = load_data(cfg) if data is None else data
    return fit_and_report(NAME, build(cfg, data), data.dynamic(), cfg)


if __name__ == "__main__":
    from drivers.models.run import main
    main(NAME, SLUG, LOSS, fit, DYNAMIC, default_states=4)
