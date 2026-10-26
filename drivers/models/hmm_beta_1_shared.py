import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from drivers.models.utils import Config, Data, Loss, fit_and_report, floored_transition_matrix, load_data, load_fit
from src.api.v5 import Params
from src.api.v5.hmm.transitions.dynamic_transition import DynamicTransitionColsShared
from src.api.v5.solvers.base_solver import SolverResult

NAME = "HMM(b,1) shared cols"
SLUG = "hmm_beta_1_shared"
LOSS = Loss()
DYNAMIC = True


def build(cfg: Config, data: Data) -> Params:
    base = load_fit(cfg, "hmm_1")
    beta = jnp.ones((len(cfg.covariate_cols), cfg.states - 1)) * 0.05
    transition = DynamicTransitionColsShared.from_params(transition_matrix=floored_transition_matrix(base.transition), beta=beta)
    return Params(emission=base.emission, transition=transition)


def fit(cfg: Config, data: Data | None = None) -> SolverResult:
    data = load_data(cfg) if data is None else data
    return fit_and_report(NAME, build(cfg, data), data.dynamic(), cfg)


if __name__ == "__main__":
    from drivers.models.run import main
    main(NAME, SLUG, LOSS, fit, DYNAMIC, default_states=4)
