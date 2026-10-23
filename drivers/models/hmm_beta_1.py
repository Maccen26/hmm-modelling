import jax
jax.config.update("jax_enable_x64", True)

from functools import partial

import jax.numpy as jnp

from drivers.models.evaluate import evaluate
from drivers.models.utils import Config, Data, fit_and_report, load_data, load_fit, parse_states, ridge_negative_likelihood, save_result
from src.api.v5 import DynamicTransition, Params
from src.api.v5.solvers.base_solver import SolverResult

NAME = "HMM(b,1)"
RIDGE_LAMBDA = 1  # L2 penalty on beta; without it some betas diverge and the fit never converges


def build(cfg: Config, data: Data) -> Params:
    base = load_fit(cfg, "hmm_1")
    beta = jnp.ones((len(cfg.covariate_cols), cfg.states, cfg.states - 1)) * 0.05
    # Floor Γ: entries that underflowed to 0 in hmm_1 would give -inf logits, which L-BFGS turns into NaN on step 1
    gamma = jnp.maximum(base.transition.transition_matrix(), 1e-8)
    gamma = gamma / gamma.sum(axis=1, keepdims=True)
    transition = DynamicTransition.from_params(transition_matrix=gamma, beta=beta)
    return Params(emission=base.emission, transition=transition)


def fit(cfg: Config, data: Data | None = None) -> SolverResult:
    data = load_data(cfg) if data is None else data
    return fit_and_report(NAME, build(cfg, data), data.dynamic(), cfg, loss_fn=partial(ridge_negative_likelihood, lam=RIDGE_LAMBDA))


if __name__ == "__main__":
    ROOM, TAG      = "room_009", "30min"
    OBS_COLS       = ["co2"]  # ≥2 columns -> multivariate emission
    COVARIATE_COLS = ["off_day", "sin_week_1", "cos_week_1", "sin_tod_1", "cos_tod_1", "working_hours"]
    STATES         = parse_states(default=4)

    cfg = Config(ROOM, TAG, OBS_COLS, COVARIATE_COLS, STATES)
    data = load_data(cfg)
    result = fit(cfg, data)
    save_result(result, cfg, "hmm_beta_1", data)
    evaluate(NAME, "hmm_beta_1", result.params, cfg, data, load_data(cfg, "val"))
