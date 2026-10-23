import jax
jax.config.update("jax_enable_x64", True)

from functools import partial

from drivers.models.evaluate import evaluate
from drivers.models.utils import Config, Data, fit_and_report, load_data, load_fit, parse_states, ridge_beta_sigma, save_result
from src.api.v5 import Params
from src.api.v5.solvers.base_solver import SolverResult

NAME = "AR(1)-HMM(b,1)"
RIDGE_LAMBDA = 1e-2  # L2 penalty on beta; without it some betas diverge and the fit never converges


def build(cfg: Config, data: Data) -> Params:
    emission = load_fit(cfg, "ar1_hmm_1").emission
    transition = load_fit(cfg, "hmm_beta_1", covariates=True).transition
    return Params(emission=emission, transition=transition)


def fit(cfg: Config, data: Data | None = None) -> SolverResult:
    data = load_data(cfg) if data is None else data
    return fit_and_report(NAME, build(cfg, data), data.dynamic(), cfg,
                          loss_fn=partial(ridge_beta_sigma, lam=RIDGE_LAMBDA))


if __name__ == "__main__":
    ROOM, TAG      = "room_009", "30min"
    OBS_COLS       = ["co2"]  # ≥2 columns -> multivariate emission
    COVARIATE_COLS = ["off_day", "sin_week_1", "cos_week_1", "sin_tod_1", "cos_tod_1", "working_hours"]
    STATES         = parse_states(default=6)

    cfg = Config(ROOM, TAG, OBS_COLS, COVARIATE_COLS, STATES)
    data = load_data(cfg)
    result = fit(cfg, data)
    save_result(result, cfg, "ar1_hmm_beta_1", data)
    evaluate(NAME, "ar1_hmm_beta_1", result.params, cfg, data, load_data(cfg, "val"))
