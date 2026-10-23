import jax
jax.config.update("jax_enable_x64", True)

from drivers.models.evaluate import evaluate
from drivers.models.utils import (Config, 
                                  Data, 
                                  ar1_emission_from, 
                                  fit_and_report, 
                                  load_data, 
                                  parse_states, 
                                  load_fit, 
                                  save_result
                                  )
from src.api.v5 import Params
from src.api.v5.solvers.base_solver import SolverResult

NAME = "AR(1)-HMM(1)"


def build(cfg: Config, data: Data) -> Params:
    base = load_fit(cfg, "hmm_1")
    return Params(emission=ar1_emission_from(base.emission, cfg), transition=base.transition)


def fit(cfg: Config, data: Data | None = None) -> SolverResult:
    data = load_data(cfg) if data is None else data
    return fit_and_report(NAME, build(cfg, data), data.static(), cfg)


if __name__ == "__main__":
    ROOM, TAG      = "room_009", "30min"
    OBS_COLS       = ["co2"]  # ≥2 columns -> multivariate emission
    COVARIATE_COLS = ["off_day", "sin_week_1", "cos_week_1", "sin_tod_1", "cos_tod_1", "working_hours"]  # unused: static transition
    STATES         = parse_states(default=6)

    cfg = Config(ROOM, TAG, OBS_COLS, COVARIATE_COLS, STATES)
    data = load_data(cfg)
    result = fit(cfg, data)
    save_result(result, cfg, "ar1_hmm_1", data.without_covariates())
    evaluate(NAME, "ar1_hmm_1", result.params, cfg, data, load_data(cfg, "val"))
