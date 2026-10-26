import jax
jax.config.update("jax_enable_x64", True)

from drivers.models.utils import Config, Data, Loss, ar1_emission_from, fit_and_report, load_data, load_fit
from src.api.v5 import Params
from src.api.v5.solvers.base_solver import SolverResult

NAME = "AR(1)-HMM(1)"
SLUG = "ar1_hmm_1"
LOSS = Loss()
DYNAMIC = False


def build(cfg: Config, data: Data) -> Params:
    base = load_fit(cfg, "hmm_1")
    return Params(emission=ar1_emission_from(base.emission, cfg), transition=base.transition)


def fit(cfg: Config, data: Data | None = None) -> SolverResult:
    data = load_data(cfg) if data is None else data
    return fit_and_report(NAME, build(cfg, data), data.static(), cfg)


if __name__ == "__main__":
    from drivers.models.run import main
    main(NAME, SLUG, LOSS, fit, DYNAMIC, default_states=6)
