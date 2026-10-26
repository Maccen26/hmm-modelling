import jax
jax.config.update("jax_enable_x64", True)

from drivers.models.utils import Config, Data, Loss, fit_and_report, load_data, seed_emission, seed_transition
from src.api.v5 import Params
from src.api.v5.solvers.base_solver import SolverResult

NAME = "HMM(1)"
SLUG = "hmm_1"
LOSS = Loss()
DYNAMIC = False


def build(cfg: Config, data: Data) -> Params:
    return Params(emission=seed_emission(cfg, data.ys), transition=seed_transition(cfg.states))


def fit(cfg: Config, data: Data | None = None) -> SolverResult:
    data = load_data(cfg) if data is None else data
    return fit_and_report(NAME, build(cfg, data), data.static(), cfg)


if __name__ == "__main__":
    from drivers.models.run import main
    main(NAME, SLUG, LOSS, fit, DYNAMIC, default_states=4)
