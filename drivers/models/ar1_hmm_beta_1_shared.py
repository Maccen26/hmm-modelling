import jax
jax.config.update("jax_enable_x64", True)

from drivers.models.utils import Config, Data, Loss, fit_and_report, load_data, load_fit
from src.api.v5 import Params
from src.api.v5.solvers.base_solver import SolverResult

NAME = "AR(1)-HMM(b,1) shared cols"
SLUG = "ar1_hmm_beta_1_shared"
LOSS = Loss("nll")
DYNAMIC = True


def build(cfg: Config, data: Data) -> Params:
    emission = load_fit(cfg, "ar1_hmm_1").emission
    transition = load_fit(cfg, "hmm_beta_1_shared", covariates=True).transition
    return Params(emission=emission, transition=transition)


def fit(cfg: Config, data: Data | None = None) -> SolverResult:
    data = load_data(cfg) if data is None else data
    return fit_and_report(NAME, build(cfg, data), data.dynamic(), cfg)


if __name__ == "__main__":
    from drivers.models.run import main
    main(NAME, SLUG, LOSS, fit, DYNAMIC, default_states=4)
