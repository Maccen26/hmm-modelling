import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from drivers.models.utils import Config, Data, Loss, ar1_emission_from, fit_and_report, load_data, load_fit
from src.api.v5 import AutoregressiveGaussEmission, Params
from src.api.v5.hmm.emissions.utils import phi_tilde_to_phi, phi_to_phi_tilde
from src.api.v5.solvers.base_solver import SolverResult

NAME = "AR(1)-HMM(1), phi<=0.98"
SLUG = "ar1_hmm_1_capped"
LOSS = Loss()
DYNAMIC = False
PHI_MAX = 0.98


class CappedAutoregressiveGaussEmission(AutoregressiveGaussEmission):
    """AR(1) emission with phi mapped into (-PHI_MAX, PHI_MAX) instead of (-1, 1)."""

    def phi(self):
        return PHI_MAX * phi_tilde_to_phi(self.phi_tilde)

    @classmethod
    def capped_from(cls, emission: AutoregressiveGaussEmission, margin: float = 1e-3):
        """Copy of a fitted emission with its phi pulled just inside the cap."""
        phi = jnp.clip(emission.phi(), -PHI_MAX + margin, PHI_MAX - margin)
        return cls(emission.log_mu_diff, emission.mu0, emission.log_sigma, phi_to_phi_tilde(phi / PHI_MAX))


def build(cfg: Config, data: Data) -> Params:
    base = load_fit(cfg, "hmm_1")
    emission = CappedAutoregressiveGaussEmission.capped_from(ar1_emission_from(base.emission, cfg))
    return Params(emission=emission, transition=base.transition)


def fit(cfg: Config, data: Data | None = None) -> SolverResult:
    data = load_data(cfg) if data is None else data
    return fit_and_report(NAME, build(cfg, data), data.static(), cfg)


if __name__ == "__main__":
    from drivers.models.run import main
    # Use the imported module's fit so the pickled emission class resolves by import path, not __main__
    from drivers.models import ar1_hmm_1_capped as driver
    main(NAME, SLUG, LOSS, driver.fit, DYNAMIC, default_states=4)
