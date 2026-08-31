import jax
import jax.numpy as jnp
import equinox as eqx
from jax.scipy.stats import norm

from src.deprecated.base.hmm import HMM
from src.api.v3.ar_prior_hmm import ArHMMPhiPrior
from src.api.v3.stationary_hmm import StationaryTransition
from src.api.v3.ar_hmm_constrained import ArGaussianEmisionConstrained


    
class ArHMMPriorConstrained(ArHMMPhiPrior):
    phi_sigma: float = eqx.field(static=True)

    def __init__(self, transition_logits, mu, log_sigma, phi_tilde, phi_sigma: float, lags=1):
        super().__init__(transition_logits, mu=mu, log_sigma=log_sigma, phi=phi_tilde, phi_sigma=phi_sigma, lags=lags)
        self.transition = StationaryTransition(transition_logits)
        self.emission = ArGaussianEmisionConstrained(mu, log_sigma, phi_tilde)
        self.phi_sigma = phi_sigma
