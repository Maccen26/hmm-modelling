import jax
import jax.numpy as jnp
import equinox as eqx
from jax.scipy.stats import norm

from src.deprecated.base.hmm import HMM
from src.api.v2.stationary_hmm import StationaryTransition
from src.api.v2.stationary_hmm import StationaryGaussianEmission


class ArGaussianEmisionBackground(StationaryGaussianEmission):
    phi: jnp.ndarray  # free AR(1) parameter per state, shape (num_states,)

    def __init__(self, mu, log_sigma, phi):
        super().__init__(mu, log_sigma) 
        self.phi = phi

    def step(self, xt=None):
        mu = jnp.concatenate([jnp.array([400.0]), self._compute_mu()])  # shape (num_states,)
        lags = (xt[None, :] -  mu[:, None] ) * self.phi.reshape(len(mu), -1) # shape (num_states, lags)
        mu = mu + lags.sum(axis=1) 
        return mu, jnp.exp(self.log_sigma)


class ArHMMPhiPrior(HMM):
    phi_sigma: float  = eqx.field(static=True)  

    def __init__(self, transition_logits, mu, log_sigma, phi, phi_sigma: float):
        super().__init__(transition_logits, initial_state_dist=None)
        self.transition = StationaryTransition(transition_logits)
        self.emission = ArGaussianEmisionBackground(mu, log_sigma, phi)
        self.phi_sigma = phi_sigma

    def log_likelihood(self, y, x=None):
        _, ft, _ = self.forward(y, x)
        log_likelihood = jnp.sum(jnp.log(ft)) 
        log_prior = self.log_prior()
        log_likelihood += log_prior
        return log_likelihood

    def log_prior(self):
        """Prior placed on the constrained phi (after sigmoid transform).
        
        """
        phi = self.emission.phi 
        log_likelihood = jnp.sum(norm.logpdf(phi, loc=0.0, scale=self.phi_sigma))

        return log_likelihood
    

    def filter_spec(self):
        spec = jax.tree_util.tree_map(eqx.is_inexact_array, self)

        spec = eqx.tree_at(
            lambda m: m.transition.initial_state_dist,
            spec,
            replace=False
        )
        return spec



