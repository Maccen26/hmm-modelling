import jax
import jax.numpy as jnp
import equinox as eqx
from jax.scipy.stats import norm

from src.deprecated.base.hmm import HMM
from src.api.v2.stationary_hmm import StationaryTransition
from src.api.v2.ar_hmm_constrained import ArGaussianEmisionBackgroundConstrained

    
class ArHMMPriorConstrained(HMM):
    phi_sigma: float = eqx.field(static=True)

    def __init__(self, transition_logits, mu, log_sigma, phi_tilde, phi_sigma: float):
        super().__init__(transition_logits, initial_state_dist=None)
        self.transition = StationaryTransition(transition_logits)
        self.emission = ArGaussianEmisionBackgroundConstrained(mu, log_sigma, phi_tilde)
        self.phi_sigma = phi_sigma

    def log_likelihood(self, y, x=None):
        _, ft, _ = self.forward(y, x)
        log_likelihood = jnp.sum(jnp.log(ft)) 
        log_prior = self.log_prior()
        log_likelihood += log_prior
        return log_likelihood

    def log_prior(self):
        """
        Prior placed on the constrained phi (after sigmoid transform).
        """
        phi_tilde = self.emission.phi_tilde
       # abs_grad = self._transform_phi_grad(phi_tilde) # |dphi/dphi_tilde| for the change of variables

        log_likelihood = jnp.sum(norm.logpdf(phi_tilde, loc=0.0, scale=self.phi_sigma)) #- jnp.log(abs_grad))

        return log_likelihood
    
    def _transform_phi(self, phi_tilde):
        """Transforms the unconstrained phi_tilde to the constrained phi in (-1, 1) using a sigmoid-like function."""
        return (2 * jnp.exp(phi_tilde)) / (1 + jnp.exp(phi_tilde)) - 1

    def _transform_phi_grad(self, phi_tilde):
        """Computes the absolute value of the gradient of the phi transformation.
        """
        exp_phi_tilde = jnp.exp(phi_tilde)
        denominator = jnp.power(1 + exp_phi_tilde, 2)

        return (2 * exp_phi_tilde) / denominator
    def filter_spec(self):
        spec = jax.tree_util.tree_map(eqx.is_inexact_array, self)

        spec = eqx.tree_at(
            lambda m: m.transition.initial_state_dist,
            spec,
            replace=False
        )
        return spec
