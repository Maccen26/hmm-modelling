import jax
import jax.numpy as jnp
import equinox as eqx

from src.deprecated.base.hmm import HMM
from src.api.v2.stationary_hmm import StationaryTransition
from src.api.v2.stationary_hmm import StationaryGaussianEmission


class ArGaussianEmisionBackgroundConstrained(StationaryGaussianEmission):
    phi_tilde: jnp.ndarray  # unconstrained AR(1) parameter per state, shape (num_states,)

    def __init__(self, mu, log_sigma, phi_tilde, lag = 1):
        super().__init__(mu, log_sigma)
        self.phi_tilde = phi_tilde

    def step(self, xt=None):
        phi = self.transform_phi(self.phi_tilde)
        mu = jnp.concatenate([jnp.array([400.0]), self._compute_mu()])  # shape (num_states,)
        #mu = mu + phi * (xt - mu)  # xt = y_{t-1}
        lags = (xt[None, :] -  mu[:, None] ) * phi.reshape(len(mu), -1) # shape (num_states, lags)
        mu = mu + lags.sum(axis=1) # shape (num_states,) sum rows
        return mu, jnp.exp(self.log_sigma) 
    
    def transform_phi(self, phi_tilde):
        """Transforms the unconstrained phi_tilde to the constrained phi in (-1, 1) using a sigmoid-like function."""
        return (2 * jnp.exp(phi_tilde)) / (1 + jnp.exp(phi_tilde)) - 1
    


class ArHMMConstrained(HMM):
    def __init__(self, transition_logits, mu, log_sigma, phi_tilde):
        super().__init__(transition_logits, initial_state_dist=None)
        self.transition = StationaryTransition(transition_logits)
        self.emission = ArGaussianEmisionBackgroundConstrained(mu, log_sigma, phi_tilde)

