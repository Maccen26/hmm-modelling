import jax
import jax.numpy as jnp
import equinox as eqx

from src.deprecated.base.hmm import HMM
from src.api.v1.stationary_hmm import StationaryTransition
from src.api.v1.stationary_hmm import StationaryGaussianEmission


class ArGaussianEmisionBackground(StationaryGaussianEmission):
    phi: jnp.ndarray  # AR(1) coefficient per state, shape (num_states,)

    def __init__(self, mu, log_sigma, phi):
        super().__init__(mu, log_sigma)
        self.phi = phi

    def step(self, xt=None):
        mu = jnp.concatenate([jnp.array([400.0]), self.mu])  # shape (num_states,)
        #phi = jnp.exp(self.phi)  # shape (num_states,)
        mu = mu + self.phi * (xt - mu)  # xt = y_{t-1}, 
        return mu, jnp.exp(self.log_sigma)


class ArHMM(HMM):
    def __init__(self, transition_logits, mu, log_sigma, phi):
        super().__init__(transition_logits, initial_state_dist=None)
        self.transition = StationaryTransition(transition_logits)
        self.emission = ArGaussianEmisionBackground(mu, log_sigma, phi) 
    


    def filter_spec(self):
        
        spec = jax.tree_util.tree_map(eqx.is_inexact_array, self)

        spec = eqx.tree_at(
              lambda m: m.transition.initial_state_dist,
              spec,
              replace=False
          )
        return spec
