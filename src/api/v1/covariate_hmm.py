import jax
import jax.numpy as jnp
import equinox as eqx

from src.deprecated.base.hmm import HMM
from src.deprecated.base.transition import Transition
from src.api.v1.stationary_hmm import GaussianEmisionBackground
from jax.scipy.stats import norm 


class CovariateTransition(Transition):
    beta: jnp.ndarray

    def __init__(self, transition_logits, beta, initial_state_dist):
        self.beta = beta
        super().__init__(transition_logits, initial_state_dist=initial_state_dist) 
 

    def step(self, xt = None): 
        """
        xt is the covarites at time step t. 
        Returns the the transtion logits matrix at time step t of dim (num_states, num_states) 
        """
        tGamma = jnp.zeros((self.num_states, self.num_states))

        diag_vals = self.beta @ xt  # shape (num_states,)

        tGamma = tGamma.at[jnp.diag_indices(self.num_states)].set(diag_vals)


        rows, cols = jnp.where(~jnp.eye(self.num_states, dtype=bool), size=self.num_states * (self.num_states - 1))
        tGamma = tGamma.at[rows, cols].set(self.transition_logits.flatten())
        return tGamma  

 

    

class CovariateHMM(HMM):
    def __init__(self, transition_logits, mu, log_sigma, initial_state_dist, beta):
        super().__init__(transition_logits, initial_state_dist=initial_state_dist) 
        self.transition = CovariateTransition(transition_logits=transition_logits, beta=beta, initial_state_dist=initial_state_dist)
        self.emission = GaussianEmisionBackground(mu=mu, log_sigma=log_sigma) 

    def filter_spec(self):
        
        spec = jax.tree_util.tree_map(eqx.is_inexact_array, self)

        spec = eqx.tree_at(
              lambda m: m.transition.initial_state_dist,
              spec,
              replace=False
          )
        return spec

