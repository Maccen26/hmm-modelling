from src.base import BaseTransition 
import jax.numpy as jnp

from src.base.utils import logits_to_transition_matrix 


class StaticTransition(BaseTransition):
    """
    Static transition model for an HMM. The transition matrix does not depend on the covariates at time step t. 

    transition_matrix_: jnp.ndarray is of dim (num_states, num_states - 1) and contains the off-diagonal elements of the transition matrix. 
    """

    def step(self, t: int | None, ys: jnp.ndarray | None, xs: jnp.ndarray | None = None) -> jnp.ndarray:
        """
        computes new transtions logits based on the covariates at time step t. 

        
        :param self: Description
        :param xt: Description
        :return: Description
        :rtype: ndarray
        """
        return self.transition_logits 
    
    def transition_matrix(self, t: int | None = None, ys: jnp.ndarray | None = None, xs: jnp.ndarray | None = None, dt: float | None = None) -> jnp.ndarray:
        """
        Builds the transition matrix for the observation at index `t`.

        The matrix is time-homogeneous and covariate-free, so both `t` and the
        waiting time `dt` are ignored; they are accepted to keep one signature
        across every transition.

        :return: transition matrix of dim (num_states, num_states)
        """
        logits = self.step(t, ys, xs)
        return logits_to_transition_matrix(logits)
    

