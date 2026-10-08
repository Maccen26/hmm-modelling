from src.api.v5.hmm.transitions.base_transition import BaseTransition 
import jax.numpy as jnp

from src.api.v5.base.utils import logits_to_transition_matrix 


class StaticTransition(BaseTransition):
    """
    Static transition model for an HMM. The transition matrix does not depend on the covariates at time step t. 

    transition_matrix_: jnp.ndarray is of dim (num_states, num_states - 1) and contains the off-diagonal elements of the transition matrix. 
    """

    def step(self, xt: jnp.ndarray | None = None) -> jnp.ndarray:
        """
        Returns the transition logits. They are covariate-free, so `xt` is ignored.

        :param xt: covariate row at one observation (ignored).
        :return: transition logits of shape (num_states, num_states - 1).
        """
        return self.transition_logits

    def transition_matrix(self, xt: jnp.ndarray | None = None) -> jnp.ndarray:
        """
        Builds the transition matrix.

        The matrix is time-homogeneous and covariate-free, so `xt` is ignored; it
        is accepted to keep one signature across every transition.

        :return: transition matrix of dim (num_states, num_states)
        """
        logits = self.step(xt)
        return logits_to_transition_matrix(logits)
    

