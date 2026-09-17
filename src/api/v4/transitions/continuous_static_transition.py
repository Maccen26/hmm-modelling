from src.base import BaseTransition 
import jax.numpy as jnp

from src.base.utils import logits_to_transition_matrix_continuous, transtion_matrix_to_logits_continuous, get_Q_from_logits

class ContinuousStaticTransition(BaseTransition):
    """
    Static transition model for an HMM. The transition matrix does not depend on the covariates at time step t.

    transition_matrix_: jnp.ndarray is of dim (num_states, num_states - 1) and contains the off-diagonal elements of the transition matrix.

    """

    @classmethod
    def from_params(cls, transition_matrix):
        transition_logits =  transtion_matrix_to_logits_continuous(transition_matrix, 0)
        return cls(transition_logits)

    def step(self, t: int | None, ys: jnp.ndarray | None, xs: jnp.ndarray | None = None) -> jnp.ndarray:
        """
        computes new transtions logits based on the covariates at time step t. 

        
        :param self: Description
        :param xt: Description
        :return: Description
        :rtype: ndarray
        """

        return self.transition_logits 
    
    def transition_matrix(self, t:int|None = None, ys: jnp.ndarray | None = None, xs: jnp.ndarray | None = None) -> jnp.ndarray: 
        """
        Builds the transition matrix at time step t given the covariates at time step t.
        
        :param xt: covarites at time step t. 

        :return: transition matrix at time step t of dim (num_states, num_states) 
        """

        logits = self.step(t, ys, xs)
        return logits_to_transition_matrix_continuous(logits, t) # type: ignore

    def get_Q(self): 
        return get_Q_from_logits(self.transition_logits)

        
    

