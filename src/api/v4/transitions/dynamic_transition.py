from matplotlib.pylab import beta
#from src.base import BaseTransition 
import jax.numpy as jnp
import jax 
from src.base.utils import logits_to_transition_matrix 
import equinox as eqx

class DynamicTransition:
    """
    Static transition model for an HMM. The transition matrix does not depend on the covariates at time step t. 

    The transition matrix is computed dynamically based on the covariates at each time step.

    transition_matrix_: jnp.ndarray is of dim (num_states, num_states - 1) and contains the off-diagonal elements of the transition matrix. 
    """
    beta: jax.Array 
    transition_logits: jax.Array
    num_states: int = eqx.field(static=True)


    def __init__(self, transition_logits, beta):
        self.transition_logits = jnp.asarray(transition_logits, dtype=float)
        self.beta = jnp.asarray(beta, dtype=float)
        self.num_states = transition_logits.shape[0] + 1  #transition_logits is of shape (num_states, num_states - 1)

        if (self.beta.shape != self.transition_logits.shape):
            raise ValueError(f"beta and transition_logits must have the same shape. Got beta shape: {self.beta.shape}, transition_logits shape: {self.transition_logits.shape}") 

    def step(self, t: int, xs: jnp.ndarray) -> jnp.ndarray:
        """
        computes new transtions logits based on the covariates at time step t. 

        
        :param self: Description
        :param xt: Description
        :return: Description
        :rtype: ndarray
        """

        xt = xs[t, :].flatten() #Making it 1D array. 
        tensor = self.beta * xt[:, None, None] #Making it 3D array. Broadcasting each covarites over the beta matrix making it a tensor of shape (num_covariates, num_states, num_states - 1).
        transition_logits = self.transition_logits + tensor.sum(axis=0) #Adding the covariate effect to the transition logits.

        return transition_logits
    
    def transition_matrix(self, t:int, xs: jnp.ndarray) -> jnp.ndarray: 
        """
        Builds the transition matrix at time step t given the covariates at time step t.
        
        :param xt: covarites at time step t. 

        :return: transition matrix at time step t of dim (num_states, num_states) 
        """
        logits = self.step(t, xs)
        return logits_to_transition_matrix(logits)
    

