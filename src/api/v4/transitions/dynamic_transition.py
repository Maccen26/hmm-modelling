from dataclasses import fields
from typing import Any
from matplotlib.pylab import beta
#from src.base import BaseTransition 
import jax.numpy as jnp
import jax 
from src.base.utils import logits_to_transition_matrix 
import equinox as eqx
class DynamicTransition(eqx.Module):
    """
    Static transition model for an HMM. The transition matrix does not depend on the covariates at time step t. 

    The transition matrix is computed dynamically based on the covariates at each time step.

    transition_matrix_: jnp.ndarray is of dim (num_states, num_states - 1) and contains the off-diagonal elements of the transition matrix. 
    """
    beta: jax.Array 
    transition_logits: jax.Array


    def __init__(self, transition_logits, beta):
        self.transition_logits = jnp.asarray(transition_logits, dtype=float)
        self.beta = jnp.asarray(beta, dtype=float)

        b,n,m = self.beta.shape
        if ((n,m) != self.transition_logits.shape):
            raise ValueError(f"beta and transition_logits must have the same shape. Got beta shape: {self.beta.shape}, transition_logits shape: {self.transition_logits.shape}") 

    def step(self, t: int, xs: jnp.ndarray, ys: jnp.ndarray | None = None) -> jnp.ndarray:
        """
        computes new transtions logits based on the covariates at time step t. 
        
        :param self: Description
        :param xt: Description
        :param ys: Description
        :return: Description
        :rtype: ndarray
        """

        xt = xs[t, :].flatten() #Making it 1D array. 
        tensor = self.beta * xt[:, None, None] #Making it 3D array. Broadcasting each covarites over the beta matrix making it a tensor of shape (num_covariates, num_states, num_states - 1).
        transition_logits = self.transition_logits + tensor.sum(axis=0) #Adding the covariate effect to the transition logits.

        return transition_logits
    
    def transition_matrix(self, t:int, xs: jnp.ndarray, ys: jnp.ndarray | None = None) -> jnp.ndarray: 
        """
        Builds the transition matrix at time step t given the covariates at time step t.
        
        :param xt: covarites at time step t. 

        :return: transition matrix at time step t of dim (num_states, num_states) 
        """
        logits = self.step(t, xs, ys=None) #Get the transition logits at time step t.
        return logits_to_transition_matrix(logits) 
    
    def base_transition_matrix(self) -> jnp.ndarray:
        """
        Returns the base transition matrix without any covariate effects. 
        This is useful for computing the stationary distribution of the HMM. 
        """
        return logits_to_transition_matrix(self.transition_logits)

    def __iter__(self) -> Any:
        """Make the class iterable with names. This is useful for the forward and backward algorithms, where we need to iterate over the states and compute the transition and emission probabilities."""
        return ((f.name, getattr(self, f.name)) for f in fields(self))

