import jax.numpy as jnp 
from src.base.base_transition import BaseTransition
from src.base.base_emission import BaseEmission
from abc import ABC, abstractmethod
import equinox as eqx
from typing import Iterator, Tuple
from dataclasses import fields 
import jax

class BaseHMM(ABC, eqx.Module):
    """
    Base class for a Hidden Markov Model (HMM) that combines a transition model and an emission model. 

    The HMM class should focus on combining the transition and emission models to data (y,x) and compute the different states
    troughout the sequence.
    """
    emission: BaseEmission
    transition: BaseTransition

    @abstractmethod
    def transition_matrix(self, t:int| None = None, ys: jnp.ndarray | None = None, xs: jnp.ndarray | None = None) -> jnp.ndarray:  
        """
        Builds the transition matrix at time step t given the covariates at time step t.
        
        :param xt: covarites at time step t. 

        :return: transition matrix at time step t of dim (num_states, num_states) 
        """
        ... 
    
    @abstractmethod
    def density(self,  t:int, ys: jnp.ndarray, xs: jnp.ndarray | None = None) -> jnp.ndarray:
        """
        y is the observation at time step t. 
        x is the covariates at time step t. 
        Returns the emission density p(y_t | z_t, x_t) at time step t with dimensions (num_states,).
        """
        ... 
    
    @abstractmethod
    def cdf(self, t:int, ys: jnp.ndarray, xs: jnp.ndarray | None = None) -> jnp.ndarray:
        """
        y is the observation at time step t. 
        x is the covariates at time step t. 
        Returns the emission cdf P(Y_t <= y | z_t, x_t) at time step t with dimensions (num_states,).
        """
        ...  

    def __iter__(self) -> Iterator:
        """Make the class iterable with names. This is useful for the forward and backward algorithms, where we need to iterate over the states and compute the transition and emission probabilities."""
        return((f.name, getattr(self, f.name)) for f in fields(self))
    
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, BaseHMM):
            return False
        if self.__class__.__name__ != value.__class__.__name__:
            return False
        return all(getattr(self, f.name) == getattr(value, f.name) for f in fields(self))
    

    def update_param(self, param_name: str, new_value: float|jax.Array, index: Tuple|float|None) -> 'BaseHMM':
        """Returns a new HMMParams object with the updated parameter value
        Only updates low level params as Jax Arrays
        """ 

        val = jnp.asarray(new_value, dtype=float)  # Cast to double for numerical stability

        new_components_list = {}

        for name, component in self: 
            if (hasattr(component, param_name)): 
                component = component.update_param(param_name, val, index)  
            new_components_list[name] = component
        return self.__class__(**new_components_list)
        



    

    

    


