import equinox as eqx
from abc import ABC, abstractmethod 
import jax
import jax.numpy as jnp
from typing import Any, Iterator, Tuple
from dataclasses import fields

class BaseEmission(eqx.Module, ABC): 
    """
    Base class for the emission component of an HMM.
    """

    @abstractmethod
    def density(self, t:int, ys: jnp.ndarray, xs: jnp.ndarray | None = None) -> jnp.ndarray:
        """
        ys is the observations sequence. 
        xs is the covariates sequence.
        t is the time step. 
        Returns the emission density p(y_t | z_t, x_t) at time step t with dimensions (num_states,).
        """
        ...  

    @abstractmethod
    def step(self, t: int, ys: jnp.ndarray, xs: jnp.ndarray | None) -> Any:
        """
        ys is the observations sequence. 
        xs is the covariates sequence.
        t is the time step. 

        Computes the parameters of the emission distribution at time step t given the covariates at time step t. 
        The output of this function will depend on the specific emission distribution used. 
        For example, for a Gaussian emission, this function could return the mean and standard deviation of the Gaussian distribution at time step t. 
        For a Poisson emission, this function could return the rate parameter of the Poisson distribution at time step t. 
        """
        ...
    
    @abstractmethod
    def cdf(self, t:int, ys: jnp.ndarray, xs: jnp.ndarray | None = None) -> jnp.ndarray:
        """
        ys is the observations sequence. 
        xs is the covariates sequence.
        t is the time step. 

        Returns the emission cdf P(Y_t <= y | z_t, x_t) at time step t with dimensions (num_states,).
        """
        ...

    def __iter__(self) -> Any:
        """Make the class iterable with names. This is useful for the forward and backward algorithms, where we need to iterate over the states and compute the transition and emission probabilities."""
        return ((f.name, getattr(self, f.name)) for f in fields(self))
    
    def __eq__(self, value: object) -> bool:
        is_equal = True
        
        is_equal = is_equal and isinstance(value, BaseEmission)
        is_equal = is_equal and (self.__class__.__name__ == value.__class__.__name__)

        for f in fields(self):
            a = getattr(self, f.name)
            b = getattr(value, f.name) 
            is_equal = is_equal and jnp.allclose(a, b, atol=1e-6, rtol=1e-5) 

        return bool(is_equal)
    
    def update_param(self, param_name: str, new_value: jax.Array, index: Tuple|float|None) -> 'BaseEmission': 
        new_param_list = []
        for name, param in self: 
            if name == param_name: 
                if index is not None:
                    new_param = jnp.asarray(param).at[index].set(new_value)   
                else:
                    new_param = new_value
            else: 
                new_param = param 

            new_param_list.append(new_param)

        return self.__class__(*new_param_list) 
 
