import equinox as eqx
import jax.numpy as jnp
from abc import ABC, abstractmethod

class BaseOutput(eqx.Module, ABC):
    """
    Base class for HMM output. This class is meant to be inherited by specific output classes that implement the likelihood and negative log-likelihood functions. 
    """
    @abstractmethod
    def log_likelihood(self) -> jnp.ndarray:
        """
        compute the log-likelihood of the output of the given algorithm, 
        """
        ... 

