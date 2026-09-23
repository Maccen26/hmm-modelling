import jax.numpy as jnp
import equinox as eqx
from abc import ABC, abstractmethod
from src.base.utils import logits_to_transition_matrix, transition_matrix_to_logits
from typing import Iterator, Tuple
from dataclasses import fields
import jax 
class BaseTransition(eqx.Module, ABC):
    """
    Base class for the transition component of an HMM.
    """

    transition_logits: jnp.ndarray


    def __init__(self, transition_logits):
        self.transition_logits = jnp.asarray(transition_logits, dtype=float)  # Cast to double for numerical stability

    @classmethod
    def from_params(cls, transition_matrix):
        transition_logits =  transition_matrix_to_logits(transition_matrix)
        return cls(transition_logits)
    
    @abstractmethod
    def transition_matrix(self, t: int | None = None, ys: jnp.ndarray | None = None, xs: jnp.ndarray | None = None, dt: float | None = None) -> jnp.ndarray:
        """
        Builds the transition matrix for the observation at index `t`.

        `t` and `dt` mean different things and must not be conflated:

        :param t: the *observation index*, used to look up the covariate row xs[t].
            Discrete time-homogeneous transitions ignore it.
        :param ys: observation sequence.
        :param xs: covariate sequence of shape (T, num_covariates).
        :param dt: the *waiting time* since the previous observation, used by the
            continuous-time transitions in expm(Q * dt). Discrete transitions ignore
            it; continuous ones default it to a unit step.

        :return: transition matrix of dim (num_states, num_states)
        """
        ...


    def transition_matrices(self, indices: jnp.ndarray, ts: jnp.ndarray, ys: jnp.ndarray | None = None, xs: jnp.ndarray | None = None) -> jnp.ndarray:
        """
        Builds one transition matrix per observation in a single batched call.

        :param indices: observation indices, i.e. jnp.arange(T) — used for covariate
            lookup.
        :param ts: per-observation waiting times, parallel to `indices` — used by the
            continuous-time transitions.

        Returns an array of shape (T, num_states, num_states). The default
        implementation just vmaps the per-step `transition_matrix`; subclasses can
        override it with a cheaper batched computation.
        """
        return jax.vmap(lambda i, dt: self.transition_matrix(t=i, ys=ys, xs=xs, dt=dt))(indices, ts)

    @abstractmethod
    def step(self, t: int | None, ys: jnp.ndarray | None, xs: jnp.ndarray | None) -> jnp.ndarray:
        """
        computes new transtions logits based on the covariates at time step t. 
        Return dim is (num_states, num_states - 1) and contains the off-diagonal elements of the transition matrix.

        :type t: int
        :param t: time step
        :type ys: jnp.ndarray
        :param ys: observation sequence
        :type xs: jnp.ndarray | None
        :param xs: covariate sequence (optional)
        :return: transition logits for time step t
        :rtype: jnp.ndarray
        """
        ...

    def __iter__(self) -> Iterator:
        return ((f.name, getattr(self, f.name)) for f in fields(self))
    

    def __eq__(self, value: object) -> bool:
        is_equal = True
        
        is_equal = is_equal and isinstance(value, BaseTransition)
        is_equal = is_equal and (self.__class__.__name__ == value.__class__.__name__)

        for f in fields(self):
            a = getattr(self, f.name)
            b = getattr(value, f.name) 
            is_equal = is_equal and jnp.allclose(a, b, atol=1e-6, rtol=1e-5) 

        return bool(is_equal)
    
    def update_param(self, param_name: str, new_value: jax.Array, index: Tuple|float|None) -> 'BaseTransition': 
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





    
    
    

