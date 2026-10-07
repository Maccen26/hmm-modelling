import numpy as np
import jax.numpy as jnp
import equinox as eqx
from abc import ABC, abstractmethod
from src.base.utils import transition_matrix_to_logits
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
    def transition_matrix(self, xt: jnp.ndarray | None = None) -> jnp.ndarray:
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
    
    def transition_matrices(self, N: int, xs: jnp.ndarray | None = None) -> jnp.ndarray:
        if (xs is None): 
            Gamma = self.transition_matrix()
            return jnp.broadcast_to(Gamma, (N, *Gamma.shape))
        # np.unique (not jnp) runs at trace time on the concrete xs, so this stays jittable
        xs_uniq, inverse = np.unique(np.asarray(xs), axis=0, return_inverse=True)
        transition_matrices_uniq = jax.vmap(self.transition_matrix)(jnp.asarray(xs_uniq))
        transition_matrices = transition_matrices_uniq[inverse.reshape(-1)]
        return transition_matrices



    def __iter__(self) -> Iterator:
        return ((f.name, getattr(self, f.name)) for f in fields(self))
    

    def __eq__(self, value: object) -> bool:
        # Return early: a different class may not have the same fields
        if not isinstance(value, BaseTransition) or self.__class__.__name__ != value.__class__.__name__:
            return False

        is_equal = True
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





    
    
    

