from dataclasses import dataclass 
import jax.numpy as jnp
from typing import List


@dataclass(frozen=True)
class StateResults:
    """Results class for each time step"""
    utt: List[jnp.ndarray]
    ut: List[jnp.ndarray] 
    time_index: jnp.ndarray 
    pseudo_residuals: jnp.ndarray

    def __init__(self, utt: List[jnp.ndarray], ut: List[jnp.ndarray], time_index: jnp.ndarray, pseudo_residuals: jnp.ndarray):
        object.__setattr__(self, 'utt', [u.flatten() for u in utt])  # flatten each array in the list
        object.__setattr__(self, 'ut', [u.flatten() for u in ut])  # flatten each array in the list
        object.__setattr__(self, 'time_index', time_index)
        object.__setattr__(self, 'pseudo_residuals', pseudo_residuals)

@dataclass(frozen=True)
class HMMResults:
    """Results class for the HMM model"""
    convergence: bool
    log_likelihood: float
    num_params: int

