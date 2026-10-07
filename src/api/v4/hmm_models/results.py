from dataclasses import dataclass
import jax.numpy as jnp


@dataclass(frozen=True)
class StateResults:
    """Per-observation state results for a fitted HMM.

    `utt` and `ut` are stored as (T, num_states) arrays. They arrive from the
    forward algorithm as (T, 1, num_states) and are reshaped once, rather than
    being unpacked into a Python list of T arrays -- consumers immediately convert
    back to an array anyway, and at T in the thousands the per-row device ops
    dominated the cost of computing them.
    """
    utt: jnp.ndarray
    ut: jnp.ndarray
    time_index: jnp.ndarray
    pseudo_residuals: jnp.ndarray

    def __init__(self, utt: jnp.ndarray, ut: jnp.ndarray, time_index: jnp.ndarray, pseudo_residuals: jnp.ndarray):
        object.__setattr__(self, 'utt', jnp.asarray(utt).reshape(len(time_index), -1))
        object.__setattr__(self, 'ut', jnp.asarray(ut).reshape(len(time_index), -1))
        object.__setattr__(self, 'time_index', time_index)
        object.__setattr__(self, 'pseudo_residuals', pseudo_residuals)


@dataclass(frozen=True)
class HMMResults:
    """Results class for the HMM model"""
    convergence: bool
    log_likelihood: float
    num_params: int
