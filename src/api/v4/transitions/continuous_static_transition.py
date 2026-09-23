from src.base import BaseTransition
import jax
import jax.numpy as jnp
import numpy as np

from src.base.utils import logits_to_transition_matrix_continuous, transtion_matrix_to_logits_continuous, get_Q_from_logits

class ContinuousStaticTransition(BaseTransition):
    """
    Static transition model for an HMM. The transition matrix does not depend on the covariates at time step t.

    transition_matrix_: jnp.ndarray is of dim (num_states, num_states - 1) and contains the off-diagonal elements of the transition matrix.

    """

    def __init__(self, transition_logits: jnp.ndarray):
        """
        Initializes the ContinuousStaticTransition with the given transition logits.

        :param transition_logits: jnp.ndarray of shape (num_states, num_states - 1) containing the off-diagonal elements of the transition matrix.
        """
        self.transition_logits = transition_logits

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
    
    def transition_matrix(self, t: int | None = None, ys: jnp.ndarray | None = None, xs: jnp.ndarray | None = None, dt: float | None = None) -> jnp.ndarray:
        """
        Builds the transition matrix expm(Q * dt) for a step of waiting time `dt`.

        :param t: ignored — the generator is covariate-free, so no index is needed.
        :param dt: waiting time since the previous observation. Defaults to a unit
            step, which is what the stationary-distribution computation expects.
        :return: transition matrix of dim (num_states, num_states)
        """

        logits = self.step(t, ys, xs)
        dt = 1.0 if dt is None else dt

        return logits_to_transition_matrix_continuous(logits, dt)  # type: ignore

    def transition_matrices(self, indices: jnp.ndarray, ts: jnp.ndarray, ys: jnp.ndarray | None = None, xs: jnp.ndarray | None = None) -> jnp.ndarray:
        """
        Batched transition matrices T_i = expm(Q * ts[i]).

        Since Q is constant across time steps, `expm` only depends on the waiting
        time ts[i] — `indices` is unused. Observed waiting times contain very few
        distinct values, so we compute `expm` once per unique waiting time and gather
        the result back to full length. This is dramatically cheaper than calling
        `expm` once per observation.
        """
        Q = self.get_Q()
        expm = lambda t: jax.scipy.linalg.expm(Q * t.astype(Q.dtype))

        try:
            # `ts` is a concrete constant in the fit path, so we can dedup at trace time.
            ts_np = np.asarray(ts)
        except Exception:
            # `ts` is a tracer (no concrete values available) — fall back to no dedup.
            return jax.vmap(expm)(ts)

        unique_ts, inverse = np.unique(ts_np, return_inverse=True)
        unique_mats = jax.vmap(expm)(jnp.asarray(unique_ts, dtype=Q.dtype))
        return unique_mats[jnp.asarray(inverse.reshape(-1))]

    def get_Q(self):
        return get_Q_from_logits(self.transition_logits)

        
    

