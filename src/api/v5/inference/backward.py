# The backward algorithms compute the probs for future observations given the current state.
from src.api.v5.hmm.params import Params
import jax.numpy as jnp
import jax 
from typing import Any


class BackwardAlgorithm:
    """
    Backward algorithm for computing the backward state distributions in a Hidden Markov Model (HMM).

    Tells us the probability of future observations given the current state. This is useful for smoothing and other inference tasks in HMMs.
    """

    @classmethod
    def run(cls, 
            params: Params, 
            ys: jnp.ndarray, 
            xs: jnp.ndarray | None = None, 
            prior_densities: jnp.ndarray | None = None
            ) -> jnp.ndarray:
        """
        Run the backward algorithm on a single sequence of observations.
        Returns (b_t) for each time step t, where:
        - b_t: the backward state distribution at time t
        """ 
        if (prior_densities is None):
            prior_densities = jnp.ones((len(ys), 1))  # (T, 1)

        N = len(ys) 
        Gammas = params.transition_matrices(N=N, xs=xs)  # (T, num_states, num_states)
        gs = params.densities(ys=ys)[:, 0, :]      # (T, num_states)
        b_last = jnp.ones((gs.shape[-1],)) / gs.shape[-1]  # b_{T-1} = 1, normalized

        # b_t uses Gamma_{t+1} and g_{t+1}, so the scan runs over time steps 1..T-1
        _, outputs = jax.lax.scan(f=cls.step, init=b_last, xs=(cls.reverse_time_series(Gammas[1:]), cls.reverse_time_series(gs[1:]), cls.reverse_time_series(prior_densities[1:])))
        return jnp.concatenate([cls.reverse_time_series(outputs), b_last[None, :]])  # (T, num_states)
    
    @staticmethod
    def reverse_time_series(arr: jnp.ndarray) -> jnp.ndarray:
        """
        Reverse the time series along the first axis.
        """
        return arr[::-1]
    
    @staticmethod
    def step(carry: Any, step_input: Any) -> Any:
        """One backward recursion step over a precomputed (Gamma, density) pair."""
        
        Gamma, g_t_plus_1, prior_dens_plus_1 = step_input
        b_t_plus_1 = carry
        b_t = Gamma @ (g_t_plus_1 * b_t_plus_1 * prior_dens_plus_1)
        b_t = b_t/ jnp.sum(b_t)  # normalize to avoid underflow
        
        return b_t, b_t