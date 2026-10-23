# The backward algorithms compute the probs for future observations given the current state.
from src.api.v5.hmm.params import Params
import jax.numpy as jnp
import jax 
from typing import Any
from .backward import BackwardAlgorithm
from .forward_algorithm import ForwardAlgorithm



class ForwardBackwardAlgorithm:
    """
    Forward-backward algorithm for computing the smoothed state distributions in a Hidden Markov Model (HMM).

    Combines the forward and backward algorithms to compute the smoothed state distributions, which represent the probability of being in each state at each time step given all observations.
    """

    @classmethod
    def run(cls, 
            params: Params, 
            initial_dist: jnp.ndarray, 
            ys: jnp.ndarray, 
            xs: jnp.ndarray | None = None, 
            prior_densities: jnp.ndarray | None = None
            ) -> jnp.ndarray:
        """
        Run the forward-backward algorithm on a single sequence of observations.
        Returns (s_t) for each time step t, where:
        - s_t: the smoothed state distribution at time t
        """ 
        if (prior_densities is None):
            prior_densities = jnp.ones((len(ys), 1))  # (T, 1)

        # Run forward algorithm
        forward_outputs = ForwardAlgorithm.run(params=params, initial_dist=initial_dist, ys=ys, xs=xs, prior_densities=prior_densities)
        u_tt, _, _ = forward_outputs
        u_tt = jnp.squeeze(u_tt, axis=1)  # (T, num_states) 

        # Run backward algorithm
        backward_outputs = BackwardAlgorithm.run(params=params, ys=ys, xs=xs, prior_densities=prior_densities)
        b_tt = backward_outputs 



        # Compute smoothed state distributions
        s_ts = u_tt * b_tt
        s_ts = s_ts / jnp.sum(s_ts, axis=-1)[:, None]  # normalize

        return s_ts