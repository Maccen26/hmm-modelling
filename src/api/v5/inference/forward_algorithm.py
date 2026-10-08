from src.api.v5.base import BaseInference
from src.api.v5.hmm.params import Params
import jax.numpy as jnp
#from src.api.v5.inference.forward_outout import ForwardOutput
from typing import Any, Callable
import jax 
from dataclasses import dataclass




class ForwardAlgorithm(BaseInference):


    def run(self, 
            params: Params, 
            initial_dist: jnp.ndarray, 
            ys: jnp.ndarray, 
            xs: jnp.ndarray | None = None, 
            ) -> Any:
        """
        Run the forward algorithm on a single sequence of observations.
        """ 
        N = len(ys) 
        Gammas = params.transition_matrices(N=N, xs=xs)  # (T, num_states, num_states)
        gs = params.densities(ys=ys)                # (T, 1, num_states)
        _, outputs = jax.lax.scan(f=self.step, init=initial_dist, xs=(Gammas, gs))
        
        return outputs

    def step(self, carry: Any, step_input: Any) -> Any:
        """One forward recursion step over a precomputed (Gamma, density) pair."""
        
        Gamma, g_t = step_input
        ut_prev = carry

        u_t = ut_prev @ Gamma
        f_t = jnp.sum(u_t * g_t)

        # To do: Make 1 if f_t is zero. This results in Density being zero, which is
        # correct. But we cannot divide by zero
        f_t = jnp.clip(f_t, a_min=1e-10)

        u_tt = u_t * g_t / f_t

        return u_tt, (u_tt, f_t, u_t)
