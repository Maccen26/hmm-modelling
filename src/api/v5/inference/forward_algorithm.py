from src.api.v5.base import BaseInference
from src.api.v5.hmm.params import Params
import jax.numpy as jnp
#from src.api.v5.inference.forward_outout import ForwardOutput
from typing import Any
import jax
from pydantic import BaseModel




class ForwardAlgorithm(BaseInference):


    def run_discrete(self, 
                    params: Params, 
                    initial_dist: jnp.ndarray, 
                    ys: jnp.ndarray, 
                    ts: jnp.ndarray, 
                    xs: jnp.ndarray | None = None, 
                    mask: jnp.ndarray | None = None
                    ) -> Any:
        """
        Run the forward algorithm on a single sequence of observations.
        """ 

        Gammas = params.transition_matrices(ts, ys, xs)  # (T, num_states, num_states)
        gs = hmm_params.densities(indices, ys, xs, ts)                # (T, 1, num_states)
        if mask is not None:
            gs = jnp.where(mask.reshape((-1,) + (1,) * (gs.ndim - 1)), gs, 1.0)

        carry_final, outputs = jax.lax.scan(self.step, carry_pre, (Gammas, gs))
        return self.postprocess(carry_pre, carry_final, outputs)

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

    def postprocess(self, carry_0, carry_final, outputs) -> ForwardOutput:
        utt, ft, ut = outputs
        return ForwardOutput(ft=ft, utt=utt, ut=ut)
