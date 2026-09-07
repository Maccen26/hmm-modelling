from src.base import BaseInference
from src.api.v4.hmm_models.hmm_params import HMMParams 
import jax.numpy as jnp
from src.api.v4.algorithms.forward_outout import ForwardOutput
from typing import Any
from jax import lax 
import jax


@jax.jit
def normalize_probs(probs: jax.Array) -> jax.Array:
    total = jnp.sum(probs)
    probs = probs / total
    return probs

class ForwardAlgorithm(BaseInference):


    def step(self, hmm_params: Any, carry: Any, t: int, ys: jnp.ndarray, xs: jnp.ndarray | None = None) -> Any:
        ut_prev = carry

        Gamma = hmm_params.transition_matrix(t, ys, xs)  # shape (num_states, num_states)
        u_t = ut_prev @ Gamma
        g_t = hmm_params.density(t, ys, xs)  # shape (1, num_states)

        f_t = jnp.sum(u_t * g_t)
        
        #To do: Make 1 if f_t is zero. This results in Density being zero, which is correct. But we cannot divide by zero
        f_t = jnp.clip(f_t, a_min=1e-10) 

        u_tt = u_t * g_t / f_t


        return u_tt, (u_tt, f_t, u_t) 
    

    def postprocess(self, carry_0, carry_final, outputs) -> ForwardOutput:
        utt, ft, ut = outputs
        return ForwardOutput(ft=ft, utt=utt, ut=ut)
    
