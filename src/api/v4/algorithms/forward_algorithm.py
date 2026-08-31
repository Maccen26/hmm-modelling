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



    def step_logspace(self, hmm_params: Any, carry: Any, t: int, ys: jnp.ndarray, xs: jnp.ndarray | None = None) -> Any:
        ut_prev = carry
        Gamma = hmm_params.transition_matrix(t, ys, xs)  # shape (num_states, num_states)
        log_u_t = jnp.log(ut_prev @ Gamma)
        log_g_t = jnp.log(hmm_params.density(t, ys, xs))  # shape (1, num_states)



        #

    def step(self, hmm_params: Any, carry: Any, t: int, ys: jnp.ndarray, xs: jnp.ndarray | None = None) -> Any:
        ut_prev = carry
        #tol = 1e-10
        Gamma = hmm_params.transition_matrix(t, ys, xs)  # shape (num_states, num_states)
        u_t = ut_prev @ Gamma
        g_t = hmm_params.density(t, ys, xs)  # shape (1, num_states)
        #Cap g_t to avoid numerical issues
        #g_t = jnp.clip(g_t, a_min=1e-10, a_max=1e10)
        f_t = jnp.sum(u_t * g_t)
        
        #log_f_t = jnp.log(f_t)
        #log_utt = jnp.log(u_t).flatten() + jnp.log(g_t).flatten() - log_f_t
        #u_tt = jnp.exp(log_utt).reshape(1, -1)  # shape (1, num_states)
        #clip f_t to avoid numerical issues
        f_t = jnp.clip(f_t, a_min=1e-10) 

        #u = u_t * g_t 

        u_tt = u_t * g_t / f_t


        return u_tt, (u_tt, f_t, u_t) 
    

    def postprocess(self, carry_0, carry_final, outputs) -> ForwardOutput:
        utt, ft, ut = outputs
        return ForwardOutput(ft=ft, utt=utt, ut=ut)
    
