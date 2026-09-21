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


    def run(self, hmm_params: Any, carry_pre: Any, ys: jnp.ndarray, ts: jnp.ndarray | None = None, xs: jnp.ndarray | None = None) -> Any:
        """
        Run the forward algorithm over a sequence using jax.lax.scan.

        The transition matrices for every time step are precomputed in a single
        batched call (letting the transition dedup expensive per-step work), and the
        scan then indexes into them. The scan iterates over observation indices, so
        the emission density is always read at the correct observation `ys[i]` —
        independent of the waiting-time values carried in `ts`.
        """
        self._validate_inputs(hmm_params, ys, ts, xs, carry_pre)

        if ts is None:
            ts = jnp.arange(len(ys))

        Gammas = hmm_params.transition_matrices(ts, ys, xs)  # (T, num_states, num_states)
        indices = jnp.arange(len(ys))

        def scan_fn(carry, step_input):
            i, Gamma = step_input
            # `ts` (the full waiting-time array) is threaded so a continuous-time
            # emission can read the gap ts[i]; time-independent emissions ignore it.
            return self.step(hmm_params, carry, i, ys, xs, Gamma, ts)

        carry_final, outputs = jax.lax.scan(scan_fn, carry_pre, (indices, Gammas))
        return self.postprocess(carry_pre, carry_final, outputs)

    def step(self, hmm_params: Any, carry: Any, t: float, ys: jnp.ndarray, xs: jnp.ndarray | None = None, Gamma: jnp.ndarray | None = None, ts: jnp.ndarray | None = None) -> Any:
        ut_prev = carry

        if Gamma is None:
            Gamma = hmm_params.transition_matrix(t, ys, xs)  # shape (num_states, num_states)
        u_t = ut_prev @ Gamma
        g_t = hmm_params.density(t, ys, xs, ts)  # shape (1, num_states)

        f_t = jnp.sum(u_t * g_t)
        
        #To do: Make 1 if f_t is zero. This results in Density being zero, which is correct. But we cannot divide by zero
        f_t = jnp.clip(f_t, a_min=1e-10) 

        u_tt = u_t * g_t / f_t


        return u_tt, (u_tt, f_t, u_t) 
    

    def postprocess(self, carry_0, carry_final, outputs) -> ForwardOutput:
        utt, ft, ut = outputs
        return ForwardOutput(ft=ft, utt=utt, ut=ut)
    
