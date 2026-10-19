from typing import Any
import jax.numpy as jnp
from src.api.v5.hmm.params import Params


def negative_log_likelihood(output: Any, params: Params) -> jnp.ndarray:
    """
    Negative log-likelihood of the observed sequence from the forward algorithm output.

    `output` is the (u_tt, f_t, u_t) tuple returned by ForwardAlgorithm.run, where
    f_t are the per-step likelihood factors p(y_t | y_{1:t-1}).
    """
    _, fts, _ = output
    return -jnp.sum(jnp.log(fts))
