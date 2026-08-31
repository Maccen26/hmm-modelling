
import jax
from jax import lax
import jax.numpy as jnp


@jax.jit
def recursive_filter(Utt0, g, Gamma):
    def scan_fn(Utt_prev, g_i):
        Ut_i = Utt_prev @ Gamma
        v = Ut_i * g_i
        ft_i = jnp.sum(v)
        Utt_i = v / ft_i
        return Utt_i, (Ut_i, ft_i, Utt_i)

    _, (Ut, ft, Utt) = lax.scan(scan_fn, Utt0, g)
    return Ut, ft, Utt


@jax.jit
def recursive_filter_dynamic(Utt0, g, Gammas):
    """Like recursive_filter but with a different transition matrix at each time step.

    Gammas: (T-1, num_states, num_states) array of transition matrices.
    """
    def scan_fn(Utt_prev, inputs):
        g_i, Gamma_i = inputs
        Ut_i = Utt_prev @ Gamma_i
        v = Ut_i * g_i
        ft_i = jnp.sum(v)
        Utt_i = v / ft_i
        return Utt_i, (Ut_i, ft_i, Utt_i)

    _, (Ut, ft, Utt) = lax.scan(scan_fn, Utt0, (g, Gammas))
    return Ut, ft, Utt


def to_logit_matrix(transition_matrix):
    return jnp.log(transition_matrix)