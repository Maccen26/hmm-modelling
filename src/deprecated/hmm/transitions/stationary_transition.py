import functools
import jax.numpy as jnp
import equinox as eqx
import jax

@functools.partial(jax.jit, static_argnums=(1,))
def _to_transition_matrix(transition_logits, num_states):
    m = num_states
    exp_pars = jnp.exp(transition_logits)
    Gamma = jnp.zeros((m, m))
    Gamma = Gamma.at[jnp.diag_indices(m)].set(1.0)
    rows, cols = jnp.where(~jnp.eye(m, dtype=bool), size=m * (m - 1))
    Gamma = Gamma.at[rows, cols].set(exp_pars)
    Gamma = Gamma.T / Gamma.sum(axis=0)
    return Gamma




class StationaryTransition(eqx.Module):
    transition_logits: jnp.ndarray
    num_states: int = eqx.field(static=True)

    def __init__(self, transition_logits, num_states):
        self.transition_logits = transition_logits
        self.num_states = num_states

    @property
    def transition_matrix(self):
        return _to_transition_matrix(self.transition_logits, self.num_states)

    def init_stationary_distribution(self):
        I = jnp.eye(self.num_states)
        E = jnp.ones((self.num_states, self.num_states))
        e = jnp.ones((self.num_states, 1))
        delta = e.T @ jnp.linalg.inv(I - self.transition_matrix + E)  # (1, num_states)
        return delta.flatten()  # (num_states,) 
    
