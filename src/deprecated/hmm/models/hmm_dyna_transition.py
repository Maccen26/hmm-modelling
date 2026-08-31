from src.deprecated.hmm.models.base import BaseHMM
from src.deprecated.distribution.base import BaseGaussianEmission
from src.deprecated.hmm.transitions.base import BaseTransition
from src.deprecated.hmm.utils import recursive_filter_dynamic
import jax.numpy as jnp
import jax
import functools



@functools.partial(jax.jit, static_argnums=(2,))
def _to_dynamic_transition_matrix(off_diag_logits, diag_values, num_states):
    m = num_states
    Gamma = jnp.zeros((m, m))
    rows, cols = jnp.where(~jnp.eye(m, dtype=bool), size=m * (m - 1))
    Gamma = Gamma.at[rows, cols].set(jnp.exp(off_diag_logits))
    Gamma = Gamma.at[jnp.diag_indices(m)].set(jnp.exp(diag_values))
    Gamma = Gamma.T / Gamma.sum(axis=0)
    return Gamma



class GaussianEmission(BaseGaussianEmission):
    def __init__(self, mu, log_sigma):
        super().__init__(mu, log_sigma) 
    
class DynamicTransition(BaseTransition):
    beta: jax.Array  # shape (p, m)
    def __init__(self, transition_logits, beta: jnp.ndarray, num_states=None):
        super().__init__(transition_logits, num_states=num_states)
        self.beta = beta

    def transition_matrix(self, x=None):
        """Compute transition matrix for a single covariate vector x of shape (p,)."""
        beta = self.beta.reshape(self.num_states, -1)
        diag_values = (beta @ x).flatten()
        return _to_dynamic_transition_matrix(self.transition_logits, diag_values, self.num_states)

    def all_transition_matrices(self, X):
        """Compute transition matrices for all time steps. X shape (T, p) -> (T, m, m)."""
        return jax.vmap(self.transition_matrix)(X)






class HMMDynaTransition(BaseHMM):
    def __init__(self, transition_logits, mu, log_sigma, beta, num_states=None):
        super().__init__(transition_logits, mu, log_sigma, num_states=num_states)
        self.emission_distributions = GaussianEmission(mu, log_sigma)
        self.transition = DynamicTransition(transition_logits, beta=beta, num_states=num_states)

    def forward(self, y, x=None):
        g = self.emission_distributions.density(y, x)  # (T, num_states)

        # Compute all T transition matrices: (T, m, m)
        Gammas = self.transition.all_transition_matrices(x)

        # --- t=0: initialize ---
        ut0 = self.transition.init_stationary_distribution(x=x[0])
        g0 = g[0]
        ft0 = jnp.sum(ut0 * g0)
        utt0 = ut0 * g0 / ft0

        # --- t=1..T-1: scan with time-varying Gamma ---
        g_rest = g[1:]
        Gammas_rest = Gammas[1:]  # transition matrices for t=1..T-1

        Ut, ft_rest, Utt = recursive_filter_dynamic(utt0, g_rest, Gammas_rest)

        # --- Concatenate t=0 with t=1..T-1 ---
        ut = jnp.concatenate([ut0[None, :], Ut], axis=0)
        u_norm = jnp.concatenate([utt0[None, :], Utt], axis=0)
        ft = jnp.concatenate([ft0[None], ft_rest], axis=0)

        return ut, u_norm, ft 