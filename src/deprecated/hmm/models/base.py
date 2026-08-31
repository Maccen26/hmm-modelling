import jax.numpy as jnp
from src.deprecated.hmm.utils import recursive_filter 
from src.deprecated.distribution.base import BaseGaussianEmission
from src.deprecated.hmm.transitions.base import BaseTransition

import equinox as eqx 



class BaseHMM(eqx.Module):
    transition: BaseTransition
    emission_distributions: BaseGaussianEmission 

    def __init__(self, transition_logits, mu, log_sigma, num_states=None):
        self.transition = BaseTransition(transition_logits, num_states=num_states)
        self.emission_distributions = BaseGaussianEmission(mu, log_sigma) 

    def forward(self, y : jnp.ndarray, x : jnp.ndarray = None):
        # --- t=0: initialize ---
        g = self.emission_distributions.density(y, x)  # (num_states, T) 

        x0 = x[0] if x is not None else None
        ut0   = self.transition.init_stationary_distribution(x=x0)
        g0    =  g[0]
        ft0   = jnp.sum(ut0 * g0)

        utt0  = ut0 * g0 / ft0


        g_rest = g[1:]

        Ut, ft_rest, Utt = recursive_filter(utt0, g_rest, self.transition.transition_matrix(x))

        # --- Concatenate t=0 with t=1..T-1 ---
        ut     = jnp.concatenate([ut0[None, :],  Ut],  axis=0)  # (T, num_states)
        u_norm = jnp.concatenate([utt0[None, :], Utt], axis=0)  # (T, num_states)
        ft     = jnp.concatenate([ft0[None],     ft_rest], axis=0)  # (T,) 

        return ut, u_norm, ft
 
    def log_likelihood(self, y, x=None):
        _, _, ft = self.forward(y, x)
        return jnp.sum(jnp.log(ft))  
    
    def likelihood(self, y, x=None):
        return jnp.exp(self.log_likelihood(y, x)) 