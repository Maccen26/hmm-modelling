import jax.numpy as jnp 
from jax import lax
import equinox as eqx
import numpy as np


from src.deprecated.base.transition import Transition
from src.deprecated.base.emmision import Emission
from jax.scipy.stats import norm
import jax

class HMM(eqx.Module):
    transition: Transition
    emission: Emission

    def __init__(self, transition_logits: jnp.ndarray, initial_state_dist: jnp.ndarray | None = None, ):
        self._validate_transition_logits(transition_logits) 

        self.transition = Transition(transition_logits, initial_state_dist)
        self.emission = Emission()  


    
    def _validate_transition_logits(self, transition_logits):
        m, n = transition_logits.shape 
        if (n != (m - 1)):
            raise ValueError(f"Expected transition_logits to have shape (m, m-1), but got {transition_logits.shape}") 
        

    def forward(self, y, x=None):
        """
        implements the forward algorithm for HMMs. 
        x is of dim (T, covariestes_dim) 
        y is of dim (T, obs_dim) 

        where T is the sequence length.
        """

        ut0 = self.transition.u0()

        if x is None:
            _, (Ut, ft, Utt) = lax.scan(lambda carry, yt: self.step(carry, yt), ut0, y)
        else:
            _, (Ut, ft, Utt) = lax.scan(lambda carry, yx: self.step(carry, yx[0], yx[1]), ut0, (y, x)) 
        return Ut, ft, Utt 
    
    def step(self, ut_prev, yt, xt = None): 

        Gamma = self.transition_matrix(xt) 

        u_t = ut_prev @ Gamma 

        g_t = self.density(yt, xt) 

        f_t = jnp.sum(u_t * g_t) 

        u_tt = u_t * g_t / f_t

        return u_tt, (u_tt, f_t, u_t) 
    
    def transition_matrix(self, xt = None): 
        return self.transition.transition_matrix(xt) 
    def density(self, yt, xt = None): 
        return self.emission.density(yt, xt)
    
    def cdf(self, y, x=None):
        """
        computes the cdf of the HMM. 
        x is of dim (T, covariestes_dim) 
        y is of dim (T, obs_dim) 

        where T is the sequence length.
        """
        state_dist = self.compute_all_state_distributions(y, x)
        state_cdf = jnp.sum(self.emission.cdf(y, x) * state_dist, axis=-1)
        return state_cdf 
    
    def compute_all_state_distributions(self, y, x=None):
        ut0 = self.transition.u0()
        state_distributions = np.zeros((y.shape[0], ut0.shape[0]))

        state_distributions[0] = ut0

        for t in range(1, y.shape[0]):
            if x is None:
                ut0, _ = self.step(ut0, y[t]) 
            else:
                ut0, _ = self.step(ut0, y[t], x[t]) 

            state_distributions[t] = ut0 

        return jnp.array(state_distributions)
    
    def pseudo_residuals(self, y, x=None):
        marginal_cdf = self.cdf(y, x)
        pseudo_residuals = norm.ppf(marginal_cdf)
        return pseudo_residuals
    
    def log_likelihood(self, y, x=None):
        _, ft, _ = self.forward(y, x)
        log_likelihood = jnp.sum(jnp.log(ft))
        return log_likelihood
    

    def __repr__(self):
        # iterate over the class fields and print their names and values 
        field_str = ""
        for field_name in self.__dict__:
            field_value = getattr(self, field_name)
            field_str += f"{field_name}: {field_value}, "
        
        class_name = self.__class__.__name__  
        return f"{class_name}({field_str.rstrip(', ')})" 

            





    
    def filter_spec(self):
        
        spec = jax.tree_util.tree_map(eqx.is_inexact_array, self)

        spec = eqx.tree_at(
              lambda m: m.transition.initial_state_dist,
              spec,
              replace=False
          )
        return spec

    






















