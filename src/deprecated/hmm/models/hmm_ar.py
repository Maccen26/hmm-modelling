from src.deprecated.hmm.models.base import BaseHMM 
from src.deprecated.distribution.base import BaseGaussianEmission
from src.deprecated.hmm.transitions.base import BaseTransition 
from src.deprecated.distribution.utils import _gaussian_pdf 
import jax.numpy as jnp
import jax




class GaussianARREmission(BaseGaussianEmission):
    phi: jax.Array

    def __init__(self, mu, log_sigma, log_phi):
        super().__init__(mu, log_sigma) 
        self.phi = jnp.exp(log_phi) 

    def density(self, y, x = None):
        """Compute Gaussian density for numerical stability."""
        y_prev = jnp.concatenate([jnp.array([0]), y[:-1]])  # Shifted version of y for AR(1) model
        mu = self.mu[None, :] + self.phi[None, :] * y_prev[:, None] # AR(1) model: mu + phi * x 
        return _gaussian_pdf(y, mu, self.sigma) 
    
class StationaryTransition(BaseTransition):
    def __init__(self, transition_logits, num_states=None):
        super().__init__(transition_logits, num_states=num_states)

class HMMAR(BaseHMM):
    def __init__(self, transition_logits, mu, log_sigma, log_phi, num_states=None):
        super().__init__(transition_logits, mu, log_sigma, num_states=num_states)
        self.emission_distributions = GaussianARREmission(mu, log_sigma, log_phi) 
        self.transition = StationaryTransition(transition_logits, num_states=num_states) 

  




