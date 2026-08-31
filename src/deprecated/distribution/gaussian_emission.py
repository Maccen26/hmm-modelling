import equinox as eqx
import jax
import jax.numpy as jnp

from src.deprecated.distribution.utils import _gaussian_pdf 

class GaussianEmission(eqx.Module):
    mu: jax.Array
    log_sigma: jax.Array
    def __init__(self, mu, log_sigma):
        self.mu = mu
        self.log_sigma = log_sigma #We need to ensure that sigma is positive, so we can parameterize it as log_sigma and then exponentiate it when we need the actual sigma value. 

    @property
    def sigma(self):
        return jnp.exp(self.log_sigma)   

    def density(self, x):
        return _gaussian_pdf(x, self.mu, self.sigma)  
    


class GaussianEmissionExample(GaussianEmission): 
    def __init__(self, mu, log_sigma):
        super().__init__(mu, log_sigma) 


    def density(self, x):
        """Compute Gaussian density for numerical stability."""
        mu = jnp.concatenate([jnp.array([400]), self.mu])  # Add a fixed mean for the first state
        return _gaussian_pdf(x, mu, self.sigma) 
        

    
    

