import equinox as eqx
import jax
import jax.numpy as jnp

from src.deprecated.distribution.utils import _gaussian_pdf 

class BaseGaussianEmission(eqx.Module):
    mu: jax.Array
    log_sigma: jax.Array
    def __init__(self, mu, log_sigma):
        self.mu = mu
        self.log_sigma = log_sigma

    @property
    def sigma(self):
        return jnp.exp(self.log_sigma)   

    def density(self, y, x=None):
        return _gaussian_pdf(y, self.mu, self.sigma)  