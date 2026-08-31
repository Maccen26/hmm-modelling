from src.base.base_emission import BaseEmission 
import jax.numpy as jnp 
import jax.scipy.stats as stats 


class GaussEmission(BaseEmission):
    """
    Gaussian emission model for an HMM. The emission density is a Gaussian distribution with mean and variance that can depend on the covariates at time step t. 
    """ 
    log_mu_diff: jnp.ndarray
    mu0: jnp.ndarray
    log_sigma: jnp.ndarray 

    def __init__(self, log_mu_diff, mu0, log_sigma):
        self.log_mu_diff = jnp.asarray(log_mu_diff, dtype=float)
        self.mu0 =jnp.asarray(mu0, dtype=float)
        self.log_sigma = jnp.asarray(log_sigma, dtype=float)

    @classmethod
    def from_params(cls, mu, sigma):
        mu = jnp.asarray(mu, dtype=float)
        sigma = jnp.asarray(sigma, dtype=float)
        log_mu_diff = jnp.log(jnp.diff(mu))  # Store log-differences to ensure monotonicity
        mu0 = mu[0]
        log_sigma = jnp.log(sigma)  # Store log of sigma to ensure positivity
        return cls(log_mu_diff, mu0, log_sigma)

    def density(self, t:int, ys: jnp.ndarray, xs: jnp.ndarray | None = None) -> jnp.ndarray:
        mu, sigma = self.step(t, ys, xs) 
        yt = ys[t]
        return stats.norm.pdf(jnp.atleast_1d(yt)[:, None], loc=mu, scale=sigma)
    
    def step(self, t: int, ys: jnp.ndarray, xs: jnp.ndarray | None = None):
        return self.mu(t, ys, xs), self.sigma(t, ys, xs) 
    
    def mu(self, t: int, ys: jnp.ndarray, xs: jnp.ndarray | None = None):
        return jnp.concatenate([jnp.array([self.mu0]), self.mu0 + jnp.cumsum(jnp.exp(self.log_mu_diff))]) 

    def sigma(self, t: int, ys: jnp.ndarray, xs: jnp.ndarray | None = None):
        return jnp.exp(self.log_sigma) 
    
    def cdf(self, t: int, ys: jnp.ndarray, xs: jnp.ndarray | None = None) -> jnp.ndarray:
        mu = self.mu(t, ys, xs)
        sigma = self.sigma(t, ys, xs)
        return stats.norm.cdf(jnp.atleast_1d(ys[t])[:, None], loc=mu, scale=sigma)
    