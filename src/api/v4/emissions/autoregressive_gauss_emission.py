from src.base.base_emission import BaseEmission 
import jax.scipy.stats as stats 
import jax.numpy as jnp 
import jax 
from src.api.v4.utils import phi_to_phi_tilde, phi_tilde_to_phi
class AutoregressiveGaussEmission(BaseEmission):
    """
    Gaussian emission model for an HMM. The emission density is a Gaussian distribution with mean and variance that can depend on the covariates at time step t. 
    """ 
    log_mu_diff: jnp.ndarray
    mu0: jnp.ndarray
    log_sigma: jnp.ndarray
    phi_tilde: jnp.ndarray  # tuple of 1D arrays, one per lag — allows per-lag freezing


    def __init__(self, log_mu_diff, mu0, log_sigma, phi_tilde):
        self.log_mu_diff = jnp.asarray(log_mu_diff, dtype=float)
        self.mu0 = jnp.asarray(mu0, dtype=float)
        self.log_sigma = jnp.asarray(log_sigma, dtype=float)
        self.phi_tilde = jnp.atleast_2d(jnp.asarray(phi_tilde, dtype=float))


    @classmethod
    def from_params(cls, mu, sigma, phi):
        mu = jnp.asarray(mu, dtype=float)
        sigma = jnp.asarray(sigma, dtype=float)
        log_mu_diff = jnp.log(jnp.diff(mu))  # Store log-differences to ensure monotonicity
        mu0 = mu[0]
        log_sigma = jnp.log(sigma)  # Store log of sigma to ensure positivity
        phi_tilde = phi_to_phi_tilde(phi)  # Store transformed phi values
        return cls(log_mu_diff, mu0, log_sigma, phi_tilde)
    

    def density(self, t:int, ys: jnp.ndarray, xs: jnp.ndarray | None = None) -> jnp.ndarray:
        mu, sigma = self.step(t, ys, xs)
        yt = ys[t]
        density = stats.norm.pdf(jnp.atleast_1d(yt)[:, None], loc=mu, scale=sigma)
        k = len(self.phi_tilde)  # no of lags
        return_val = jnp.where(t < k, jnp.ones_like(density), density)  #Density of 1 gives a log like of 0
        return return_val
    
    def step(self, t: int, ys: jnp.ndarray, xs: jnp.ndarray | None = None):
        return self.mu(t, ys, xs), self.sigma(t, ys, xs) 
    
    def mu(self, t: int, ys: jnp.ndarray, xs: jnp.ndarray | None = None):
        base_mu = self.mu_vals(t, ys, xs)
        k = len(self.phi_tilde)
        
        # Always produces shape (k,) — safe even when t < k
        # After flip: lags[0] = y_{t-1} (lag 1), lags[k-1] = y_{t-k} (lag k), matching phi row ordering
        lags = jnp.flip(jax.lax.dynamic_slice(ys, (jnp.maximum(t - k, 0),), (k,)))
        
        ar = jnp.sum(self.phi() * (lags[:, None] - base_mu[None, :]), axis=0)
        return jnp.where(t < k, base_mu, base_mu + ar)
    
    def mu_vals(self, t: int, ys: jnp.ndarray, xs: jnp.ndarray | None = None):
        base = jnp.concatenate([
            jnp.array([self.mu0]),
            self.mu0 + jnp.cumsum(jnp.exp(self.log_mu_diff)),
        ])
        n_tiles = self.log_sigma.shape[0] // base.shape[0]
        return jnp.tile(base, n_tiles)

    def sigma(self, t: int, ys: jnp.ndarray, xs: jnp.ndarray | None = None):
        return jnp.exp(self.log_sigma) 
    
    def phi(self):
        return phi_tilde_to_phi(self.phi_tilde)  # (num_lags, num_states)
    
    def cdf(self, t: int, ys: jnp.ndarray, xs: jnp.ndarray | None = None) -> jnp.ndarray:
        mu = self.mu(t, ys, xs)
        sigma = self.sigma(t, ys, xs)
        return stats.norm.cdf(jnp.atleast_1d(ys[t])[:, None], loc=mu, scale=sigma)