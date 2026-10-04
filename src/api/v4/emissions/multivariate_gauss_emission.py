from src.base.base_emission import BaseEmission 
import jax.numpy as jnp 
import jax.scipy.stats as stats 


class MultivariateGaussEmission(BaseEmission):
    """
    Multivariate Gaussian emission model for an HMM.

    Observations are `ys` of shape (T, K), where K is the number of observed
    series (e.g. CO2 and humidity). Given the state, the K dimensions are taken
    to be conditionally independent -- a diagonal covariance, no cross-dimension
    correlation -- so the emission probability of a state is the product of its
    per-dimension marginal Gaussian densities (see docs/review/multivariate-hmm.md).
    Because that product collapses the K axis, `density` returns the same
    (1, num_states) shape as the univariate emissions and the forward algorithm,
    the solvers and the likelihoods work unchanged.

    Parameters, with K dimensions and N states:
        mu0          (K,)     state-0 mean of each dimension
        log_mu_diff  (K, N-1) log gaps between consecutive state means, per dimension
        log_sigma    (K, N)   log standard deviation per dimension per state

    Storing the gaps in logs keeps the state means strictly increasing within each
    dimension, which is what identifies the states -- the same device GaussEmission
    uses.
    """ 
    log_mu_diff: jnp.ndarray
    mu0: jnp.ndarray
    log_sigma: jnp.ndarray 

    def __init__(self, log_mu_diff, mu0, log_sigma):
        self.log_mu_diff = jnp.asarray(log_mu_diff, dtype=float)
        self.mu0 = jnp.asarray(mu0, dtype=float)
        self.log_sigma = jnp.asarray(log_sigma, dtype=float)

    @classmethod
    def from_params(cls, mu, sigma):
        """`mu` and `sigma` are (K, N): one row per observed dimension."""
        mu = jnp.atleast_2d(jnp.asarray(mu, dtype=float))
        sigma = jnp.atleast_2d(jnp.asarray(sigma, dtype=float))
        log_mu_diff = jnp.log(jnp.diff(mu, axis=1))  # gaps between state means
        return cls(log_mu_diff, mu[:, 0], jnp.log(sigma))

    def density(self, t:int, ys: jnp.ndarray, xs: jnp.ndarray | None = None, ts: jnp.ndarray | None = None) -> jnp.ndarray:
        """Joint density p(y_t | z_t) per state, shape (1, num_states).

        The product over the K conditionally independent dimensions.
        """
        marginals = self.marginal_densities(t, ys, xs, ts)
        return jnp.prod(marginals, axis=0, keepdims=True)

    def marginal_densities(self, t: int, ys: jnp.ndarray, xs: jnp.ndarray | None = None, ts: jnp.ndarray | None = None) -> jnp.ndarray:
        """Per-dimension, per-state density of y_t, shape (K, num_states)."""
        mu, sigma = self.step(t, ys, xs)
        return stats.norm.pdf(ys[t, :][:, None], loc=mu, scale=sigma)

    def step(self, t: int, ys: jnp.ndarray, xs: jnp.ndarray | None = None, ts: jnp.ndarray | None = None):
        return self.mu(t, ys, xs), self.sigma(t, ys, xs)

    def mu(self, t: int, ys: jnp.ndarray, xs: jnp.ndarray | None = None, ts: jnp.ndarray | None = None):
        """State means, shape (K, num_states).

        Rebuilt per dimension from the state-0 mean plus the cumulative gaps, so
        the accumulation runs along the state axis and each dimension keeps its
        own offset.
        """
        first = self.mu0[:, None]                                 # (K, 1)
        gaps = jnp.cumsum(jnp.exp(self.log_mu_diff), axis=1)      # (K, N-1)
        return jnp.concatenate([first, first + gaps], axis=1)

    def sigma(self, t: int, ys: jnp.ndarray, xs: jnp.ndarray | None = None, ts: jnp.ndarray | None = None):
        return jnp.exp(self.log_sigma)

    def cdf(self, t: int, ys: jnp.ndarray, xs: jnp.ndarray | None = None, ts: jnp.ndarray | None = None) -> jnp.ndarray:
        """Product of the marginal CDFs per state, shape (1, num_states).

        Mirrors `density` so `HMM.pseudo_residuals` runs, but note the product of
        marginal CDFs is *not* a probability-integral transform: for K > 1 the
        resulting pseudo-residuals are not uniform under the model, so their QQ
        plots must not be read as a goodness-of-fit check.
        """
        marginals = self.marginal_cdfs(t, ys, xs, ts)
        return jnp.prod(marginals, axis=0, keepdims=True)

    def marginal_cdfs(self, t: int, ys: jnp.ndarray, xs: jnp.ndarray | None = None, ts: jnp.ndarray | None = None) -> jnp.ndarray:
        """Per-dimension, per-state CDF at y_t, shape (K, num_states)."""
        mu, sigma = self.step(t, ys, xs)
        return stats.norm.cdf(ys[t, :][:, None], loc=mu, scale=sigma)
