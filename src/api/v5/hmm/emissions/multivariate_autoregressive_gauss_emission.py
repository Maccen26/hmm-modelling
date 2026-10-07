from src.api.v5.base import BaseEmission 
import jax.scipy.stats as stats 
import jax.numpy as jnp 
import jax 
from .utils import phi_to_phi_tilde, phi_tilde_to_phi


class MultivariateAutoregressiveGaussEmission(BaseEmission):
    """
    Multivariate autoregressive Gaussian emission model for an HMM.

    The multivariate counterpart of AutoregressiveGaussEmission: observations are
    `ys` of shape (T, K), and given the state the K dimensions are conditionally
    independent, so a state's emission probability is the product of its
    per-dimension marginal densities (docs/review/multivariate-hmm.md). Each
    dimension carries its own AR recursion on its own past values,

        mu[j, i](t) = m[j, i] + sum_l phi[l, j, i] * (y[t-l, j] - m[j, i])

    where m is the state mean of dimension j in state i. No dimension reads
    another's history -- the cross-dimension coupling in this model runs through
    the shared hidden state, not through the AR term.

    Parameters, with K dimensions, N states and k lags:
        mu0          (K,)       state-0 mean of each dimension
        log_mu_diff  (K, N-1)   log gaps between consecutive state means, per dimension
        log_sigma    (K, N)     log standard deviation per dimension per state
        phi_tilde    (k, K, N)  unconstrained AR coefficients, mapped into (-1, 1)

    The first k observations have no complete history, so their density is 1,
    contributing exactly 0 to the log-likelihood.
    """ 
    log_mu_diff: jnp.ndarray
    mu0: jnp.ndarray
    log_sigma: jnp.ndarray
    phi_tilde: jnp.ndarray  # (num_lags, num_dims, num_states) — allows per-lag freezing


    def __init__(self, log_mu_diff, mu0, log_sigma, phi_tilde):
        self.log_mu_diff = jnp.asarray(log_mu_diff, dtype=float)
        self.mu0 = jnp.asarray(mu0, dtype=float)
        self.log_sigma = jnp.asarray(log_sigma, dtype=float)
        self.phi_tilde = self._as_lag_dim_state(jnp.asarray(phi_tilde, dtype=float))

    @staticmethod
    def _as_lag_dim_state(phi_tilde: jnp.ndarray) -> jnp.ndarray:
        """Normalise AR coefficients to (num_lags, num_dims, num_states).

        A 2-D (num_lags, num_states) array is read as a single dimension, which is
        what makes the K=1 case spell the same as the univariate emission.
        """
        phi_tilde = jnp.atleast_2d(phi_tilde)
        return phi_tilde[:, None, :] if phi_tilde.ndim == 2 else phi_tilde

    @classmethod
    def from_params(cls, mu, sigma, phi):
        """`mu`/`sigma` are (K, N) and `phi` is (k, K, N): a row per observed dimension."""
        mu = jnp.atleast_2d(jnp.asarray(mu, dtype=float))
        sigma = jnp.atleast_2d(jnp.asarray(sigma, dtype=float))
        log_mu_diff = jnp.log(jnp.diff(mu, axis=1))  # gaps between state means
        return cls(log_mu_diff, mu[:, 0], jnp.log(sigma), phi_to_phi_tilde(phi))

    def density(self, t:int, ys: jnp.ndarray) -> jnp.ndarray:
        """Joint density p(y_t | z_t) per state, shape (1, num_states).

        The product over the K conditionally independent dimensions, neutralised to
        1 for the first k steps whose lags do not exist yet.
        """
        density = jnp.prod(self.marginal_densities(t, ys), axis=0, keepdims=True)
        return jnp.where(t < len(self.phi_tilde), jnp.ones_like(density), density)

    def marginal_densities(self, t: int, ys: jnp.ndarray) -> jnp.ndarray:
        """Per-dimension, per-state density of y_t, shape (K, num_states)."""
        mu, sigma = self.step(t, ys)
        return stats.norm.pdf(ys[t, :][:, None], loc=mu, scale=sigma)

    def step(self, t: int, ys: jnp.ndarray):
        return self.mu(t, ys), self.sigma(t, ys)

    def mu(self, t: int, ys: jnp.ndarray):
        """State means with the AR correction applied, shape (K, num_states)."""
        base_mu = self.mu_vals(t, ys)                      # (K, N)
        k = len(self.phi_tilde)
        lags = self.lags(t, ys, k)                             # (k, K)
        ar = jnp.sum(self.phi() * (lags[:, :, None] - base_mu[None, :, :]), axis=0)
        return jnp.where(t < k, base_mu, base_mu + ar)

    @staticmethod
    def lags(t: int, ys: jnp.ndarray, k: int) -> jnp.ndarray:
        """The k most recent observations before t, shape (k, K).

        Always k rows, so it is safe under jit even when t < k (those rows are
        garbage, and `mu` discards them). After the flip lags[0] = y_{t-1} and
        lags[k-1] = y_{t-k}, matching the lag ordering of `phi`'s leading axis.
        """
        window = jax.lax.dynamic_slice(ys, (jnp.maximum(t - k, 0), 0), (k, ys.shape[1]))
        return jnp.flip(window, axis=0)

    def mu_vals(self, t: int, ys: jnp.ndarray):
        """State means before the AR correction, shape (K, num_states).

        Rebuilt per dimension from the state-0 mean plus the cumulative gaps, so the
        accumulation runs along the state axis and each dimension keeps its own
        offset. The means are then tiled to however many states `log_sigma` carries,
        which is how a higher-order transition ties one mean across the augmented
        states (as the univariate emission does).
        """
        first = self.mu0[:, None]                                  # (K, 1)
        gaps = jnp.cumsum(jnp.exp(self.log_mu_diff), axis=1)       # (K, N-1)
        base = jnp.concatenate([first, first + gaps], axis=1)      # (K, N)
        n_tiles = self.log_sigma.shape[1] // base.shape[1]
        return jnp.tile(base, (1, n_tiles))

    def sigma(self, t: int, ys: jnp.ndarray):
        return jnp.exp(self.log_sigma)

    def phi(self):
        return phi_tilde_to_phi(self.phi_tilde)  # (num_lags, num_dims, num_states)

    def cdf(self, t: int, ys: jnp.ndarray) -> jnp.ndarray:
        """Product of the marginal CDFs per state, shape (1, num_states).

        Mirrors `density` so `HMM.pseudo_residuals` runs, but note the product of
        marginal CDFs is *not* a probability-integral transform: for K > 1 the
        resulting pseudo-residuals are not uniform under the model, so their QQ
        plots must not be read as a goodness-of-fit check.
        """
        return jnp.prod(self.marginal_cdfs(t, ys), axis=0, keepdims=True)

    def marginal_cdfs(self, t: int, ys: jnp.ndarray) -> jnp.ndarray:
        """Per-dimension, per-state CDF at y_t, shape (K, num_states)."""
        mu, sigma = self.step(t, ys)
        return stats.norm.cdf(ys[t, :][:, None], loc=mu, scale=sigma)
