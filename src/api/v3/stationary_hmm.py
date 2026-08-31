import jax
import jax.numpy as jnp
import equinox as eqx

from src.deprecated.base.hmm import HMM
from src.deprecated.base.transition import Transition
from src.deprecated.base.emmision import Emission
from jax.scipy.stats import norm


class StationaryTransition(Transition):
    def __init__(self, transition_logits):
        super().__init__(transition_logits, initial_state_dist=None)


    def u0(self):
        return self._compute_inital_state_distribution()


    def _compute_inital_state_distribution(self):
        I = jnp.eye(self.num_states)
        E = jnp.ones((self.num_states, self.num_states))
        e = jnp.ones((self.num_states, 1))

        try:
            Gamma = self._inital_transition_matrix()
            delta = e.T @ jnp.linalg.inv(I - Gamma + E)  # (1, num_states)
            return delta.flatten()  # (num_states,)

        except Exception as e:
            raise ValueError(f"Error computing inital state distribution. Maybe the Stationary Transition matrix is not invertible? {e}")


    def _inital_transition_matrix(self):
        try:
            Gamma = self.transition_matrix()
            return Gamma
        except Exception as e:
            raise ValueError(f"Error computing transition matrix for initial state distribution. Did you step function include covaries xt?:\n {e}")



class StationaryGaussianEmission(Emission):
    mu_diff: jnp.ndarray
    log_sigma: jnp.ndarray

    def __init__(self, mu, log_sigma):
        self.mu_diff = jnp.concatenate([mu[0:1], jnp.array(jnp.log(jnp.diff(mu)))])  # Store log-differences to ensure monotonicity
        self.log_sigma = log_sigma

    def _compute_mu(self):
        return jnp.concatenate([self.mu_diff[0:1], self.mu_diff[0:1] + jnp.cumsum(jnp.exp(self.mu_diff[1:]))])

    def step(self, xt=None):
        """$
        Compute the emission parameters at time step t given the covariates xt.
        """
        sigma = jnp.exp(self.log_sigma)
        return self._compute_mu(), sigma

    def density(self, yt, xt=None):
        """
        Returns the emission density for a Gaussian Distribution one for each state given the observation yt and covariates xt.
        """
        mu, sigma = self.step(xt)
        return norm.pdf(yt, loc=mu, scale=sigma)

    def cdf(self, yt, xt=None):
        """
        Returns the emission cdf for a Gaussian Distribution one for each state given the observation yt and covariates xt.
        """
        mu, sigma = self.step(xt)
        return norm.cdf(yt[:, None], loc=mu, scale=sigma)
    



class StationaryHMM(HMM):
    def __init__(self, transition_logits, mu, log_sigma):
        super().__init__(transition_logits, initial_state_dist=None)
        self.transition = StationaryTransition(transition_logits)
        self.emission = StationaryGaussianEmission(mu, log_sigma)

    def mu(self, xt=None):
        mu, _ = self.emission.step(xt=xt)
        return mu

    def sigma(self, xt=None):
        _, sigma = self.emission.step(xt=xt)
        return sigma

    def transition_matrix(self, xt=None):
        return self.transition.transition_matrix(xt=xt)


    def filter_spec(self):
        spec = jax.tree_util.tree_map(eqx.is_inexact_array, self)

        spec = eqx.tree_at(
              lambda m: m.transition.initial_state_dist,
              spec,
              replace=False
          )
        return spec
