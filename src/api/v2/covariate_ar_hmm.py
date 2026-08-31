import jax
import jax.numpy as jnp
import equinox as eqx

from src.deprecated.base.hmm import HMM
from src.deprecated.base.transition import Transition

from src.api.v2.stationary_hmm import StationaryGaussianEmission


class CovariateTransition(Transition):
    beta: jnp.ndarray
    def __init__(self, transition_logits, beta, initial_state_dist):
        self.beta = beta
        super().__init__(transition_logits, initial_state_dist=initial_state_dist)


    def step(self, xt=None):
        """
        xt is the covarites at time step t.
        Returns the the transtion logits matrix at time step t of dim (num_states, num_states)
        """
        time = xt[self.num_states:]  # shape (num_covariates,)
        tGamma = jnp.zeros((self.num_states, self.num_states))

        diag_vals = self.beta @ time  # shape (num_states,)

        tGamma = tGamma.at[jnp.diag_indices(self.num_states)].set(diag_vals)

        rows, cols = jnp.where(~jnp.eye(self.num_states, dtype=bool), size=self.num_states * (self.num_states - 1))
        tGamma = tGamma.at[rows, cols].set(self.transition_logits.flatten())
        return tGamma


class ArGaussianEmisionBackground(StationaryGaussianEmission):
    phi: jnp.ndarray  # AR(1) coefficient per state, shape (num_states,)

    def __init__(self, mu_diff, log_sigma, phi):
        super().__init__(mu_diff, log_sigma)
        self.phi = phi

    def step(self, xt=None):
        yt = xt[0: len(self.phi)]
        mu = jnp.concatenate([jnp.array([400.0]), self._compute_mu()])  # shape (num_states,)
        mu = mu + self.phi * (yt - mu)  # xt = y_{t-1}
        return mu, jnp.exp(self.log_sigma)


class CovariateArHMM(HMM):
    def __init__(self, transition_logits, mu_diff, log_sigma, initial_state_dist, beta, phi):
        super().__init__(transition_logits, initial_state_dist=initial_state_dist)
        self.transition = CovariateTransition(transition_logits=transition_logits, beta=beta, initial_state_dist=initial_state_dist)
        self.emission = ArGaussianEmisionBackground(mu_diff=mu_diff, log_sigma=log_sigma, phi=phi)

    def filter_spec(self):
        spec = jax.tree_util.tree_map(eqx.is_inexact_array, self)

        spec = eqx.tree_at(
              lambda m: m.transition.initial_state_dist,
              spec,
              replace=False
          )
        return spec
