import jax
import jax.numpy as jnp
import equinox as eqx
from jax import lax

from src.deprecated.base.hmm import HMM
from src.api.v3.stationary_hmm import StationaryTransition
from src.api.v3.stationary_hmm import StationaryGaussianEmission


class ArGaussianEmisionConstrained(StationaryGaussianEmission):
    phi_tilde: jnp.ndarray  # unconstrained AR(1) parameter per state, shape (num_states,)

    def __init__(self, mu, log_sigma, phi_tilde):
        super().__init__(mu, log_sigma)
        self.phi_tilde = phi_tilde

    def step(self, xt=None):
        phi = self.phi
        mu = self._compute_mu()
        #mu = mu + phi * (xt - mu)  # xt = y_{t-1}
        lags = (xt[None, :] -  mu[:, None] ) * phi.reshape(len(mu), -1) # shape (num_states, lags)
        mu = mu + lags.sum(axis=1) # shape (num_states,) sum rows
        return mu, jnp.exp(self.log_sigma) 
    
    def transform_phi(self, phi_tilde):
        """Transforms the unconstrained phi_tilde to the constrained phi in (-1, 1) using a sigmoid-like function."""
        return (2 * jnp.exp(phi_tilde)) / (1 + jnp.exp(phi_tilde)) - 1
    
    @property
    def phi(self):
        return self.transform_phi(self.phi_tilde)
    


class ArHMMConstrained(HMM):
    lags: int = eqx.field(static=True)

    def __init__(self, transition_logits, mu, log_sigma, phi_tilde, lags=1):
        super().__init__(transition_logits, initial_state_dist=None)
        self.transition = StationaryTransition(transition_logits)
        self.emission = ArGaussianEmisionConstrained(mu, log_sigma, phi_tilde)
        self.lags = lags

    def forward(self, y, x=None):
        """
        implements the forward algorithm for HMMs. 
        x is of dim (T, covariestes_dim) 
        y is of dim (T, obs_dim) 

        where T is the sequence length.
        """
        T = y.shape[0]
        yt = y[self.lags:]  # shape (T-lags, obs_dim)

        x = jnp.stack(
        tuple(y[self.lags - k : T - k] for k in range(1, self.lags + 1)),
        axis=1,
        )  # shape (T-lags, lags, obs_dim)
        y = yt

        ut0 = self.transition.u0()

        if x is None:
            _, (Ut, ft, Utt) = lax.scan(lambda carry, yt: self.step(carry, yt), ut0, y)
        else:
            _, (Ut, ft, Utt) = lax.scan(lambda carry, yx: self.step(carry, yx[0], yx[1]), ut0, (y, x)) 
        return Ut, ft, Utt 

