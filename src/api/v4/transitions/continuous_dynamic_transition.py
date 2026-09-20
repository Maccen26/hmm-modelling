from src.base import BaseTransition
import jax
import jax.numpy as jnp

from src.base.utils import (
    get_Q_from_logits,
    transtion_matrix_to_logits_continuous,
)


class ContinuousDynamicTransition(BaseTransition):
    """
    Continuous-time transition model whose generator depends on covariates.

    The off-diagonal log-rates of the generator matrix Q are shifted by a linear
    combination of the covariates at each time step, so the instantaneous rate of
    moving from state i to state j is

        q_ij(x_t) = exp( transition_logits[i, j] + sum_k beta[k, i, j] * x_t[k] ).

    The transition matrix for an observation with waiting time dt is then
    ``expm(Q(x_t) * dt)``, exactly as in the static continuous case but with a
    per-step generator.

    transition_logits: jnp.ndarray of shape (num_states, num_states - 1) holding
        the baseline off-diagonal log-rates.
    beta: jax.Array of shape (num_covariates, num_states, num_states - 1) holding
        the covariate effects on those log-rates.
    """

    beta: jax.Array

    def __init__(self, transition_logits, beta):
        self.transition_logits = jnp.asarray(transition_logits, dtype=float)
        self.beta = jnp.asarray(beta, dtype=float)

        b, n, m = self.beta.shape
        if (n, m) != self.transition_logits.shape:
            raise ValueError(
                f"beta and transition_logits must share their (num_states, num_states - 1) "
                f"shape. Got beta shape: {self.beta.shape}, transition_logits shape: "
                f"{self.transition_logits.shape}"
            )

    @classmethod
    def from_params(cls, transition_matrix, beta):
        transition_logits = transtion_matrix_to_logits_continuous(transition_matrix, 0)
        return cls(transition_logits, beta)

    def _covariate_effect(self, x_row: jnp.ndarray) -> jnp.ndarray:
        """Linear covariate shift for a single covariate vector `x_row`.

        Broadcasts each covariate scalar over its (num_states, num_states - 1)
        beta slice and sums across covariates, mirroring `DynamicTransition`.
        """
        x_row = x_row.flatten()
        tensor = self.beta * x_row[:, None, None]
        return tensor.sum(axis=0)

    def step(self, t: int | None, ys: jnp.ndarray | None = None, xs: jnp.ndarray | None = None) -> jnp.ndarray:
        """
        Computes the covariate-adjusted off-diagonal log-rates at time step `t`.

        :param t: index of the observation whose covariates are used.
        :param xs: covariate sequence of shape (T, num_covariates). If None the
            baseline log-rates are returned unchanged.
        :return: adjusted logits of shape (num_states, num_states - 1).
        """
        if xs is None:
            return self.transition_logits
        return self.transition_logits + self._covariate_effect(xs[t])

    def get_Q(self, t: int | None = None, ys: jnp.ndarray | None = None, xs: jnp.ndarray | None = None) -> jnp.ndarray:
        """Generator matrix Q at time step `t` given its covariates."""
        return get_Q_from_logits(self.step(t, ys, xs))

    def transition_matrix(self, t: int | None = None, ys: jnp.ndarray | None = None, xs: jnp.ndarray | None = None, dt: float = 1.0) -> jnp.ndarray:
        """
        Builds the transition matrix for the step at index `t`.

        :param t: index whose covariates parameterize the generator (None -> baseline).
        :param xs: covariate sequence of shape (T, num_covariates).
        :param dt: waiting time used in expm(Q * dt). Defaults to a unit step, which
            is what the stationary-distribution computation expects.
        :return: transition matrix of shape (num_states, num_states).
        """
        Q = self.get_Q(t, ys, xs)
        return jax.scipy.linalg.expm(Q * jnp.asarray(dt, dtype=Q.dtype))

    def transition_matrices(self, ts: jnp.ndarray, ys: jnp.ndarray | None = None, xs: jnp.ndarray | None = None) -> jnp.ndarray:
        """
        Batched transition matrices T_i = expm(Q(x_i) * ts[i]).

        `ts` are the per-observation waiting times and `xs` the parallel covariate
        rows; observation i uses both ts[i] and xs[i]. When no covariates are given
        the generator is constant and this reduces to the static continuous case.
        """
        if xs is None:
            Q = get_Q_from_logits(self.transition_logits)
            return jax.vmap(lambda t: jax.scipy.linalg.expm(Q * t.astype(Q.dtype)))(ts)

        indices = jnp.arange(ts.shape[0])

        def one(i):
            Q = get_Q_from_logits(self.step(i, ys, xs))
            return jax.scipy.linalg.expm(Q * ts[i].astype(Q.dtype))

        return jax.vmap(one)(indices)

    def base_transition_matrix(self, dt: float = 1.0) -> jnp.ndarray:
        """Transition matrix from the baseline generator, ignoring covariates."""
        Q = get_Q_from_logits(self.transition_logits)
        return jax.scipy.linalg.expm(Q * jnp.asarray(dt, dtype=Q.dtype))
