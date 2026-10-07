import jax.numpy as jnp
import jax

from src.base import BaseTransition
from src.base.utils import logits_to_transition_matrix


class DynamicTransition(BaseTransition):
    """
    Covariate-driven transition model for an HMM.

    The off-diagonal transition logits are shifted by a linear combination of the
    covariates at each observation, so the matrix changes from step to step:

        logits(x_t) = transition_logits + sum_k beta[k] * x_t[k]

    transition_logits: jnp.ndarray of shape (num_states, num_states - 1) holding the
        baseline off-diagonal logits.
    beta: jax.Array of shape (num_covariates, num_states, num_states - 1) holding the
        covariate effects on those logits.
    """

    beta: jax.Array

    # Explicit __init__ with transition_logits first: BaseTransition.update_param
    # rebuilds the class positionally via self.__class__(*params), so the argument
    # order must match the field declaration order.
    def __init__(self, transition_logits, beta):
        self.transition_logits = jnp.asarray(transition_logits, dtype=float)
        self.beta = jnp.asarray(beta, dtype=float)

        b, n, m = self.beta.shape
        if (n, m) != self.transition_logits.shape:
            raise ValueError(
                f"beta and transition_logits must have the same shape. Got beta shape: "
                f"{self.beta.shape}, transition_logits shape: {self.transition_logits.shape}")

    @classmethod
    def from_params(cls, transition_matrix, beta):
        from src.base.utils import transition_matrix_to_logits
        return cls(transition_matrix_to_logits(transition_matrix), beta)

    def step(self, t: int | None, ys: jnp.ndarray | None = None, xs: jnp.ndarray | None = None) -> jnp.ndarray:
        """
        Computes the covariate-adjusted transition logits at observation index `t`.

        :param t: index of the observation whose covariate row xs[t] is used.
        :param xs: covariate sequence of shape (T, num_covariates).
        :return: transition logits of shape (num_states, num_states - 1).
        """
        if xs is None:
            # A dynamic transition has no single covariate-free matrix. Raising keeps
            # that explicit rather than silently substituting the baseline logits --
            # callers that need one (e.g. a stationary distribution) must supply it.
            raise ValueError(
                "DynamicTransition requires covariates `xs`; it has no time-invariant "
                "transition matrix. Pass xs, or supply an explicit initial distribution.")

        xt = xs[t, :].flatten()  # covariate row at this observation, as a 1D array
        # Broadcast each covariate scalar over its (num_states, num_states - 1) beta
        # slice and sum across covariates.
        tensor = self.beta * xt[:, None, None]
        return self.transition_logits + tensor.sum(axis=0)

    def transition_matrix(self, t: int | None = None, ys: jnp.ndarray | None = None, xs: jnp.ndarray | None = None, dt: float | None = None) -> jnp.ndarray:
        """
        Builds the transition matrix for the observation at index `t`.

        :param t: index whose covariate row parameterises the matrix.
        :param xs: covariate sequence of shape (T, num_covariates).
        :param dt: ignored — this is a discrete-time transition.
        :return: transition matrix of dim (num_states, num_states)
        """
        return logits_to_transition_matrix(self.step(t, ys, xs))

    def base_transition_matrix(self) -> jnp.ndarray:
        """
        Returns the base transition matrix without any covariate effects.
        This is useful for computing the stationary distribution of the HMM.
        """
        return logits_to_transition_matrix(self.transition_logits)
