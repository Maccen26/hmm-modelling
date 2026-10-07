import jax
import jax.numpy as jnp
import equinox as eqx

from src.base import BaseTransition
from src.api.v4.transitions.static_transition_higher_order import (
    _make_transition_logits,
    logits_to_transition_matrix_higher_order,
    offdiag_logits_to_higher_order,
)


class DynamicTransitionHigherOrder(BaseTransition):
    """
    Covariate-driven transition model for a higher-order HMM.

    Combines the two existing extensions of `StaticTransition`: the chain runs on
    augmented states (s_{t-order+1}, ..., s_t) like `StaticTransitionHigherOrder`,
    and the logits of every allowed transition are shifted by a linear combination
    of the covariates at each observation like `DynamicTransition`:

        logits(x_t)[i, :] = transition_logits[i, :] + sum_k beta[k, i % K, :] * x_t[k]

    The covariate effects are *tied over history*: row i of the augmented matrix
    encodes (s_{t-1}, s_t) as i = s_{t-1} * K + s_t, so `i % K` is the current base
    state and every augmented row sharing that current state gets the same covariate
    shift. That keeps the covariate budget identical to the first-order
    `DynamicTransition` (D * K * (K-1) parameters instead of D * K**order * (K-1)),
    so a comparison against it isolates the added memory.

    transition_logits: jnp.ndarray of shape (K**order, K - 1) holding the baseline
        logits of the allowed transitions out of each augmented state.
    beta: jax.Array of shape (num_covariates, K, K - 1) holding the covariate effects,
        indexed by the *current* base state.
    order: order of the Markov chain (static — it fixes the shapes, so it is not
        trainable).
    """

    beta: jax.Array
    order: int = eqx.field(static=True, default=2)

    # Explicit __init__ with transition_logits first: BaseTransition.update_param
    # rebuilds the class positionally via self.__class__(*params), so the argument
    # order must match the field declaration order (transition_logits, beta, order).
    def __init__(self, transition_logits, beta, order=2):
        self.transition_logits = jnp.asarray(transition_logits, dtype=float)
        self.beta = jnp.asarray(beta, dtype=float)
        self.order = order
        self._validate_shapes()

    def _validate_shapes(self):
        if self.transition_logits.ndim != 2:
            raise ValueError(
                f"transition_logits must be 2-D of shape (K**order, K - 1). "
                f"Got shape {self.transition_logits.shape}.")
        num_states = self.num_states
        expected_rows = num_states ** self.order
        if self.transition_logits.shape[0] != expected_rows:
            raise ValueError(
                f"Expected {expected_rows} rows of transition logits for order "
                f"{self.order} and {num_states} states, but got "
                f"{self.transition_logits.shape[0]}.")
        if self.beta.ndim != 3 or self.beta.shape[1:] != (num_states, num_states - 1):
            raise ValueError(
                f"beta must have shape (num_covariates, {num_states}, "
                f"{num_states - 1}) -- covariate effects are tied over history. "
                f"Got beta shape: {self.beta.shape}.")

    @property
    def num_states(self) -> int:
        """Number of *base* states K (the augmented chain has K**order)."""
        return self.transition_logits.shape[1] + 1

    @classmethod
    def from_params(cls, transition_matrix, beta, order=2):
        """
        Builds the transition from a *base* (K, K) transition matrix, lifting its
        logits onto the augmented state space.

        The augmented matrix is structurally constrained (only K of its K**order
        columns are reachable from any row), so there is no meaningful inverse from
        a full (K**order, K**order) matrix -- the natural seed is a first-order
        matrix lifted to augmented states, which is what this does.

        The lift is exact: the resulting chain, read on the current base state, has
        exactly the dynamics of `transition_matrix`, because every augmented row
        sharing a current state gets the same (re-referenced) base row.

        :param transition_matrix: base transition matrix of shape (K, K).
        :param beta: covariate effects of shape (num_covariates, K, K - 1).
        """
        from src.base.utils import transition_matrix_to_logits
        base_logits = transition_matrix_to_logits(transition_matrix)
        return cls(cls._lift_logits(base_logits, order), beta, order)

    @classmethod
    def from_first_order(cls, transition, order=2):
        """
        Lifts a fitted first-order `DynamicTransition` onto the augmented state space.

        Both the baseline logits and the covariate effects are re-referenced (see
        `offdiag_logits_to_higher_order`) and the baseline is tiled across histories,
        so at every covariate value the lifted chain reproduces the first-order one
        exactly. It is the natural warm start: the second-order fit begins at the
        first-order optimum and can only improve on it.

        :param transition: a fitted `DynamicTransition` with beta of shape (D, K, K-1).
        """
        logits = offdiag_logits_to_higher_order(transition.transition_logits)
        beta = jax.vmap(offdiag_logits_to_higher_order)(transition.beta)
        num_states = logits.shape[0]
        tiled = jnp.tile(logits, (num_states ** (order - 1), 1))
        return cls(tiled, beta, order)

    @staticmethod
    def _lift_logits(base_logits, order):
        """Re-reference a (K, K-1) first-order logit block and tile it over histories."""
        lifted = offdiag_logits_to_higher_order(base_logits)
        return jnp.tile(lifted, (lifted.shape[0] ** (order - 1), 1))

    def step(self, t: int | None, ys: jnp.ndarray | None = None, xs: jnp.ndarray | None = None) -> jnp.ndarray:
        """
        Computes the covariate-adjusted logits of the full augmented matrix at
        observation index `t`.

        Like `StaticTransitionHigherOrder.step` this returns the *full*
        (K**order, K**order) logit matrix, with structurally impossible transitions
        driven to -1000, not the (K**order, K - 1) block of free logits.

        :param t: index of the observation whose covariate row xs[t] is used.
        :param xs: covariate sequence of shape (T, num_covariates).
        :return: transition logits of shape (K**order, K**order).
        """
        if xs is None:
            # A dynamic transition has no single covariate-free matrix. Raising keeps
            # that explicit rather than silently substituting the baseline logits --
            # callers that need one (e.g. a stationary distribution) must supply it.
            raise ValueError(
                "DynamicTransitionHigherOrder requires covariates `xs`; it has no "
                "time-invariant transition matrix. Pass xs, or supply an explicit "
                "initial distribution.")

        xt = xs[t, :].flatten()  # covariate row at this observation, as a 1D array
        # Broadcast each covariate scalar over its (K, K - 1) beta slice and sum
        # across covariates -> the shift for each *current* base state.
        shift = (self.beta * xt[:, None, None]).sum(axis=0)  # (K, K - 1)
        # Tile over the K**(order-1) histories: row i = ... * K + s_t picks shift[i % K].
        num_states = self.num_states
        tiled = jnp.tile(shift, (num_states ** (self.order - 1), 1))
        return _make_transition_logits(self.transition_logits + tiled, self.order)

    def transition_matrix(self, t: int | None = None, ys: jnp.ndarray | None = None, xs: jnp.ndarray | None = None, dt: float | None = None) -> jnp.ndarray:
        """
        Builds the augmented transition matrix for the observation at index `t`.

        :param t: index whose covariate row parameterises the matrix.
        :param xs: covariate sequence of shape (T, num_covariates).
        :param dt: ignored -- this is a discrete-time transition.
        :return: transition matrix of dim (K**order, K**order)
        """
        return logits_to_transition_matrix_higher_order(self.step(t, ys, xs))

    def base_transition_matrix(self) -> jnp.ndarray:
        """
        Returns the augmented transition matrix without any covariate effects.
        This is useful for seeding an initial distribution for the HMM.
        """
        logits = _make_transition_logits(self.transition_logits, self.order)
        return logits_to_transition_matrix_higher_order(logits)
