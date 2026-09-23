from unittest import TestCase

import jax
import jax.numpy as jnp
import numpy as np

from src.api.v4 import DynamicTransition, DynamicTransitionHigherOrder, StaticTransitionHigherOrder
from src.api.v4.transitions.static_transition_higher_order import decode_possible_transitions
from src.base.utils import transition_matrix_to_logits


class TestDynamicTransitionHigherOrder(TestCase):
    """
    Covariate-driven higher-order transition: shapes, the tying of beta over
    histories, the beta = 0 reduction to the static higher-order transition, and
    that gradients reach both parameter blocks.
    """

    def setUp(self) -> None:
        self.K = 4          # base states
        self.order = 2      # -> 16 augmented states
        self.D = 3          # covariates
        self.T = 12         # observations

        base = jnp.full((self.K, self.K), 0.1).at[jnp.diag_indices(self.K)].set(0.7)
        # One baseline logit block per augmented row: (K**order, K - 1).
        self.transition_logits = jnp.tile(transition_matrix_to_logits(base), (self.K, 1))
        self.beta = jax.random.uniform(
            jax.random.PRNGKey(0),
            shape=(self.D, self.K, self.K - 1),
            minval=-1.0,
            maxval=1.0,
        )
        self.xs = jax.random.normal(jax.random.PRNGKey(1), (self.T, self.D))

        self.transition = DynamicTransitionHigherOrder(
            self.transition_logits, self.beta, order=self.order)
        self.num_augmented = self.K ** self.order

    def test_build_from_params_lifts_a_base_matrix_exactly(self):
        """from_params(P) must reproduce P's rows on the reachable columns."""
        P = jnp.array([[0.70, 0.10, 0.15, 0.05],
                       [0.20, 0.50, 0.20, 0.10],
                       [0.05, 0.05, 0.80, 0.10],
                       [0.10, 0.20, 0.30, 0.40]])
        transition = DynamicTransitionHigherOrder.from_params(P, self.beta, order=self.order)
        self.assertEqual(transition.transition_logits.shape, (self.num_augmented, self.K - 1))
        lifted = np.asarray(transition.base_transition_matrix())
        for i in range(self.num_augmented):
            s_t = i % self.K                     # current base state of row i
            block = lifted[i, s_t * self.K:(s_t + 1) * self.K]
            self.assertTrue(np.allclose(block, np.asarray(P[s_t])),
                            msg=f"row {i} does not reproduce base row {s_t}")

    def test_from_first_order_is_an_exact_lift(self):
        """A lifted DynamicTransition must have identical base-state dynamics."""
        first = DynamicTransition(
            transition_matrix_to_logits(
                jnp.full((self.K, self.K), 0.1).at[jnp.diag_indices(self.K)].set(0.7)),
            self.beta)
        lifted = DynamicTransitionHigherOrder.from_first_order(first, order=self.order)
        for t in range(self.T):
            G_first = np.asarray(first.transition_matrix(t=t, xs=self.xs))
            G_lift = np.asarray(lifted.transition_matrix(t=t, xs=self.xs))
            for i in range(self.num_augmented):
                s_t = i % self.K
                block = G_lift[i, s_t * self.K:(s_t + 1) * self.K]
                self.assertTrue(np.allclose(block, G_first[s_t]),
                                msg=f"t = {t}, augmented row {i}")

    def test_transition_matrix_is_row_stochastic(self):
        Gamma = self.transition.transition_matrix(t=3, xs=self.xs)
        self.assertEqual(Gamma.shape, (self.num_augmented, self.num_augmented))
        self.assertTrue(np.allclose(np.asarray(Gamma.sum(axis=1)), 1.0))
        self.assertTrue(bool(jnp.all(Gamma >= 0.0)))

    def test_impossible_transitions_are_zero(self):
        """(s_{t-1}, s_t) -> (a, b) is only allowed when a == s_t."""
        Gamma = self.transition.transition_matrix(t=2, xs=self.xs)
        allowed = decode_possible_transitions(Gamma, order=self.order)
        for from_state, targets in allowed.items():
            # Each augmented state has exactly K successors with non-negligible mass.
            self.assertEqual(len(targets), self.K, msg=f"from {from_state}")
            for to_state, prob in targets:
                self.assertEqual(to_state[:-1], from_state[1:])
                self.assertGreater(float(prob), 1e-8)

    def test_zero_beta_reduces_to_static_higher_order(self):
        static = StaticTransitionHigherOrder(self.transition_logits, order=self.order)
        dynamic = DynamicTransitionHigherOrder(
            self.transition_logits, jnp.zeros_like(self.beta), order=self.order)
        expected = static.transition_matrix()
        for t in range(self.T):
            self.assertTrue(
                np.allclose(np.asarray(dynamic.transition_matrix(t=t, xs=self.xs)),
                            np.asarray(expected)),
                msg=f"beta = 0 should reproduce the static matrix at t = {t}")

    def test_base_transition_matrix_ignores_covariates(self):
        static = StaticTransitionHigherOrder(self.transition_logits, order=self.order)
        self.assertTrue(np.allclose(np.asarray(self.transition.base_transition_matrix()),
                                    np.asarray(static.transition_matrix())))

    def test_beta_is_tied_over_history(self):
        """Rows sharing a current base state get the same covariate shift."""
        logits = self.transition.step(t=4, xs=self.xs)
        baseline = StaticTransitionHigherOrder(
            self.transition_logits, order=self.order).step(t=None, ys=None)
        shift = np.asarray(logits - baseline)
        # Row i encodes (s_{t-1}, s_t) with s_t = i % K, so rows i and i + K share s_t.
        for i in range(self.K):
            for history in range(1, self.K ** (self.order - 1)):
                row_a = shift[i]
                row_b = shift[history * self.K + i]
                # Compare only the free (finite-baseline) entries: the impossible
                # transitions sit at different columns for different histories.
                mask = np.isfinite(row_a) & np.isfinite(row_b)
                self.assertTrue(np.allclose(np.sort(row_a[mask]), np.sort(row_b[mask])),
                                msg=f"rows {i} and {history * self.K + i} differ")

    def test_batched_matches_per_step(self):
        indices = jnp.arange(self.T)
        batched = self.transition.transition_matrices(indices, jnp.ones(self.T), None, self.xs)
        self.assertEqual(batched.shape, (self.T, self.num_augmented, self.num_augmented))
        for t in range(self.T):
            self.assertTrue(np.allclose(
                np.asarray(batched[t]),
                np.asarray(self.transition.transition_matrix(t=t, xs=self.xs))))

    def test_missing_covariates_raises(self):
        with self.assertRaises(ValueError):
            self.transition.step(t=0, ys=None, xs=None)
        with self.assertRaises(ValueError):
            self.transition.transition_matrix(t=0)

    def test_gradients_flow_to_both_parameter_blocks(self):
        def loss(transition):
            Gamma = transition.transition_matrix(t=2, xs=self.xs)
            return jnp.sum(jnp.log(Gamma + 1e-12))

        grads = jax.grad(loss)(self.transition)
        for name, g in (("transition_logits", grads.transition_logits), ("beta", grads.beta)):
            self.assertTrue(bool(jnp.all(jnp.isfinite(g))), msg=f"{name} has non-finite grads")
            self.assertGreater(float(jnp.abs(g).sum()), 0.0, msg=f"{name} has zero grads")

    def test_update_param_round_trip(self):
        new_beta = jnp.zeros_like(self.beta)
        updated = self.transition.update_param("beta", new_beta, None)
        self.assertIsInstance(updated, DynamicTransitionHigherOrder)
        self.assertEqual(updated.order, self.order)
        self.assertTrue(np.allclose(np.asarray(updated.beta), 0.0))
        self.assertTrue(np.allclose(np.asarray(updated.transition_logits),
                                    np.asarray(self.transition_logits)))

    def test_shape_validation(self):
        with self.assertRaises(ValueError):
            # beta must be tied over history: (D, K, K - 1), not (D, K**order, K - 1)
            DynamicTransitionHigherOrder(
                self.transition_logits,
                jnp.zeros((self.D, self.num_augmented, self.K - 1)),
                order=self.order)
        with self.assertRaises(ValueError):
            # transition_logits must have K**order rows
            DynamicTransitionHigherOrder(
                jnp.zeros((self.K, self.K - 1)), self.beta, order=self.order)
