import jax
jax.config.update("jax_enable_x64", True)

from unittest import TestCase
import numpy as np
import jax.numpy as jnp

from src.api.v5.base import BaseTransition
from src.api.v5.hmm.transitions import (
    StaticTransition,
    StaticTransitionHigherOrder,
    DynamicTransition,
    DynamicTransitionHigherOrder,
)
from src.base.utils import logits_to_transition_matrix


def naive_transition_matrices(transition, xs):
    """Reference: one matrix per observation, no deduplication."""
    return jax.vmap(transition.transition_matrix)(xs)


class TestBaseTransitionInterface(TestCase):
    """
    Tests the contract every v5 transition shares through BaseTransition.
    """

    def setUp(self) -> None:
        self.K = 3
        self.P = jnp.array([[0.8, 0.1, 0.1],
                            [0.1, 0.8, 0.1],
                            [0.1, 0.1, 0.8]])
        rng = np.random.default_rng(0)
        self.beta = jnp.asarray(rng.normal(0, 0.3, (2, self.K, self.K - 1)))

    def test_cannot_instantiate_abstract_base(self):
        with self.assertRaises(TypeError):
            BaseTransition(jnp.zeros((self.K, self.K - 1)))

    def test_all_transitions_subclass_v5_base(self):
        for cls in (StaticTransition, StaticTransitionHigherOrder, DynamicTransition, DynamicTransitionHigherOrder):
            self.assertTrue(issubclass(cls, BaseTransition), cls.__name__)

    def test_from_params_round_trip(self):
        transition = StaticTransition.from_params(self.P)
        self.assertTrue(jnp.allclose(transition.transition_matrix(), self.P))

    def test_equality(self):
        a = DynamicTransition.from_params(self.P, self.beta)
        b = DynamicTransition.from_params(self.P, self.beta)
        c = DynamicTransition.from_params(self.P, self.beta + 1.0)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, StaticTransition.from_params(self.P))

    def test_update_param_whole_leaf(self):
        transition = DynamicTransition.from_params(self.P, self.beta)
        new = transition.update_param("beta", jnp.zeros_like(self.beta), None)
        self.assertIsInstance(new, DynamicTransition)
        self.assertTrue(jnp.allclose(new.beta, 0.0))
        self.assertTrue(jnp.allclose(new.transition_logits, transition.transition_logits))

    def test_update_param_single_element(self):
        transition = DynamicTransition.from_params(self.P, self.beta)
        new = transition.update_param("beta", 5.0, (0, 1, 1))
        self.assertEqual(float(new.beta[0, 1, 1]), 5.0)
        self.assertEqual(float(new.beta[1, 1, 1]), float(self.beta[1, 1, 1]))


class TestTransitionMatrices(TestCase):
    """
    Tests BaseTransition.transition_matrices: the per-unique-covariate dedup must be
    indistinguishable from computing one matrix per observation.
    """

    def setUp(self) -> None:
        self.K, self.C, self.T = 3, 2, 500
        self.P = jnp.array([[0.8, 0.1, 0.1],
                            [0.1, 0.8, 0.1],
                            [0.1, 0.1, 0.8]])
        rng = np.random.default_rng(1)
        self.beta = jnp.asarray(rng.normal(0, 0.3, (self.C, self.K, self.K - 1)))
        # Few distinct rows, so the dedup actually kicks in (U << T).
        self.xs = jnp.asarray(rng.integers(0, 3, (self.T, self.C)), dtype=float)
        self.dynamic = {
            "DynamicTransition": DynamicTransition.from_params(self.P, self.beta),
            "DynamicTransitionHigherOrder": DynamicTransitionHigherOrder.from_params(self.P, self.beta),
        }

    def test_no_covariates_returns_single_matrix(self):
        transition = StaticTransition.from_params(self.P)
        Gamma = transition.transition_matrix()
        self.assertEqual(Gamma.shape, (self.K, self.K))
        self.assertTrue(jnp.allclose(Gamma, self.P))

    def test_no_covariates_higher_order(self):
        transition = StaticTransitionHigherOrder(jnp.zeros((self.K ** 2, self.K - 1)))
        Gamma = transition.transition_matrix()
        self.assertEqual(Gamma.shape, (self.K ** 2, self.K ** 2))
        self.assertTrue(jnp.allclose(Gamma.sum(axis=1), 1.0))

    def test_static_higher_order_ignores_covariates(self):
        transition = StaticTransitionHigherOrder(jnp.zeros((self.K ** 2, self.K - 1)))
        Gammas = transition.transition_matrices(len(self.xs), self.xs)
        self.assertEqual(Gammas.shape, (self.T, self.K ** 2, self.K ** 2))
        self.assertTrue(jnp.allclose(Gammas, transition.transition_matrix()))

    def test_dynamic_requires_covariates(self):
        for name, transition in self.dynamic.items():
            with self.subTest(name), self.assertRaises(ValueError):
                transition.transition_matrix()

    def test_shape(self):
        for name, transition in self.dynamic.items():
            with self.subTest(name):
                n = transition.transition_matrix(self.xs[0]).shape[0]
                self.assertEqual(transition.transition_matrices(len(self.xs), self.xs).shape, (self.T, n, n))

    def test_rows_are_stochastic(self):
        for name, transition in self.dynamic.items():
            with self.subTest(name):
                Gammas = transition.transition_matrices(len(self.xs), self.xs)
                self.assertTrue(jnp.allclose(Gammas.sum(axis=-1), 1.0))
                self.assertTrue(bool(jnp.all(Gammas >= 0.0)))

    def test_matches_naive_per_observation(self):
        for name, transition in self.dynamic.items():
            with self.subTest(name):
                self.assertTrue(jnp.allclose(
                    transition.transition_matrices(len(self.xs), self.xs),
                    naive_transition_matrices(transition, self.xs)))

    def test_preserves_sequence_order(self):
        transition = self.dynamic["DynamicTransition"]
        Gammas = transition.transition_matrices(len(self.xs), self.xs)
        for t in (0, 1, 17, self.T - 1):
            self.assertTrue(jnp.allclose(Gammas[t], transition.transition_matrix(self.xs[t])))

    def test_all_rows_unique(self):
        # No repeated covariate rows: every matrix must match the naive loop.
        xs = jnp.asarray(np.random.default_rng(2).normal(size=(50, self.C)))
        transition = self.dynamic["DynamicTransition"]
        self.assertTrue(jnp.allclose(transition.transition_matrices(len(xs), xs),
                                     naive_transition_matrices(transition, xs)))

    def test_single_unique_row(self):
        xs = jnp.ones((20, self.C))
        transition = self.dynamic["DynamicTransition"]
        Gammas = transition.transition_matrices(len(xs), xs)
        self.assertEqual(Gammas.shape, (20, self.K, self.K))
        self.assertTrue(jnp.allclose(Gammas, transition.transition_matrix(xs[0])))

    def test_zero_beta_gives_base_matrix(self):
        transition = DynamicTransition.from_params(self.P, jnp.zeros_like(self.beta))
        self.assertTrue(jnp.allclose(transition.transition_matrices(len(self.xs), self.xs), self.P))

    def test_static_ignores_covariates(self):
        transition = StaticTransition.from_params(self.P)
        Gammas = transition.transition_matrices(len(self.xs), self.xs)
        self.assertEqual(Gammas.shape, (self.T, self.K, self.K))
        self.assertTrue(jnp.allclose(Gammas, self.P))

    def test_gradients_match_naive(self):
        # Repeated rows must accumulate gradients exactly like the naive version.
        # Weighted sum rather than log: higher-order matrices have structural zeros.
        weights = jnp.asarray(np.random.default_rng(3).normal(size=(self.T,)))
        loss = lambda Gammas: (weights[:, None, None] * Gammas ** 2).sum()
        for name, transition in self.dynamic.items():
            with self.subTest(name):
                g_dedup = jax.grad(lambda m: loss(m.transition_matrices(len(self.xs), self.xs)))(transition)
                g_naive = jax.grad(lambda m: loss(naive_transition_matrices(m, self.xs)))(transition)
                for a, b in zip(jax.tree.leaves(g_dedup), jax.tree.leaves(g_naive)):
                    self.assertTrue(jnp.allclose(a, b))

    def test_jit_with_closed_over_covariates(self):
        # The solvers jit the loss with the data closed over, so this path must compile.
        transition = self.dynamic["DynamicTransition"]
        Gammas = jax.jit(lambda m: m.transition_matrices(len(self.xs), self.xs))(transition)
        self.assertTrue(jnp.allclose(Gammas, naive_transition_matrices(transition, self.xs)))


class TestDynamicTransitionStep(TestCase):
    """
    Tests the per-observation API that transition_matrices builds on.
    """

    def setUp(self) -> None:
        self.P = jnp.array([[0.9, 0.1], [0.2, 0.8]])
        self.beta = jnp.array([[[0.5], [-0.5]]])  # (C=1, K=2, K-1=1)

    def test_step_shifts_logits_linearly(self):
        transition = DynamicTransition.from_params(self.P, self.beta)
        xt = jnp.array([2.0])
        expected = transition.transition_logits + 2.0 * self.beta[0]
        self.assertTrue(jnp.allclose(transition.step(xt), expected))
        self.assertTrue(jnp.allclose(transition.transition_matrix(xt), logits_to_transition_matrix(expected)))

    def test_invalid_beta_shape_raises(self):
        with self.assertRaises(ValueError):
            DynamicTransition.from_params(self.P, jnp.zeros((1, 3, 2)))

    def test_higher_order_zero_beta_matches_static_lift(self):
        P = jnp.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        transition = DynamicTransitionHigherOrder.from_params(P, jnp.zeros((1, 3, 2)))
        self.assertTrue(jnp.allclose(transition.transition_matrix(jnp.array([3.0])),
                                     transition.base_transition_matrix()))
