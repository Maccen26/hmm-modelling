import jax
jax.config.update("jax_enable_x64", True)

from unittest import TestCase
import numpy as np
import jax.numpy as jnp

from src.api.v5.hmm.params import Params
from src.api.v5.hmm.transitions import (StaticTransition, StaticTransitionHigherOrder,
                                       DynamicTransition, DynamicTransitionHigherOrder)
from src.api.v5.hmm.emissions import GaussEmission, AutoregressiveGaussEmission, MultivariateGaussEmission


class TestParams(TestCase):
    """
    Tests that Params composes a transition and an emission: it delegates to each
    component, counts their trainable parameters, and behaves as one pytree.
    """

    def setUp(self) -> None:
        self.K, self.T, self.C = 3, 20, 2
        self.P = jnp.array([[0.8, 0.1, 0.1],
                            [0.1, 0.8, 0.1],
                            [0.1, 0.1, 0.8]])
        self.mean = jnp.array([0.0, 1.0, 2.0])
        self.sigma = jnp.array([1.0, 0.5, 2.0])

        rng = np.random.default_rng(0)
        self.ys = jnp.asarray(rng.normal(1.0, 1.0, self.T))
        self.xs = jnp.asarray(rng.integers(0, 3, (self.T, self.C)), dtype=float)
        self.beta = jnp.asarray(rng.normal(0, 0.3, (self.C, self.K, self.K - 1)))

        self.transition = StaticTransition.from_params(self.P)
        self.emission = GaussEmission.from_params(self.mean, self.sigma)
        self.params = Params(transition=self.transition, emission=self.emission)

    # --- construction -----------------------------------------------------

    def test_holds_components(self):
        self.assertIs(self.params.transition, self.transition)
        self.assertIs(self.params.emission, self.emission)

    def test_is_iterable(self):
        names = [name for name, _ in self.params]
        self.assertEqual(names, ["transition", "emission"])

    def test_equality_is_true(self):
        other = Params(transition=StaticTransition.from_params(self.P),
                       emission=GaussEmission.from_params(self.mean, self.sigma))
        self.assertEqual(self.params, other)

    def test_equality_is_false(self):
        other = Params(transition=self.transition,
                       emission=GaussEmission.from_params(self.mean + 1.0, self.sigma))
        self.assertNotEqual(self.params, other)

    def test_equality_is_false_for_different_component_class(self):
        other = Params(transition=DynamicTransition.from_params(self.P, self.beta),
                       emission=self.emission)
        self.assertNotEqual(self.params, other)

    # --- parameter count --------------------------------------------------

    def test_len_counts_all_trainable_params(self):
        # transition_logits K*(K-1) + mu0 1 + log_mu_diff K-1 + log_sigma K
        expected = self.K * (self.K - 1) + 1 + (self.K - 1) + self.K
        self.assertEqual(len(self.params), expected)

    def test_len_includes_covariate_and_ar_params(self):
        params = Params(transition=DynamicTransition.from_params(self.P, self.beta),
                        emission=AutoregressiveGaussEmission.from_params(
                            self.mean, self.sigma, jnp.full((2, self.K), 0.5)))
        expected = (self.K * (self.K - 1) + self.beta.size) + (1 + (self.K - 1) + self.K + 2 * self.K)
        self.assertEqual(len(params), expected)

    def test_len_skips_static_order_field(self):
        transition = StaticTransitionHigherOrder(jnp.zeros((self.K ** 2, self.K - 1)))
        params = Params(transition=transition, emission=self.emission)
        expected = self.K ** 2 * (self.K - 1) + 1 + (self.K - 1) + self.K
        self.assertEqual(len(params), expected)

    # --- transition delegation --------------------------------------------

    def test_transition_matrix_delegates(self):
        self.assertTrue(jnp.allclose(self.params.transition_matrix(), self.P))

    def test_transition_matrix_delegates_covariates(self):
        transition = DynamicTransition.from_params(self.P, self.beta)
        params = Params(transition=transition, emission=self.emission)
        xt = self.xs[3]
        self.assertTrue(jnp.allclose(params.transition_matrix(xt), transition.transition_matrix(xt)))

    def test_transition_matrices_without_covariates(self):
        Gammas = self.params.transition_matrices(self.T)
        self.assertEqual(Gammas.shape, (self.T, self.K, self.K))
        self.assertTrue(jnp.allclose(Gammas, self.P))

    def test_transition_matrices_with_covariates(self):
        transition = DynamicTransition.from_params(self.P, self.beta)
        params = Params(transition=transition, emission=self.emission)
        Gammas = params.transition_matrices(self.T, self.xs)
        self.assertEqual(Gammas.shape, (self.T, self.K, self.K))
        self.assertTrue(jnp.allclose(Gammas, transition.transition_matrices(self.T, self.xs)))

    # --- emission delegation ----------------------------------------------

    def test_density_delegates(self):
        for t in (0, 5, self.T - 1):
            self.assertTrue(jnp.allclose(self.params.density(t, self.ys), self.emission.density(t, self.ys)))

    def test_densities_cover_every_observation(self):
        densities = self.params.densities(self.ys)
        self.assertEqual(densities.shape, (self.T, 1, self.K))
        for t in range(self.T):
            self.assertTrue(jnp.allclose(densities[t], self.params.density(t, self.ys)))

    def test_cdf_delegates(self):
        for t in (0, 5, self.T - 1):
            self.assertTrue(jnp.allclose(self.params.cdf(t, self.ys), self.emission.cdf(t, self.ys)))

    def test_cdfs_cover_every_observation(self):
        cdfs = self.params.cdfs(self.ys)
        self.assertEqual(cdfs.shape, (self.T, 1, self.K))
        for t in range(self.T):
            self.assertTrue(jnp.allclose(cdfs[t], self.params.cdf(t, self.ys)))

    # --- pytree behaviour -------------------------------------------------

    def test_gradients_reach_both_components(self):
        # The solvers differentiate one loss w.r.t. the whole Params pytree.
        def loss(p):
            return jnp.log(p.densities(self.ys)).sum() + jnp.log(p.transition_matrices(self.T)).sum()

        grads = jax.grad(loss)(self.params)
        self.assertTrue(bool(jnp.any(grads.transition.transition_logits != 0.0)))
        self.assertTrue(bool(jnp.any(grads.emission.log_sigma != 0.0)))
        for leaf in jax.tree.leaves(grads):
            self.assertTrue(bool(jnp.all(jnp.isfinite(leaf))))

    def test_jit(self):
        densities = jax.jit(lambda p: p.densities(self.ys))(self.params)
        self.assertTrue(jnp.allclose(densities, self.params.densities(self.ys)))

    def test_jit_transition_matrices(self):
        Gammas = jax.jit(lambda p: p.transition_matrices(self.T))(self.params)
        self.assertTrue(jnp.allclose(Gammas, self.params.transition_matrices(self.T)))

    # --- other component combinations -------------------------------------

    def test_transition_matrices_rows_sum_to_one(self):
        params = Params(transition=DynamicTransition.from_params(self.P, self.beta),
                        emission=self.emission)
        Gammas = params.transition_matrices(self.T, self.xs)
        self.assertTrue(jnp.allclose(Gammas.sum(axis=-1), 1.0))

    def test_higher_order_transition_matrices_shape(self):
        params = Params(transition=StaticTransitionHigherOrder(jnp.zeros((self.K ** 2, self.K - 1))),
                        emission=self.emission)
        Gammas = params.transition_matrices(self.T)
        self.assertEqual(Gammas.shape, (self.T, self.K ** 2, self.K ** 2))
        self.assertTrue(jnp.allclose(Gammas.sum(axis=-1), 1.0))

    def test_dynamic_higher_order_transition_matrices_shape(self):
        transition = DynamicTransitionHigherOrder.from_params(self.P, self.beta)
        params = Params(transition=transition, emission=self.emission)
        Gammas = params.transition_matrices(self.T, self.xs)
        self.assertEqual(Gammas.shape, (self.T, self.K ** 2, self.K ** 2))
        self.assertTrue(jnp.allclose(Gammas, transition.transition_matrices(self.T, self.xs)))

    def test_ar_densities_are_one_before_first_lag(self):
        num_lags = 2
        emission = AutoregressiveGaussEmission.from_params(self.mean, self.sigma, jnp.full((num_lags, self.K), 0.5))
        params = Params(transition=self.transition, emission=emission)
        densities = params.densities(self.ys)
        self.assertEqual(densities.shape, (self.T, 1, self.K))
        self.assertTrue(jnp.allclose(densities[:num_lags], 1.0))
        self.assertTrue(jnp.allclose(densities[num_lags:], emission.densities(jnp.arange(num_lags, self.T), self.ys)))

    def test_multivariate_densities_shape(self):
        ys = jnp.stack([self.ys, 2.0 * self.ys], axis=1)  # (T, 2)
        emission = MultivariateGaussEmission.from_params(jnp.stack([self.mean, 2.0 * self.mean]),
                                                         jnp.stack([self.sigma, self.sigma]))
        params = Params(transition=self.transition, emission=emission)
        self.assertEqual(params.densities(ys).shape, (self.T, 1, self.K))
        self.assertEqual(params.cdfs(ys).shape, (self.T, 1, self.K))
