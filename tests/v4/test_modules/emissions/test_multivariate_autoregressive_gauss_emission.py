from unittest import TestCase

from scipy import stats
import jax.numpy as jnp

from src.api.v4 import AutoregressiveGaussEmission, HMM, StaticTransition
from src.api.v4.utils import phi_to_phi_tilde
from src.api.v4.emissions.multivariate_autoregressive_gauss_emission import (
    MultivariateAutoregressiveGaussEmission,
)


class TestMultivariateAutoregressiveGaussEmission(TestCase):
    """
    Should test build and computation of emission parameters, density and cdf for
    observations of shape (T, K) with a per-dimension AR recursion.
    """

    def setUp(self) -> None:
        # (K=2 dimensions, N=3 states), means increasing along the state axis
        self.mean = jnp.array([[0.0, 1.0, 2.0],
                               [1.0, 3.0, 6.0]])
        self.sigma = jnp.array([[1.0, 1.0, 1.0],
                                [2.0, 2.0, 2.0]])
        # (k=2 lags, K=2 dimensions, N=3 states)
        self.phi = jnp.array([[[0.5, 0.5, 0.5], [0.4, 0.4, 0.4]],
                              [[0.3, 0.3, 0.3], [0.2, 0.2, 0.2]]])
        self.num_lags, self.num_dims, self.num_states = self.phi.shape

        self.mu0 = self.mean[:, 0]
        self.log_mu_diff = jnp.log(jnp.diff(self.mean, axis=1))
        self.log_sigma = jnp.log(self.sigma)
        self.phi_tilde = phi_to_phi_tilde(self.phi)

        # (T=5, K=2) observation sequence
        self.ys = jnp.array([[0.0, 1.0],
                             [1.5, 4.0],
                             [2.0, 6.0],
                             [1.0, 3.0],
                             [0.5, 2.0]])

    def build(self, phi=None) -> MultivariateAutoregressiveGaussEmission:
        return MultivariateAutoregressiveGaussEmission.from_params(
            self.mean, self.sigma, self.phi if phi is None else phi
        )

    # --- construction -----------------------------------------------------

    def test_build(self):
        emission = MultivariateAutoregressiveGaussEmission(
            log_mu_diff=self.log_mu_diff, mu0=self.mu0,
            log_sigma=self.log_sigma, phi_tilde=self.phi_tilde,
        )

        self.assertTrue(jnp.allclose(emission.mu0, self.mu0))
        self.assertTrue(jnp.allclose(emission.log_mu_diff, self.log_mu_diff))
        self.assertTrue(jnp.allclose(emission.log_sigma, self.log_sigma))
        self.assertTrue(jnp.allclose(emission.phi_tilde, self.phi_tilde))

    def test_build_from_params(self):
        emission = self.build()

        self.assertTrue(jnp.allclose(emission.mu0, self.mu0))
        self.assertTrue(jnp.allclose(emission.log_mu_diff, self.log_mu_diff))
        self.assertTrue(jnp.allclose(emission.log_sigma, self.log_sigma))
        self.assertTrue(jnp.allclose(emission.phi_tilde, self.phi_tilde))

    def test_param_shapes(self):
        emission = self.build()

        self.assertEqual(emission.mu0.shape, (self.num_dims,))
        self.assertEqual(emission.log_mu_diff.shape, (self.num_dims, self.num_states - 1))
        self.assertEqual(emission.log_sigma.shape, (self.num_dims, self.num_states))
        self.assertEqual(emission.phi_tilde.shape, self.phi.shape)

    def test_two_dimensional_phi_is_read_as_one_dimension(self):
        """A (k, N) phi means K=1, so the univariate spelling keeps working."""
        phi = jnp.array([[0.5, 0.5, 0.5]])
        emission = MultivariateAutoregressiveGaussEmission.from_params(
            self.mean[:1], self.sigma[:1], phi
        )

        self.assertEqual(emission.phi_tilde.shape, (1, 1, self.num_states))

    def test_iterable(self):
        """update_param and parameter freezing rely on the named fields."""
        params = list(self.build())

        self.assertEqual(len(params), 4)
        param_names = [name for name, _ in params]
        for expected in ('mu0', 'log_mu_diff', 'log_sigma', 'phi_tilde'):
            self.assertIn(expected, param_names)

    def test_equality_is_true(self):
        self.assertEqual(self.build(), self.build())

    def test_equality_is_false(self):
        different = MultivariateAutoregressiveGaussEmission.from_params(
            self.mean, self.sigma, self.phi * 0.5
        )

        self.assertNotEqual(self.build(), different)

    # --- parameters -------------------------------------------------------

    def test_mu_vals_reconstructs_mean_matrix(self):
        mu_vals = self.build().mu_vals(0, self.ys)

        self.assertEqual(mu_vals.shape, (self.num_dims, self.num_states))
        self.assertTrue(jnp.allclose(mu_vals, self.mean))

    def test_mu_vals_monotonic_per_dimension(self):
        mu_vals = self.build().mu_vals(0, self.ys)

        self.assertTrue(jnp.all(jnp.diff(mu_vals, axis=1) > 0))

    def test_mu_vals_tiles_to_the_augmented_state_space(self):
        """A higher-order transition ties one mean across the augmented states."""
        sigma_6 = jnp.ones((self.num_dims, 2 * self.num_states))
        phi_6 = jnp.full((1, self.num_dims, 2 * self.num_states), 0.5)
        emission = MultivariateAutoregressiveGaussEmission.from_params(
            self.mean, sigma_6, phi_6
        )

        mu_vals = emission.mu_vals(0, self.ys)

        self.assertEqual(mu_vals.shape, (self.num_dims, 2 * self.num_states))
        self.assertTrue(jnp.allclose(mu_vals, jnp.tile(self.mean, (1, 2))))

    def test_sigma_is_positive(self):
        sigma = self.build().sigma(0, self.ys)

        self.assertEqual(sigma.shape, (self.num_dims, self.num_states))
        self.assertTrue(jnp.all(sigma > 0))
        self.assertTrue(jnp.allclose(sigma, self.sigma))

    def test_phi_round_trips_and_stays_in_unit_interval(self):
        phi = self.build().phi()

        self.assertEqual(phi.shape, self.phi.shape)
        self.assertTrue(jnp.allclose(phi, self.phi))
        self.assertTrue(jnp.all(jnp.abs(phi) < 1.0))

    # --- lags and the AR mean ---------------------------------------------

    def test_lags_are_the_previous_observations_most_recent_first(self):
        k, t = 2, 3
        lags = MultivariateAutoregressiveGaussEmission.lags(t, self.ys, k)

        self.assertEqual(lags.shape, (k, self.num_dims))
        self.assertTrue(jnp.allclose(lags[0], self.ys[t - 1]))
        self.assertTrue(jnp.allclose(lags[1], self.ys[t - 2]))

    def test_mu_equals_base_mu_when_lags_are_incomplete(self):
        emission = self.build()

        for t in range(self.num_lags):
            self.assertTrue(
                jnp.allclose(emission.mu(t, self.ys), emission.mu_vals(t, self.ys)),
                f"at t={t} < k, mu should equal base_mu (no AR term)",
            )

    def test_zero_phi_gives_base_mu(self):
        emission = self.build(phi=jnp.zeros_like(self.phi))

        for t in range(len(self.ys)):
            self.assertTrue(
                jnp.allclose(emission.mu(t, self.ys), emission.mu_vals(t, self.ys), atol=1e-5),
                f"with phi=0, mu at t={t} should equal base_mu",
            )

    def test_mu_matches_the_ar_recursion(self):
        """mu[j, i] = m[j, i] + sum_l phi[l, j, i] * (y[t-l, j] - m[j, i])."""
        t = 4
        emission = self.build()
        base_mu = emission.mu_vals(t, self.ys)

        expected = base_mu
        for lag in range(1, self.num_lags + 1):
            lagged = self.ys[t - lag][:, None]                 # (K, 1)
            expected = expected + self.phi[lag - 1] * (lagged - base_mu)

        self.assertTrue(jnp.allclose(emission.mu(t, self.ys), expected))

    def test_single_lag_uses_only_the_previous_observation(self):
        phi = jnp.array([[[0.8, 0.8, 0.8], [0.6, 0.6, 0.6]]])  # k=1
        emission = self.build(phi=phi)
        t = 2

        base_mu = emission.mu_vals(t, self.ys)
        expected = base_mu + phi[0] * (self.ys[t - 1][:, None] - base_mu)

        self.assertTrue(jnp.allclose(emission.mu(t, self.ys), expected))

    def test_dimensions_do_not_read_each_others_history(self):
        """The AR term is per-dimension; coupling runs through the hidden state."""
        emission = self.build()
        t = 4
        perturbed = self.ys.at[t - 1, 1].add(100.0)  # only dimension 1's past

        before = emission.marginal_densities(t, self.ys)
        after = emission.marginal_densities(t, perturbed)

        self.assertTrue(jnp.allclose(before[0], after[0]),
                        "dimension 0's density changed when only dimension 1's past moved")
        self.assertFalse(jnp.allclose(before[1], after[1]),
                         "dimension 1's density should respond to its own past")

    # --- densities --------------------------------------------------------

    def test_marginal_densities_shape_and_values(self):
        t = 3
        emission = self.build()
        mu = emission.mu(t, self.ys)
        marginals = emission.marginal_densities(t, self.ys)
        expected = stats.norm.pdf(self.ys[t][:, None], loc=mu, scale=self.sigma)

        self.assertEqual(marginals.shape, (self.num_dims, self.num_states))
        self.assertTrue(jnp.allclose(marginals, expected))

    def test_density_shape_consistent(self):
        """(1, num_states) regardless of t < k or t >= k, as ForwardAlgorithm needs."""
        emission = self.build()

        for t in range(len(self.ys)):
            self.assertEqual(emission.density(t, self.ys).shape, (1, self.num_states))

    def test_density_is_product_over_dimensions(self):
        t = 3
        emission = self.build()
        marginals = emission.marginal_densities(t, self.ys)

        self.assertTrue(jnp.allclose(
            emission.density(t, self.ys), jnp.prod(marginals, axis=0, keepdims=True)
        ))

    def test_density_is_one_for_t_less_than_k(self):
        """Density 1 contributes exactly 0 to the log-likelihood."""
        emission = self.build()

        for t in range(self.num_lags):
            density = emission.density(t, self.ys)
            self.assertTrue(jnp.allclose(density, jnp.ones_like(density)),
                            f"density at t={t} should be 1.0 when t < k")

    def test_density_is_not_one_for_t_geq_k(self):
        emission = self.build()

        for t in range(self.num_lags, len(self.ys)):
            density = emission.density(t, self.ys)
            self.assertFalse(jnp.allclose(density, jnp.ones_like(density)),
                             f"density at t={t} should NOT be 1.0 when t >= k")

    def test_density_reduces_to_univariate_for_one_dimension(self):
        mean, sigma = jnp.array([0.0, 1.0, 2.0]), jnp.array([1.0, 1.0, 1.0])
        phi = jnp.array([[0.5, 0.5, 0.5], [0.3, 0.3, 0.3]])  # (k=2, N=3)
        univariate = AutoregressiveGaussEmission.from_params(mean, sigma, phi)
        multivariate = MultivariateAutoregressiveGaussEmission.from_params(
            mean[None, :], sigma[None, :], phi
        )
        ys = jnp.array([0.5, 1.5, 2.5, 1.0, 0.0])

        for t in range(len(ys)):
            self.assertTrue(
                jnp.allclose(multivariate.density(t, ys[:, None]), univariate.density(t, ys)),
                f"multivariate density differs from the univariate one at t={t}",
            )

    def test_densities_batched_shape(self):
        """The inherited vmap default maps over t only and keeps the (1, N) rows."""
        emission = self.build()
        indices = jnp.arange(len(self.ys))
        densities = emission.densities(indices, self.ys)

        self.assertEqual(densities.shape, (len(self.ys), 1, self.num_states))
        for t in indices:
            self.assertTrue(jnp.allclose(densities[t], emission.density(t, self.ys)))

    # --- cdf --------------------------------------------------------------

    def test_cdf_shape_and_product(self):
        t = 3
        emission = self.build()
        mu = emission.mu(t, self.ys)
        expected = jnp.prod(
            stats.norm.cdf(self.ys[t][:, None], loc=mu, scale=self.sigma), axis=0
        )

        cdf = emission.cdf(t, self.ys)

        self.assertEqual(cdf.shape, (1, self.num_states))
        self.assertTrue(jnp.allclose(cdf[0], expected))

    def test_cdf_between_0_and_1(self):
        emission = self.build()

        for t in range(len(self.ys)):
            cdf = emission.cdf(t, self.ys)
            self.assertTrue(jnp.all(cdf >= 0.0))
            self.assertTrue(jnp.all(cdf <= 1.0))

    # --- integration ------------------------------------------------------

    def test_forward_pass_log_likelihood_is_finite(self):
        """The joint density really does flow through the forward algorithm."""
        mean = jnp.array([[0.0, 5.0], [1.0, 3.0]])
        sigma = jnp.array([[1.0, 1.0], [1.0, 1.0]])
        phi = jnp.array([[[0.5, 0.5], [0.4, 0.4]]])
        emission = MultivariateAutoregressiveGaussEmission.from_params(mean, sigma, phi)
        transition = StaticTransition.from_params(jnp.array([[0.9, 0.1], [0.2, 0.8]]))
        hmm = HMM(transition=transition, emission=emission)

        ll = hmm.log_likelihood(self.ys)

        self.assertTrue(jnp.isfinite(ll), f"log-likelihood was {ll}")
