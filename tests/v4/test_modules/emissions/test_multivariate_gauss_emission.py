from unittest import TestCase

from scipy import stats
import jax.numpy as jnp

from src.api.v4 import GaussEmission, HMM, StaticTransition
from src.api.v4.emissions.multivariate_gauss_emission import MultivariateGaussEmission


class TestMultivariateGaussEmission(TestCase):
    """
    Should test build and computation of emission parameters, density and cdf for
    observations of shape (T, K).
    """

    def setUp(self) -> None:
        # (K=2 dimensions, N=3 states), means increasing along the state axis
        self.mean = jnp.array([[0.0, 1.0, 2.0],
                               [1.0, 3.0, 6.0]])
        self.sigma = jnp.array([[1.0, 1.0, 1.0],
                                [2.0, 2.0, 2.0]])
        self.num_dims, self.num_states = self.mean.shape

        self.mu0 = self.mean[:, 0]
        self.log_mu_diff = jnp.log(jnp.diff(self.mean, axis=1))
        self.log_sigma = jnp.log(self.sigma)

        # (T=3, K=2) observation sequence
        self.ys = jnp.array([[0.0, 1.0],
                             [1.5, 4.0],
                             [2.0, 6.0]])

    def build(self) -> MultivariateGaussEmission:
        return MultivariateGaussEmission.from_params(self.mean, self.sigma)

    def test_build(self):
        emission = MultivariateGaussEmission(
            mu0=self.mu0, log_mu_diff=self.log_mu_diff, log_sigma=self.log_sigma
        )

        self.assertTrue(jnp.allclose(emission.mu0, self.mu0))
        self.assertTrue(jnp.allclose(emission.log_mu_diff, self.log_mu_diff))
        self.assertTrue(jnp.allclose(emission.log_sigma, self.log_sigma))

    def test_build_from_params(self):
        emission = self.build()

        self.assertTrue(jnp.allclose(emission.mu0, self.mu0))
        self.assertTrue(jnp.allclose(emission.log_mu_diff, self.log_mu_diff))
        self.assertTrue(jnp.allclose(emission.log_sigma, self.log_sigma))

    def test_param_shapes(self):
        emission = self.build()

        self.assertEqual(emission.mu0.shape, (self.num_dims,))
        self.assertEqual(emission.log_mu_diff.shape, (self.num_dims, self.num_states - 1))
        self.assertEqual(emission.log_sigma.shape, (self.num_dims, self.num_states))

    def test_mu_reconstructs_mean_matrix(self):
        """mu() must rebuild the (K, N) means it was constructed from."""
        mu = self.build().mu(0, self.ys)

        self.assertEqual(mu.shape, (self.num_dims, self.num_states))
        self.assertTrue(jnp.allclose(mu, self.mean))

    def test_mu_monotonic_per_dimension(self):
        mu = self.build().mu(0, self.ys)

        self.assertTrue(jnp.all(jnp.diff(mu, axis=1) > 0))

    def test_sigma_is_positive(self):
        sigma = self.build().sigma(0, self.ys)

        self.assertEqual(sigma.shape, (self.num_dims, self.num_states))
        self.assertTrue(jnp.all(sigma > 0))
        self.assertTrue(jnp.allclose(sigma, self.sigma))

    def test_marginal_densities_shape_and_values(self):
        t = 1
        marginals = self.build().marginal_densities(t, self.ys)
        expected = stats.norm.pdf(self.ys[t][:, None], loc=self.mean, scale=self.sigma)

        self.assertEqual(marginals.shape, (self.num_dims, self.num_states))
        self.assertTrue(jnp.allclose(marginals, expected))

    def test_density_shape(self):
        """(1, num_states) is what ForwardAlgorithm consumes."""
        density = self.build().density(0, self.ys)

        self.assertEqual(density.shape, (1, self.num_states))

    def test_density_is_product_over_dimensions(self):
        t = 1
        emission = self.build()
        density = emission.density(t, self.ys)
        marginals = emission.marginal_densities(t, self.ys)
        expected = jnp.prod(
            stats.norm.pdf(self.ys[t][:, None], loc=self.mean, scale=self.sigma), axis=0
        )

        self.assertTrue(jnp.allclose(density, jnp.prod(marginals, axis=0, keepdims=True)))
        self.assertTrue(jnp.allclose(density[0], expected))

    def test_density_reduces_to_univariate_for_one_dimension(self):
        mean, sigma = jnp.array([0.0, 1.0, 2.0]), jnp.array([1.0, 1.0, 1.0])
        univariate = GaussEmission.from_params(mean, sigma)
        multivariate = MultivariateGaussEmission.from_params(mean[None, :], sigma[None, :])
        ys = jnp.array([0.5, 1.5, 2.5])

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

    def test_cdf_shape_and_product(self):
        t = 1
        cdf = self.build().cdf(t, self.ys)
        expected = jnp.prod(
            stats.norm.cdf(self.ys[t][:, None], loc=self.mean, scale=self.sigma), axis=0
        )

        self.assertEqual(cdf.shape, (1, self.num_states))
        self.assertTrue(jnp.allclose(cdf[0], expected))

    def test_cdf_between_0_and_1(self):
        cdf = self.build().cdf(0, self.ys)

        self.assertTrue(jnp.all(cdf >= 0.0))
        self.assertTrue(jnp.all(cdf <= 1.0))

    def test_iterable(self):
        """update_param and parameter freezing rely on the named fields."""
        params = list(self.build())

        self.assertEqual(len(params), 3)
        param_names = [name for name, _ in params]
        self.assertIn('mu0', param_names)
        self.assertIn('log_mu_diff', param_names)
        self.assertIn('log_sigma', param_names)

    def test_equality_is_true(self):
        self.assertEqual(self.build(), self.build())

    def test_equality_is_false(self):
        different = MultivariateGaussEmission.from_params(self.mean + 1.0, self.sigma)

        self.assertNotEqual(self.build(), different)

    def test_forward_pass_log_likelihood_is_finite(self):
        """The joint density really does flow through the forward algorithm."""
        mean = jnp.array([[0.0, 5.0], [1.0, 3.0]])
        sigma = jnp.array([[1.0, 1.0], [1.0, 1.0]])
        emission = MultivariateGaussEmission.from_params(mean, sigma)
        transition = StaticTransition.from_params(jnp.array([[0.9, 0.1], [0.2, 0.8]]))
        hmm = HMM(transition=transition, emission=emission)

        ll = hmm.log_likelihood(self.ys)

        self.assertTrue(jnp.isfinite(ll), f"log-likelihood was {ll}")
