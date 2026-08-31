from unittest import TestCase

from scipy import stats 
from src.api.v4 import GaussEmission 
import jax.numpy as jnp


class TestGaussEmission(TestCase):
    """
    Should test build and computation of emission parameters, density and cdf.
    """
    
    def setUp(self) -> None:
        self.mean = jnp.array([0.0, 1.0, 2.0]) 
        self.sigma = jnp.array([1.0, 1.0, 1.0])

        self.mu0 = jnp.array(self.mean[0])
        self.log_mu_diff = jnp.log(jnp.array([self.mean[i] - self.mean[i-1] for i in range(1, len(self.mean))]))
        self.log_sigma = jnp.log(self.sigma)

        self.yt = jnp.array([0.0, 1.0, 2.0])  # 1-dimensional observation for testing

    def test_build(self):
        emission = GaussEmission(mu0=self.mu0, log_mu_diff=self.log_mu_diff, log_sigma=self.log_sigma)

        self.assertTrue(jnp.allclose(emission.mu0, self.mu0))
        self.assertTrue(jnp.allclose(emission.log_mu_diff, self.log_mu_diff))
        self.assertTrue(jnp.allclose(emission.log_sigma, self.log_sigma))

    def test_build_from_params(self):
        emission = GaussEmission.from_params(self.mean, self.sigma)

        self.assertTrue(jnp.allclose(emission.mu0, self.mu0))
        self.assertTrue(jnp.allclose(emission.log_mu_diff, self.log_mu_diff))
        self.assertTrue(jnp.allclose(emission.log_sigma, self.log_sigma))

    def test_iterable(self):
        emission = GaussEmission.from_params(self.mean, self.sigma)
        try:
            params = list(emission)
        except Exception as e:
            self.fail(f"Iterating over GaussEmission parameters failed with error: {e}") 

        self.assertEqual(len(params), 3)  # Should have three parameters: mu0, log_mu_diff, log_sigma
        param_names = [name for name, _ in params]
        self.assertIn('mu0', param_names)
        self.assertIn('log_mu_diff', param_names)
        self.assertIn('log_sigma', param_names)
    def test_equality_is_true(self):
        emission1 = GaussEmission.from_params(self.mean, self.sigma)
        emission2 = GaussEmission.from_params(self.mean, self.sigma)

        self.assertEqual(emission1, emission2)  # Should be equal since they have the same parameters

    def test_equality_is_false(self):
        different_mean = jnp.array([1.0, 2.0, 3.0]) 
        different_sigma = jnp.array([1.0, 1.0, 1.0])
        different_emission = GaussEmission.from_params(different_mean, different_sigma)

        emission = GaussEmission.from_params(self.mean, self.sigma)

        self.assertNotEqual(emission, different_emission)  # Should not be equal since they have different parameters

    def test_density_success(self):
        emission = GaussEmission.from_params(self.mean, self.sigma)
        try: 
            density = emission.density(0, self.yt)
            self.assertTrue(density.shape == (1, len(self.mean))) 
        except Exception as e:
            self.fail(f"Density computation failed with error: {e}") 

    def test_density_correctness(self):
        emission = GaussEmission.from_params(self.mean, self.sigma)
        yt = jnp.array([0.0])
        density = emission.density(0, self.yt)
        expected_density = stats.norm.pdf(yt, loc=self.mean, scale=self.sigma)
        self.assertTrue(jnp.allclose(density, expected_density))


    def test_density_dim(self): 
        y = jnp.array([0.0])
        emission = GaussEmission.from_params(self.mean, self.sigma)
        density = emission.density(0, self.yt)
        self.assertTrue(density.shape == (1, len(self.mean))) #Every row is a new obs  

    def test_cdf_success(self):
        emission = GaussEmission.from_params(self.mean, self.sigma)
        try:
            yt = jnp.array([0.0])
            cdf = emission.cdf(0, self.yt)
            self.assertTrue(cdf.shape == (1, len(self.mean))) #Every row is a new obs
            self.assertTrue(jnp.isclose(cdf[0, 0], 0.5)) # CDF at mean should be 0.5
        except Exception as e:
            self.fail(f"CDF computation failed with error: {e}")

    def test_cdf_correctness(self):
        from scipy import stats as scipy_stats
        emission = GaussEmission.from_params(self.mean, self.sigma)
        yt = jnp.array([1.0])
        cdf = emission.cdf(0, yt)
        expected = scipy_stats.norm.cdf(yt, loc=self.mean, scale=self.sigma)
        self.assertTrue(jnp.allclose(cdf, expected))

    def test_cdf_values_between_0_and_1(self):
        emission = GaussEmission.from_params(self.mean, self.sigma)
        cdf = emission.cdf(0, self.yt)
        self.assertTrue(jnp.all(cdf >= 0.0))
        self.assertTrue(jnp.all(cdf <= 1.0))

    def test_mu_is_monotonically_increasing(self):
        emission = GaussEmission.from_params(self.mean, self.sigma)
        mu = emission.mu(0, self.yt)
        diffs = jnp.diff(mu)
        self.assertTrue(jnp.all(diffs > 0))

    def test_sigma_is_positive(self):
        emission = GaussEmission.from_params(self.mean, self.sigma)
        sigma = emission.sigma(0, self.yt)
        self.assertTrue(jnp.all(sigma > 0))





