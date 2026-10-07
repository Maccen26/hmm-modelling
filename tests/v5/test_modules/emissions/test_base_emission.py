import jax
jax.config.update("jax_enable_x64", True)

import inspect
from unittest import TestCase
import jax.numpy as jnp

from src.api.v5.base import BaseEmission
from src.api.v5.hmm.emissions import (
    GaussEmission,
    AutoregressiveGaussEmission,
    MultivariateGaussEmission,
    MultivariateAutoregressiveGaussEmission,
)


class TestBaseEmissionInterface(TestCase):
    """
    Tests the contract every v5 emission shares through BaseEmission: methods take
    only (t, ys), and the batched densities/cdfs stack the per-step results.
    """

    def setUp(self) -> None:
        self.T = 8
        ys = jnp.linspace(0.0, 3.0, self.T)
        self.emissions = {
            "GaussEmission": (
                GaussEmission.from_params(jnp.array([0.0, 2.0]), jnp.array([1.0, 1.0])), ys),
            "AutoregressiveGaussEmission": (
                AutoregressiveGaussEmission.from_params(
                    jnp.array([0.0, 2.0]), jnp.array([1.0, 1.0]), jnp.array([[0.5, 0.3]])), ys),
            "MultivariateGaussEmission": (
                MultivariateGaussEmission.from_params(
                    jnp.array([[0.0, 2.0], [1.0, 3.0]]), jnp.ones((2, 2))), jnp.stack([ys, 2 * ys], axis=1)),
            "MultivariateAutoregressiveGaussEmission": (
                MultivariateAutoregressiveGaussEmission.from_params(
                    jnp.array([[0.0, 2.0], [1.0, 3.0]]), jnp.ones((2, 2)), jnp.full((1, 2, 2), 0.4)),
                jnp.stack([ys, 2 * ys], axis=1)),
        }

    def test_cannot_instantiate_abstract_base(self):
        with self.assertRaises(TypeError):
            BaseEmission()

    def test_all_emissions_subclass_v5_base(self):
        for name, (emission, _) in self.emissions.items():
            with self.subTest(name):
                self.assertIsInstance(emission, BaseEmission)

    def test_methods_take_only_t_and_ys(self):
        for name, (emission, _) in self.emissions.items():
            for method in ("density", "cdf", "mu", "step", "sigma"):
                with self.subTest(name, method=method):
                    params = list(inspect.signature(getattr(emission, method)).parameters)
                    self.assertEqual(params, ["t", "ys"])

    def test_densities_stack_per_step_density(self):
        for name, (emission, ys) in self.emissions.items():
            with self.subTest(name):
                densities = emission.densities(jnp.arange(self.T), ys)
                self.assertEqual(densities.shape, (self.T, 1, 2))
                for t in range(self.T):
                    self.assertTrue(jnp.allclose(densities[t], emission.density(t, ys)))

    def test_cdfs_stack_per_step_cdf(self):
        for name, (emission, ys) in self.emissions.items():
            with self.subTest(name):
                cdfs = emission.cdfs(jnp.arange(self.T), ys)
                self.assertEqual(cdfs.shape, (self.T, 1, 2))
                for t in range(self.T):
                    self.assertTrue(jnp.allclose(cdfs[t], emission.cdf(t, ys)))

    def test_densities_jit(self):
        # The solvers jit the loss with the data closed over, so this path must compile.
        for name, (emission, ys) in self.emissions.items():
            with self.subTest(name):
                jitted = jax.jit(lambda e: e.densities(jnp.arange(self.T), ys))(emission)
                self.assertTrue(jnp.allclose(jitted, emission.densities(jnp.arange(self.T), ys)))

    def test_update_param_returns_same_class(self):
        for name, (emission, _) in self.emissions.items():
            with self.subTest(name):
                new = emission.update_param("log_sigma", jnp.zeros_like(emission.log_sigma), None)
                self.assertIsInstance(new, type(emission))
                self.assertTrue(jnp.allclose(new.log_sigma, 0.0))
                self.assertTrue(jnp.allclose(new.mu0, emission.mu0))
