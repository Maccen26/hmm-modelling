import jax
jax.config.update("jax_enable_x64", True)

import itertools
from unittest import TestCase
import numpy as np
import jax.numpy as jnp

from src.api.v5.hmm.params import Params
from src.api.v5.hmm.transitions import StaticTransition, DynamicTransition
from src.api.v5.hmm.emissions import GaussEmission
from src.api.v5.inference.forward_algorithm import ForwardAlgorithm
from src.api.v5.inference.forward_backward import ForwardBackwardAlgorithm


def brute_force_smoothed(params: Params, u0: jnp.ndarray, ys: jnp.ndarray,
                         xs: jnp.ndarray | None = None, priors: np.ndarray | None = None) -> np.ndarray:
    """
    p(z_t = i | y_{0:T-1}) for every t, by enumerating every state path.

    Uses the forward recursion's convention: Gamma_0 is applied to u0 before y_0,
    and prior_densities[t] multiplies the state probabilities at time t.
    """
    T = len(ys)
    Gammas = np.asarray(params.transition_matrices(T, xs))
    gs = np.asarray(params.densities(ys))[:, 0, :]  # (T, K)
    delta = np.asarray(u0).ravel()
    K = len(delta)
    priors = np.ones((T, K)) if priors is None else np.broadcast_to(priors, (T, K))

    marginals = np.zeros((T, K))
    for path in itertools.product(range(K), repeat=T):
        prob = (delta @ Gammas[0])[path[0]] * priors[0, path[0]] * gs[0, path[0]]
        for t in range(1, T):
            prob *= Gammas[t, path[t - 1], path[t]] * priors[t, path[t]] * gs[t, path[t]]
        marginals[np.arange(T), path] += prob
    return marginals / marginals.sum(axis=1, keepdims=True)


class TestForwardBackwardAlgorithm(TestCase):
    """
    Unit tests for ForwardBackwardAlgorithm: s_t(i) = p(z_t = i | y_{0:T-1}).
    """

    def setUp(self) -> None:
        self.K, self.T, self.C = 3, 5, 2
        self.P = jnp.array([[0.80, 0.15, 0.05],
                            [0.10, 0.70, 0.20],
                            [0.25, 0.25, 0.50]])
        self.mean = jnp.array([-1.0, 0.5, 2.0])
        self.sigma = jnp.array([0.8, 1.0, 0.6])
        self.u0 = jnp.array([[0.5, 0.3, 0.2]])

        rng = np.random.default_rng(0)
        self.ys = jnp.asarray(rng.normal(0.5, 1.5, self.T))
        self.xs = jnp.asarray(rng.integers(0, 3, (self.T, self.C)), dtype=float)
        self.beta = jnp.asarray(rng.normal(0, 0.4, (self.C, self.K, self.K - 1)))

        self.params = Params(transition=StaticTransition.from_params(self.P),
                             emission=GaussEmission.from_params(self.mean, self.sigma))
        self.dynamic_params = Params(transition=DynamicTransition.from_params(self.P, self.beta),
                                     emission=self.params.emission)

    def smoothed(self, params: Params, ys: jnp.ndarray, xs=None, prior_densities=None, u0=None) -> np.ndarray:
        u0 = self.u0 if u0 is None else u0
        return np.asarray(ForwardBackwardAlgorithm.run(params, u0, ys, xs=xs, prior_densities=prior_densities))

    # --- output structure -------------------------------------------------

    def test_output_shape(self):
        self.assertEqual(self.smoothed(self.params, self.ys).shape, (self.T, self.K))

    def test_rows_are_distributions(self):
        s = self.smoothed(self.params, self.ys)
        self.assertTrue(np.all(s >= 0))
        np.testing.assert_allclose(s.sum(axis=1), np.ones(self.T), rtol=1e-12)

    def test_long_sequence_is_finite(self):
        ys = jnp.asarray(np.random.default_rng(3).normal(0.5, 1.5, 5000))
        s = self.smoothed(self.params, ys)
        self.assertTrue(np.all(np.isfinite(s)))
        np.testing.assert_allclose(s.sum(axis=1), np.ones(len(ys)), rtol=1e-12)

    # --- correctness ------------------------------------------------------

    def test_matches_brute_force_posterior(self):
        expected = brute_force_smoothed(self.params, self.u0, self.ys)
        np.testing.assert_allclose(self.smoothed(self.params, self.ys), expected, rtol=1e-8)

    def test_dynamic_transition_matches_brute_force_posterior(self):
        expected = brute_force_smoothed(self.dynamic_params, self.u0, self.ys, self.xs)
        np.testing.assert_allclose(self.smoothed(self.dynamic_params, self.ys, xs=self.xs), expected, rtol=1e-8)

    def test_state_dependent_prior_densities_match_brute_force(self):
        # forward and backward must apply prior_densities at the same time step
        priors = np.random.default_rng(1).uniform(0.1, 1.0, (self.T, self.K))
        expected = brute_force_smoothed(self.params, self.u0, self.ys, priors=priors)
        s = self.smoothed(self.params, self.ys, prior_densities=jnp.asarray(priors))
        np.testing.assert_allclose(s, expected, rtol=1e-8)

    def test_scalar_prior_densities_do_not_change_result(self):
        # a per-time constant is the same for every path, so it cancels in the posterior
        priors = jnp.asarray(np.random.default_rng(2).uniform(0.1, 1.0, (self.T, 1)))
        np.testing.assert_allclose(self.smoothed(self.params, self.ys, prior_densities=priors),
                                   self.smoothed(self.params, self.ys), rtol=1e-10)

    def test_uninformative_emissions_give_prior_state_marginals(self):
        # identical emissions => smoothed = predicted marginals u0 @ Gamma^{t+1}
        params = Params(transition=self.params.transition,
                        emission=GaussEmission.from_params(jnp.zeros(self.K), jnp.ones(self.K)))
        expected = np.stack([np.asarray(self.u0 @ jnp.linalg.matrix_power(self.P, t + 1))[0]
                             for t in range(self.T)])
        np.testing.assert_allclose(self.smoothed(params, self.ys), expected, rtol=1e-8)

    # --- consistency with ForwardAlgorithm --------------------------------

    def test_last_smoothed_equals_last_filtered(self):
        utts = np.asarray(ForwardAlgorithm.run(self.params, self.u0, self.ys)[0])[:, 0, :]
        np.testing.assert_allclose(self.smoothed(self.params, self.ys)[-1], utts[-1], rtol=1e-10)

    def test_single_observation_equals_filtered(self):
        utts = np.asarray(ForwardAlgorithm.run(self.params, self.u0, self.ys[:1])[0])[:, 0, :]
        np.testing.assert_allclose(self.smoothed(self.params, self.ys[:1]), utts, rtol=1e-10)

    def test_smoothing_differs_from_filtering_before_last_step(self):
        utts = np.asarray(ForwardAlgorithm.run(self.params, self.u0, self.ys)[0])[:, 0, :]
        s = self.smoothed(self.params, self.ys)
        self.assertFalse(np.allclose(s[:-1], utts[:-1]))

    # --- jax ----------------------------------------------------------------

    def test_jit_matches_eager(self):
        eager = ForwardBackwardAlgorithm.run(self.params, self.u0, self.ys)
        jitted = jax.jit(ForwardBackwardAlgorithm.run)(self.params, self.u0, self.ys)
        np.testing.assert_allclose(np.asarray(jitted), np.asarray(eager))
