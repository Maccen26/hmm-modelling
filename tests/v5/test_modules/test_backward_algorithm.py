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
from src.api.v5.inference.backward import BackwardAlgorithm


def reference_backward(Gammas: np.ndarray, gs: np.ndarray, priors: np.ndarray | None = None) -> np.ndarray:
    """
    beta_{T-1} = 1,  beta_t = Gamma_{t+1} @ (g_{t+1} * prior_{t+1} * beta_{t+1}).

    Same convention as ForwardAlgorithm: Gamma_t moves the chain from t-1 to t.
    """
    T, K = gs.shape
    priors = np.ones(T) if priors is None else priors
    betas = np.ones((T, K))
    for t in range(T - 2, -1, -1):
        betas[t] = Gammas[t + 1] @ (gs[t + 1] * priors[t + 1] * betas[t + 1])
    return betas


def normalise(a: np.ndarray) -> np.ndarray:
    return a / a.sum(axis=1, keepdims=True)


def brute_force_backward(Gammas: np.ndarray, gs: np.ndarray) -> np.ndarray:
    """beta_t(i) = p(y_{t+1:T-1} | z_t = i), by summing over every future state path."""
    T, K = gs.shape
    betas = np.ones((T, K))
    for t in range(T - 1):
        for i in range(K):
            total = 0.0
            for future in itertools.product(range(K), repeat=T - 1 - t):
                prev, prob = i, 1.0
                for s, z in zip(range(t + 1, T), future):
                    prob *= Gammas[s, prev, z] * gs[s, z]
                    prev = z
                total += prob
            betas[t, i] = total
    return betas


class TestBackwardAlgorithm(TestCase):
    """
    Unit tests for BackwardAlgorithm. The returned b_t is normalised to sum to one,
    so b_t = beta_t / sum(beta_t) with beta_t(i) = p(y_{t+1:T-1} | z_t = i).
    """

    def setUp(self) -> None:
        self.K, self.T, self.C = 3, 6, 2
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

    def run_backward(self, params: Params, ys: jnp.ndarray, xs=None, prior_densities=None) -> np.ndarray:
        betas = BackwardAlgorithm.run(params, ys, xs=xs, prior_densities=prior_densities)
        return np.asarray(betas).reshape(len(ys), -1)

    def inputs(self, params: Params, ys: jnp.ndarray, xs=None) -> tuple[np.ndarray, np.ndarray]:
        Gammas = np.asarray(params.transition_matrices(len(ys), xs))
        gs = np.asarray(params.densities(ys))[:, 0, :]
        return Gammas, gs

    # --- output structure -------------------------------------------------

    def test_output_shape(self):
        betas = BackwardAlgorithm.run(self.params, self.ys)
        self.assertEqual(betas.shape[0], self.T)
        self.assertEqual(int(np.prod(betas.shape[1:])), self.K)

    def test_each_beta_sums_to_one(self):
        betas = self.run_backward(self.params, self.ys)
        np.testing.assert_allclose(betas.sum(axis=1), np.ones(self.T), rtol=1e-12)

    def test_last_beta_is_uniform(self):
        # beta_{T-1} = 1, normalised => 1/K
        betas = self.run_backward(self.params, self.ys)
        np.testing.assert_allclose(betas[-1], np.full(self.K, 1 / self.K))

    def test_betas_are_positive_and_finite(self):
        betas = self.run_backward(self.params, self.ys)
        self.assertTrue(np.all(np.isfinite(betas)))
        self.assertTrue(np.all(betas > 0))

    def test_single_observation(self):
        betas = self.run_backward(self.params, self.ys[:1])
        np.testing.assert_allclose(betas, np.full((1, self.K), 1 / self.K))

    def test_long_sequence_does_not_underflow(self):
        # unscaled beta_0 underflows to 0 long before T = 5000
        ys = jnp.asarray(np.random.default_rng(3).normal(0.5, 1.5, 5000))
        betas = self.run_backward(self.params, ys)
        self.assertTrue(np.all(np.isfinite(betas)))
        self.assertTrue(np.all(betas > 0))
        np.testing.assert_allclose(betas.sum(axis=1), np.ones(len(ys)), rtol=1e-12)

    # --- correctness (normalised b_t is beta_t up to a per-t constant) ----

    def test_matches_reference_recursion(self):
        betas = self.run_backward(self.params, self.ys)
        expected = normalise(reference_backward(*self.inputs(self.params, self.ys)))
        np.testing.assert_allclose(betas, expected, rtol=1e-10)

    def test_matches_brute_force_enumeration(self):
        ys = self.ys[:4]
        betas = self.run_backward(self.params, ys)
        expected = normalise(brute_force_backward(*self.inputs(self.params, ys)))
        np.testing.assert_allclose(betas, expected, rtol=1e-10)

    def test_dynamic_transition_uses_gamma_of_next_time_step(self):
        betas = self.run_backward(self.dynamic_params, self.ys, xs=self.xs)
        expected = normalise(reference_backward(*self.inputs(self.dynamic_params, self.ys, self.xs)))
        np.testing.assert_allclose(betas, expected, rtol=1e-10)

    def test_prior_densities_are_applied_to_next_time_step(self):
        rng = np.random.default_rng(1)
        priors = rng.uniform(0.2, 1.0, self.T)
        betas = self.run_backward(self.params, self.ys, prior_densities=jnp.asarray(priors)[:, None])
        expected = normalise(reference_backward(*self.inputs(self.params, self.ys), priors=priors))
        np.testing.assert_allclose(betas, expected, rtol=1e-10)

    def test_default_prior_densities_equal_ones(self):
        a = self.run_backward(self.params, self.ys)
        b = self.run_backward(self.params, self.ys, prior_densities=jnp.ones((self.T, 1)))
        np.testing.assert_allclose(a, b)

    # --- consistency with ForwardAlgorithm --------------------------------

    def test_smoothed_marginals_match_brute_force_posterior(self):
        ys = self.ys[:4]
        T = len(ys)
        Gammas, gs = self.inputs(self.params, ys)
        delta = np.asarray(self.u0).ravel()
        expected = np.zeros((T, self.K))
        for path in itertools.product(range(self.K), repeat=T):
            prob = (delta @ Gammas[0])[path[0]] * gs[0, path[0]]
            for t in range(1, T):
                prob *= Gammas[t, path[t - 1], path[t]] * gs[t, path[t]]
            expected[np.arange(T), path] += prob

        utts = np.asarray(ForwardAlgorithm.run(self.params, self.u0, ys)[0])[:, 0, :]
        smoothed = normalise(utts * self.run_backward(self.params, ys))
        np.testing.assert_allclose(smoothed, normalise(expected), rtol=1e-8)

    def test_last_smoothed_equals_last_filtered(self):
        utts = np.asarray(ForwardAlgorithm.run(self.params, self.u0, self.ys)[0])[:, 0, :]
        smoothed = normalise(utts * self.run_backward(self.params, self.ys))
        np.testing.assert_allclose(smoothed[-1], utts[-1], rtol=1e-10)

    # --- jax ----------------------------------------------------------------

    def test_jit_matches_eager(self):
        eager = BackwardAlgorithm.run(self.params, self.ys)
        jitted = jax.jit(BackwardAlgorithm.run)(self.params, self.ys)
        np.testing.assert_allclose(np.asarray(jitted), np.asarray(eager))
