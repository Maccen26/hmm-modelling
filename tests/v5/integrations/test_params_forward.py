import jax
jax.config.update("jax_enable_x64", True)

import itertools
from unittest import TestCase
import numpy as np
import jax.numpy as jnp
import optax

from src.api.v5.hmm.params import Params
from src.api.v5.hmm.transitions import StaticTransition, DynamicTransition
from src.api.v5.hmm.emissions import GaussEmission, AutoregressiveGaussEmission
from src.api.v5.inference.forward_algorithm import ForwardAlgorithm


def brute_force_log_likelihood(params: Params, u0: jnp.ndarray, ys: jnp.ndarray, xs: jnp.ndarray | None = None) -> float:
    """
    log p(y_0, ..., y_{T-1}) by summing over every state path.

    Mirrors the forward recursion's convention: the initial distribution sits one
    step before the first observation, so Gamma_0 is applied before y_0.
    """
    T = len(ys)
    Gammas = np.asarray(params.transition_matrices(T, xs))
    gs = np.asarray(params.densities(ys))[:, 0, :]  # (T, K)
    delta = np.asarray(u0).ravel()
    K = len(delta)

    total = 0.0
    for path in itertools.product(range(K), repeat=T):
        prob = (delta @ Gammas[0])[path[0]] * gs[0, path[0]]
        for t in range(1, T):
            prob *= Gammas[t, path[t - 1], path[t]] * gs[t, path[t]]
        total += prob
    return float(np.log(total))


def forward_log_likelihood(params: Params, u0: jnp.ndarray, ys: jnp.ndarray, xs: jnp.ndarray | None = None) -> jnp.ndarray:
    _, fts, _ = ForwardAlgorithm().run(params, u0, ys, xs)
    return jnp.sum(jnp.log(fts))


def simulate(P: np.ndarray, mean: np.ndarray, sigma: np.ndarray, T: int, seed: int = 0) -> np.ndarray:
    """Sample T observations from a Gaussian HMM with transition matrix P."""
    rng = np.random.default_rng(seed)
    K = len(mean)
    z = rng.integers(K)
    ys = np.empty(T)
    for t in range(T):
        z = rng.choice(K, p=P[z])
        ys[t] = rng.normal(mean[z], sigma[z])
    return ys


class TestParamsForwardIntegration(TestCase):
    """
    Runs Params through the v5 ForwardAlgorithm and checks the resulting
    likelihood against exhaustive enumeration of the hidden state paths.
    """

    def setUp(self) -> None:
        self.K, self.T, self.C = 3, 6, 2
        self.P = jnp.array([[0.8, 0.1, 0.1],
                            [0.1, 0.8, 0.1],
                            [0.1, 0.1, 0.8]])
        self.mean = jnp.array([0.0, 1.0, 2.0])
        self.sigma = jnp.array([1.0, 0.5, 2.0])
        self.u0 = jnp.array([[0.5, 0.3, 0.2]])

        rng = np.random.default_rng(1)
        self.ys = jnp.asarray(rng.normal(1.0, 1.0, self.T))
        self.xs = jnp.asarray(rng.integers(0, 3, (self.T, self.C)), dtype=float)
        self.beta = jnp.asarray(rng.normal(0, 0.3, (self.C, self.K, self.K - 1)))

        self.params = Params(transition=StaticTransition.from_params(self.P),
                             emission=GaussEmission.from_params(self.mean, self.sigma))

    # --- output structure -------------------------------------------------

    def test_output_shapes(self):
        utts, fts, uts = ForwardAlgorithm().run(self.params, self.u0, self.ys)
        self.assertEqual(utts.shape, (self.T, 1, self.K))
        self.assertEqual(fts.shape, (self.T,))
        self.assertEqual(uts.shape, (self.T, 1, self.K))

    def test_filtered_and_predictive_are_distributions(self):
        utts, fts, uts = ForwardAlgorithm().run(self.params, self.u0, self.ys)
        self.assertTrue(jnp.allclose(utts.sum(axis=-1), 1.0))
        self.assertTrue(jnp.allclose(uts.sum(axis=-1), 1.0))
        self.assertTrue(bool(jnp.all(fts > 0.0)))

    def test_one_step_predictive_propagates_filtered(self):
        # u_{t} = u_{t-1|t-1} @ Gamma for a static transition
        utts, _, uts = ForwardAlgorithm().run(self.params, self.u0, self.ys)
        self.assertTrue(jnp.allclose(uts[0], self.u0 @ self.P))
        self.assertTrue(jnp.allclose(uts[1:], utts[:-1] @ self.P))

    # --- likelihood correctness -------------------------------------------

    def test_static_gauss_matches_brute_force(self):
        expected = brute_force_log_likelihood(self.params, self.u0, self.ys)
        self.assertAlmostEqual(float(forward_log_likelihood(self.params, self.u0, self.ys)), expected, places=8)

    def test_dynamic_transition_matches_brute_force(self):
        params = Params(transition=DynamicTransition.from_params(self.P, self.beta),
                        emission=self.params.emission)
        expected = brute_force_log_likelihood(params, self.u0, self.ys, self.xs)
        self.assertAlmostEqual(float(forward_log_likelihood(params, self.u0, self.ys, self.xs)), expected, places=8)

    def test_autoregressive_emission_matches_brute_force(self):
        emission = AutoregressiveGaussEmission.from_params(self.mean, self.sigma, jnp.full((1, self.K), 0.4))
        params = Params(transition=self.params.transition, emission=emission)
        expected = brute_force_log_likelihood(params, self.u0, self.ys)
        self.assertAlmostEqual(float(forward_log_likelihood(params, self.u0, self.ys)), expected, places=8)

    def test_single_state_reduces_to_iid_gaussian(self):
        params = Params(transition=StaticTransition.from_params(jnp.array([[1.0]])),
                        emission=GaussEmission.from_params(jnp.array([0.5]), jnp.array([1.5])))
        expected = jnp.sum(jax.scipy.stats.norm.logpdf(self.ys, loc=0.5, scale=1.5))
        actual = forward_log_likelihood(params, jnp.array([[1.0]]), self.ys)
        self.assertAlmostEqual(float(actual), float(expected), places=10)

    # --- end-to-end through jit / grad ------------------------------------

    def test_jit_matches_eager(self):
        jitted = jax.jit(lambda p: forward_log_likelihood(p, self.u0, self.ys))(self.params)
        self.assertAlmostEqual(float(jitted), float(forward_log_likelihood(self.params, self.u0, self.ys)), places=10)

    def test_gradient_is_finite_for_every_leaf(self):
        grads = jax.grad(lambda p: forward_log_likelihood(p, self.u0, self.ys))(self.params)
        for leaf in jax.tree.leaves(grads):
            self.assertTrue(bool(jnp.all(jnp.isfinite(leaf))))
        self.assertTrue(bool(jnp.any(grads.transition.transition_logits != 0.0)))
        self.assertTrue(bool(jnp.any(grads.emission.mu0 != 0.0)))


class TestParamsFitIntegration(TestCase):
    """
    Fits a Params pytree to simulated data with optax on the forward-algorithm
    negative log-likelihood, and checks it moves towards the generating model.
    """

    def setUp(self) -> None:
        self.P_true = np.array([[0.9, 0.1],
                                [0.2, 0.8]])
        self.mean_true = np.array([0.0, 3.0])
        self.sigma_true = np.array([0.5, 0.8])
        self.ys = jnp.asarray(simulate(self.P_true, self.mean_true, self.sigma_true, T=400, seed=2))
        self.u0 = jnp.array([[0.5, 0.5]])

        self.init = Params(transition=StaticTransition.from_params(jnp.array([[0.6, 0.4], [0.4, 0.6]])),
                           emission=GaussEmission.from_params(jnp.array([-1.0, 1.0]), jnp.array([1.0, 1.0])))

    def _fit(self, params: Params, n_iter: int = 300) -> Params:
        nll = lambda p: -forward_log_likelihood(p, self.u0, self.ys)
        optimizer = optax.adam(5e-2)
        opt_state = optimizer.init(params)

        @jax.jit
        def step(params, opt_state):
            grads = jax.grad(nll)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state

        for _ in range(n_iter):
            params, opt_state = step(params, opt_state)
        return params

    def test_fit_increases_log_likelihood(self):
        fitted = self._fit(self.init)
        ll_init = forward_log_likelihood(self.init, self.u0, self.ys)
        ll_fit = forward_log_likelihood(fitted, self.u0, self.ys)
        self.assertGreater(float(ll_fit), float(ll_init))

    def test_fit_recovers_generating_parameters(self):
        fitted = self._fit(self.init)
        mu = fitted.emission.mu(0, self.ys)
        sigma = fitted.emission.sigma(0, self.ys)
        self.assertTrue(np.allclose(mu, self.mean_true, atol=0.2), f"mu={mu}")
        self.assertTrue(np.allclose(sigma, self.sigma_true, atol=0.2), f"sigma={sigma}")
        self.assertTrue(np.allclose(fitted.transition_matrix(), self.P_true, atol=0.1),
                        f"P={fitted.transition_matrix()}")

    def test_fitted_params_stay_a_params_pytree(self):
        fitted = self._fit(self.init, n_iter=5)
        self.assertIsInstance(fitted, Params)
        self.assertEqual(len(fitted), len(self.init))
        self.assertEqual([name for name, _ in fitted], ["transition", "emission"])
