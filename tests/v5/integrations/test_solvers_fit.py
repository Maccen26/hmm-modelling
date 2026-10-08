import jax
jax.config.update("jax_enable_x64", True)

from unittest import TestCase
import numpy as np
import jax.numpy as jnp
import optax

from src.api.v5.hmm.params import Params
from src.api.v5.hmm.transitions import StaticTransition, DynamicTransition
from src.api.v5.hmm.emissions import GaussEmission, AutoregressiveGaussEmission
from src.api.v5.inference.forward_algorithm import ForwardAlgorithm
from src.api.v5.inference.likelihoods import negative_log_likelihood
from src.api.v5.solvers import GradientSolver, LBFGSSolver, SolverResult
from tests.v5.integrations.test_params_forward import simulate


def hmm_nll(params: Params, data) -> jnp.ndarray:
    """Solver loss_fn: forward-algorithm NLL with data = (u0, ys, xs)."""
    u0, ys, xs = data
    return negative_log_likelihood(ForwardAlgorithm().run(params, u0, ys, xs), params)


class SimulatedHMM:
    """Two-state Gaussian HMM data and a deliberately poor starting point."""

    def setUp(self) -> None:
        self.P_true = np.array([[0.9, 0.1],
                                [0.2, 0.8]])
        self.mean_true = np.array([0.0, 3.0])
        self.sigma_true = np.array([0.5, 0.8])
        self.ys = jnp.asarray(simulate(self.P_true, self.mean_true, self.sigma_true, T=400, seed=2))
        self.u0 = jnp.array([[0.5, 0.5]])
        self.data = (self.u0, self.ys, None)

        self.init = Params(transition=StaticTransition.from_params(jnp.array([[0.6, 0.4], [0.4, 0.6]])),
                           emission=GaussEmission.from_params(jnp.array([-1.0, 1.0]), jnp.array([1.0, 1.0])))


class SolverFitContract(SimulatedHMM):
    """
    Fits a Params pytree to simulated HMM data through solver.fit, the
    ForwardAlgorithm and negative_log_likelihood. Run against every solver.
    """

    def make_solver(self):
        raise NotImplementedError

    def fit(self, params: Params | None = None, data=None, frozen=None) -> SolverResult:
        return self.make_solver().fit(params or self.init, hmm_nll, data or self.data, frozen=frozen)

    # --- result structure -------------------------------------------------

    def test_result_is_params_pytree(self):
        result = self.fit()
        self.assertIsInstance(result, SolverResult)
        self.assertIsInstance(result.params, Params)
        self.assertEqual(jax.tree.structure(result.params), jax.tree.structure(self.init))
        for leaf in jax.tree.leaves(result.params):
            self.assertTrue(bool(jnp.all(jnp.isfinite(leaf))))

    def test_reported_loss_matches_fitted_params(self):
        result = self.fit()
        self.assertAlmostEqual(result.loss, float(hmm_nll(result.params, self.data)), places=4)

    # --- optimisation -----------------------------------------------------

    def test_fit_decreases_nll(self):
        result = self.fit()
        self.assertLess(result.loss, float(hmm_nll(self.init, self.data)))

    def test_fit_recovers_generating_parameters(self):
        fitted = self.fit().params
        mu = fitted.emission.mu(0, self.ys)
        sigma = fitted.emission.sigma(0, self.ys)
        self.assertTrue(np.allclose(mu, self.mean_true, atol=0.2), f"mu={mu}")
        self.assertTrue(np.allclose(sigma, self.sigma_true, atol=0.2), f"sigma={sigma}")
        self.assertTrue(np.allclose(fitted.transition_matrix(), self.P_true, atol=0.1),
                        f"P={fitted.transition_matrix()}")

    # --- freezing ---------------------------------------------------------

    def test_whole_leaf_freeze_keeps_leaf_and_trains_rest(self):
        result = self.fit(frozen=[("log_sigma", None)])
        self.assertTrue(jnp.array_equal(result.params.emission.log_sigma, self.init.emission.log_sigma))
        self.assertFalse(jnp.allclose(result.params.emission.mu0, self.init.emission.mu0))
        self.assertFalse(jnp.allclose(result.params.transition.transition_logits,
                                      self.init.transition.transition_logits))

    def test_frozen_fit_is_no_better_than_free_fit(self):
        free = self.fit()
        frozen = self.fit(frozen=[("log_sigma", None)])
        self.assertLessEqual(free.loss, frozen.loss + 1e-6)

    def test_element_freeze_pins_single_transition_logit(self):
        result = self.fit(frozen=[("transition_logits", (0, 0))])
        logits, init_logits = result.params.transition.transition_logits, self.init.transition.transition_logits
        self.assertEqual(float(logits[0, 0]), float(init_logits[0, 0]))
        self.assertNotAlmostEqual(float(logits[1, 0]), float(init_logits[1, 0]), places=3)

    # --- other model variants ---------------------------------------------

    def test_dynamic_transition_with_covariates(self):
        rng = np.random.default_rng(3)
        xs = jnp.asarray(rng.normal(0.0, 1.0, (len(self.ys), 2)))
        params = Params(transition=DynamicTransition.from_params(jnp.array([[0.6, 0.4], [0.4, 0.6]]),
                                                                 jnp.zeros((2, 2, 1))),
                        emission=self.init.emission)
        data = (self.u0, self.ys, xs)
        result = self.fit(params, data)
        self.assertLess(result.loss, float(hmm_nll(params, data)))
        self.assertTrue(bool(jnp.all(jnp.isfinite(result.params.transition.beta))))
        self.assertFalse(jnp.allclose(result.params.transition.beta, 0.0))

    def test_autoregressive_emission_with_element_frozen_phi(self):
        emission = AutoregressiveGaussEmission.from_params(jnp.array([-1.0, 1.0]), jnp.array([1.0, 1.0]),
                                                           jnp.full((1, 2), 0.2))
        params = Params(transition=self.init.transition, emission=emission)
        result = self.fit(params, frozen=[("phi_tilde", (0, 1))])
        phi_tilde = result.params.emission.phi_tilde
        self.assertLess(result.loss, float(hmm_nll(params, self.data)))
        self.assertEqual(float(phi_tilde[0, 1]), float(emission.phi_tilde[0, 1]))
        self.assertNotAlmostEqual(float(phi_tilde[0, 0]), float(emission.phi_tilde[0, 0]), places=3)


class TestLBFGSSolverFit(SolverFitContract, TestCase):
    def make_solver(self):
        return LBFGSSolver(n_iter=200, tol=1e-5)

    def test_converges(self):
        self.assertTrue(self.fit().converged)

    def test_freezing_everything_raises(self):
        # Intended: fitting with no trainable parameters is an error, not a no-op.
        frozen = [("transition_logits", None), ("log_mu_diff", None), ("mu0", None), ("log_sigma", None)]
        with self.assertRaises(IndexError):
            self.fit(frozen=frozen)


class TestGradientSolverFit(SolverFitContract, TestCase):
    def make_solver(self):
        return GradientSolver(optimizer=optax.adam(5e-2), n_iter=400, tol=1e-4)


class TestSolversAgree(SimulatedHMM, TestCase):
    """Both solvers should land on the same optimum of the same HMM likelihood."""

    def test_lbfgs_and_gradient_reach_same_loss(self):
        lbfgs = LBFGSSolver(n_iter=500, tol=1e-6).fit(self.init, hmm_nll, self.data)
        adam = GradientSolver(optimizer=optax.adam(5e-2), n_iter=2000, tol=1e-6).fit(self.init, hmm_nll, self.data)
        self.assertAlmostEqual(lbfgs.loss, adam.loss, delta=1e-2)
        self.assertLessEqual(lbfgs.loss, adam.loss + 1e-6)
