import jax
jax.config.update("jax_enable_x64", True)

from unittest import TestCase
import numpy as np
import jax.numpy as jnp

from src.api.v5.hmm.params import Params
from src.api.v5.hmm.transitions import StaticTransition, DynamicTransition
from src.api.v5.hmm.emissions import GaussEmission, AutoregressiveGaussEmission, MultivariateGaussEmission
from src.api.v5.inference.forward_algorithm import ForwardAlgorithm
from src.api.v5.analysis import Diagnostics, Predictor


def simulate(P: np.ndarray, mean: np.ndarray, sigma: np.ndarray, T: int, seed: int = 0) -> np.ndarray:
    """Sample T observations from a Gaussian HMM with transition matrix P."""
    rng = np.random.default_rng(seed)
    z = rng.integers(len(mean))
    ys = np.empty(T)
    for t in range(T):
        z = rng.choice(len(mean), p=P[z])
        ys[t] = rng.normal(mean[z], sigma[z])
    return ys


class TestAnalysisBase(TestCase):
    def setUp(self) -> None:
        self.P = np.array([[0.9, 0.1], [0.2, 0.8]])
        self.mean = np.array([0.0, 3.0])
        self.sigma = np.array([1.0, 0.5])
        self.ys = jnp.asarray(simulate(self.P, self.mean, self.sigma, T=400))
        self.u0 = jnp.ones((1, 2)) / 2

        self.gauss = Params(transition=StaticTransition.from_params(jnp.asarray(self.P)),
                            emission=GaussEmission.from_params(self.mean, self.sigma))
        self.ar = Params(transition=StaticTransition.from_params(jnp.asarray(self.P)),
                         emission=AutoregressiveGaussEmission.from_params(self.mean, self.sigma, jnp.array([[0.3, 0.1]])))

    def forward_ll(self, params, ys, xs=None):
        _, fts, _ = ForwardAlgorithm.run(params=params, initial_dist=self.u0, ys=ys, xs=xs)
        return float(jnp.sum(jnp.log(fts)))


class TestDiagnostics(TestAnalysisBase):

    def setUp(self) -> None:
        super().setUp()
        self.diag = Diagnostics(self.ys).add_model("gauss", self.gauss).add_model("ar", self.ar)

    def test_aic_bic_match_formula(self):
        ll, k, T = self.forward_ll(self.gauss, self.ys), len(self.gauss), len(self.ys)
        self.assertAlmostEqual(self.diag.aic("gauss"), 2 * k - 2 * ll)
        self.assertAlmostEqual(self.diag.bic("gauss"), k * np.log(T) - 2 * ll)
        summary = self.diag.summary()
        self.assertEqual(list(summary.index), ["gauss", "ar"])
        self.assertAlmostEqual(summary.loc["gauss", "LL"], ll)

    def test_n_params_override(self):
        self.diag.add_model("frozen", self.gauss, n_params=3)
        self.assertAlmostEqual(self.diag.aic("frozen"), 6 - 2 * self.forward_ll(self.gauss, self.ys))

    def test_lrt(self):
        table = self.diag.add_lrt("gauss", "ar").lrt()
        row = table.iloc[0]
        self.assertEqual(row["df"], len(self.ar) - len(self.gauss))
        self.assertAlmostEqual(row["LR"], 2 * (row["LL_expanded"] - row["LL_base"]))
        self.assertTrue(0.0 <= row["p_value"] <= 1.0)

    def test_lrt_rejects_non_nested_pair(self):
        with self.assertRaises(ValueError):
            self.diag.add_lrt("ar", "gauss")

    def test_lrt_rejects_different_data(self):
        self.diag.add_model("other", self.ar, ys=self.ys[::-1])
        with self.assertRaises(ValueError):
            self.diag.add_lrt("gauss", "other")

    def test_residuals_univariate(self):
        res = self.diag.residuals("gauss")
        self.assertEqual(res.shape, (len(self.ys),))
        # Data simulated from the model: residuals are approximately N(0, 1)
        self.assertLess(abs(float(jnp.mean(res))), 0.2)
        self.assertLess(abs(float(jnp.std(res)) - 1.0), 0.2)

    def test_residuals_ar_first_lags_nan(self):
        res = self.diag.residuals("ar")
        self.assertTrue(bool(jnp.isnan(res[0])))
        self.assertFalse(bool(jnp.any(jnp.isnan(res[1:]))))

    def test_residuals_multivariate_per_dimension(self):
        ys2 = jnp.stack([self.ys, 2 * self.ys], axis=1)
        mv = Params(transition=StaticTransition.from_params(jnp.asarray(self.P)),
                    emission=MultivariateGaussEmission.from_params(
                        jnp.stack([self.mean, 2 * self.mean]), jnp.stack([self.sigma, 2 * self.sigma])))
        self.diag.add_model("mv", mv, ys=ys2)
        self.assertEqual(self.diag.residuals("mv").shape, (len(self.ys), 2))
        self.assertEqual(set(self.diag.residuals_all()), {"gauss", "ar", "mv"})


class TestPredictor(TestAnalysisBase):

    def test_predict_state_probs_follow_gamma_powers(self):
        H = 5
        pred = Predictor.predict(self.gauss, self.ys, horizon=H)
        utts, _, _ = ForwardAlgorithm.run(params=self.gauss, initial_dist=self.u0, ys=self.ys)
        Gamma = self.gauss.transition.transition_matrix()
        for h in range(1, H + 1):
            expected = jnp.ravel(utts[-1]) @ jnp.linalg.matrix_power(Gamma, h)
            np.testing.assert_allclose(pred.state_probs[h - 1], expected, atol=1e-12)
        self.assertEqual(pred.y_pred.shape, (H,))

    def test_predict_converges_to_stationary_mean(self):
        pred = Predictor.predict(self.gauss, self.ys, horizon=200)
        delta = np.array([2 / 3, 1 / 3])  # stationary distribution of P
        self.assertAlmostEqual(float(pred.y_pred[-1]), float(delta @ self.mean), places=6)

    def test_rolling_one_step_matches_forward(self):
        ys_train, ys_test = self.ys[:300], self.ys[300:]
        for params in (self.gauss, self.ar):
            pred = Predictor.rolling(params, ys_train, ys_test, horizon=1)
            _, _, uts = ForwardAlgorithm.run(params=params, initial_dist=self.u0, ys=self.ys)
            np.testing.assert_allclose(pred.state_probs, uts[300:, 0, :], atol=1e-12)
            mus = jax.vmap(lambda t: params.emission.mu(t, self.ys))(jnp.arange(300, 400))
            np.testing.assert_allclose(pred.y_pred, jnp.sum(mus * uts[300:, 0, :], axis=1), atol=1e-10)
            self.assertTrue(np.isfinite(pred.rmse(ys_test)))

    def test_rolling_multi_step_shape(self):
        pred = Predictor.rolling(self.ar, self.ys[:300], self.ys[300:], horizon=3)
        self.assertEqual(pred.y_pred.shape, (100,))
        self.assertEqual(pred.state_probs.shape, (100, 2))

    def test_dynamic_transition_requires_covariates(self):
        beta = jnp.zeros((1, 2, 1))
        dyn = Params(transition=DynamicTransition.from_params(jnp.asarray(self.P), beta),
                     emission=GaussEmission.from_params(self.mean, self.sigma))
        xs = jnp.ones((len(self.ys), 1))
        with self.assertRaises(ValueError):
            Predictor.predict(dyn, self.ys, horizon=3, xs=xs)
        # With zero betas the dynamic model must forecast exactly like the static one
        pred = Predictor.rolling(dyn, self.ys[:300], self.ys[300:], xs=xs[:300], xs_test=xs[300:], horizon=2)
        static = Predictor.rolling(self.gauss, self.ys[:300], self.ys[300:], horizon=2)
        np.testing.assert_allclose(pred.y_pred, static.y_pred, atol=1e-12)
