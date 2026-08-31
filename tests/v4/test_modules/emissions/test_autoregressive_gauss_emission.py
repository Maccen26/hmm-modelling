from src.api.v4 import AutoregressiveGaussEmission, StaticTransitionHigherOrder
from src.api.v4.emissions.autoregressive_gauss_emission import phi_to_phi_tilde, phi_tilde_to_phi
import jax.numpy as jnp
from unittest import TestCase
import jax

class TestAutoregressiveGaussEmission(TestCase):
    def setUp(self) -> None:
        self.mean = jnp.array([0.0, 1.0, 2.0]) 
        self.sigma = jnp.array([1.0, 1.0, 1.0])
        self.phi = jnp.array([0.5, 0.5, 0.5])  # Example phi values

        self.mu0 = jnp.array(self.mean[0])
        self.log_mu_diff = jnp.log(jnp.array([self.mean[i] - self.mean[i-1] for i in range(1, len(self.mean))]))
        self.log_sigma = jnp.log(self.sigma)
        self.phi_tilde = phi_to_phi_tilde(self.phi)  # Transform phi to phi_tilde

        self.yt = jnp.array([0.0, 1.0, 2.0])  # 1-dimensional observation for testing 

    def test_build_from_params(self):
        emission = AutoregressiveGaussEmission.from_params(self.mean, self.sigma, self.phi)

        self.assertTrue(jnp.allclose(emission.mu0, self.mu0))
        self.assertTrue(jnp.allclose(emission.log_mu_diff, self.log_mu_diff))
        self.assertTrue(jnp.allclose(emission.log_sigma, self.log_sigma))
        self.assertTrue(jnp.allclose(jnp.stack(emission.phi_tilde), jnp.atleast_2d(self.phi_tilde)))

    def test_build(self):
        emission = AutoregressiveGaussEmission(log_mu_diff=self.log_mu_diff, mu0=self.mu0, log_sigma=self.log_sigma, phi_tilde=self.phi_tilde)

        self.assertTrue(jnp.allclose(emission.mu0, self.mu0))
        self.assertTrue(jnp.allclose(emission.log_mu_diff, self.log_mu_diff))
        self.assertTrue(jnp.allclose(emission.log_sigma, self.log_sigma))
        self.assertTrue(jnp.allclose(jnp.stack(emission.phi_tilde), jnp.atleast_2d(self.phi_tilde)))

    def test_density_success(self):
        emission = AutoregressiveGaussEmission.from_params(self.mean, self.sigma, self.phi)
        try: 
            density = emission.density(0, self.yt)
            self.assertTrue(density.shape == (1, len(self.mean))) 
        except Exception as e:
            self.fail(f"Density computation failed with error: {e}") 

    
    def test_cdf_success(self):
        emission = AutoregressiveGaussEmission.from_params(self.mean, self.sigma, self.phi)
        try:
            cdf = emission.cdf(0, self.yt)
            self.assertTrue(cdf.shape == (1, len(self.mean))) 
        except Exception as e:
            self.fail(f"CDF computation failed with error: {e}")

    def test_cdf_correctness(self):
        emission = AutoregressiveGaussEmission.from_params(self.mean, self.sigma, self.phi)
        yt = jnp.array([0.0])
        cdf = emission.cdf(0, self.yt)
        expected_cdf = jax.scipy.stats.norm.cdf(yt, loc=self.mean, scale=self.sigma)
        self.assertTrue(jnp.allclose(cdf, expected_cdf))

    def test_phi_transformation(self):
        original_phi = jnp.array([0.5, 0.5, 0.5])
        phi_tilde = phi_to_phi_tilde(original_phi)
        recovered_phi = phi_tilde_to_phi(phi_tilde)
        self.assertTrue(jnp.allclose(original_phi, recovered_phi), "Phi transformation and inverse should recover original phi values")

    def test_phi_correctness(self):
        emission = AutoregressiveGaussEmission.from_params(self.mean, self.sigma, self.phi)
        recovered_phi = emission.phi()
        #assert phi is between 0 and 1
        self.assertTrue(jnp.all((recovered_phi >= 0) & (recovered_phi <= 1)), "Recovered phi values should be between 0 and 1")

        self.assertTrue(jnp.allclose(recovered_phi, self.phi), "Recovered phi should match original phi values") 

        
    def test_extreme_phi_tilde_values(self):
        extreme_phi_tilde = jnp.array([-1000.0, 0.0, 1000.0])  # Very large negative, zero, and very large positive values
        emission = AutoregressiveGaussEmission(log_mu_diff=self.log_mu_diff, mu0=self.mu0, log_sigma=self.log_sigma, phi_tilde=extreme_phi_tilde)
        recovered_phi = emission.phi()
        self.assertTrue(jnp.all((recovered_phi >= -1) & (recovered_phi <= 1)), "Recovered phi values should be between 0 and 1 even for extreme phi_tilde values")

    def test_density_is_one_for_t_less_than_k(self):
        """For t < k (number of lags), density should be 1.0 (log-likelihood contribution = 0)."""
        phi = jnp.array([[0.5, 0.5, 0.5],
                  [0.3, 0.3, 0.3]])  # (k=2, num_states=3)
        emission = AutoregressiveGaussEmission.from_params(self.mean, self.sigma, phi)
        ys = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        for t in range(len(phi)):  # t=0 and t=1 should have density = 1.0
            density = emission.density(t, ys)
            self.assertTrue(jnp.allclose(density, jnp.ones_like(density)),
                            f"Density at t={t} should be 1.0 when t < k={len(phi)}")

    def test_density_is_not_one_for_t_geq_k(self):
        """For t >= k, density should be actual Gaussian pdf values, not 1.0."""
        phi = jnp.array([[0.5, 0.5, 0.5],
                  [0.3, 0.3, 0.3]])  # (k=2, num_states=3)
        emission = AutoregressiveGaussEmission.from_params(self.mean, self.sigma, phi)
        ys = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        for t in range(len(phi), len(ys)):
            density = emission.density(t, ys)
            self.assertFalse(jnp.allclose(density, jnp.ones_like(density)),
                             f"Density at t={t} should NOT be 1.0 when t >= k")

    #def test_lag_values_are_previous_observations(self):
    #    """The AR term should use exactly y[t-k:t] as the lag values.
    #    Mu is calculated as mu = base_mu + ()
    #    """
    #    phi = jnp.array([[0.5, 0.5, 0.5],
    #              [0.3, 0.3, 0.3]])  # (k=2, num_states=3)
    #    emission = AutoregressiveGaussEmission.from_params(self.mean, self.sigma, phi)
    #    ys = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
    #    t = 3  # should use lags y[1]=20, y[2]=30
#
    #    base_mu = emission.mu_vals(t, ys)
    #    actual_mu = emission.mu(t, ys)
#
    #    lags = ys[t - 2:t]  # [20.0, 30.0]
    #    phi_vals = emission.phi()
    #    expected_ar = jnp.sum(phi_vals * (lags[:, None] - base_mu[None, :]), axis=0)
    #    expected_mu = base_mu + expected_ar
#
    #    self.assertTrue(jnp.allclose(actual_mu, expected_mu),
                       # f"AR mean should use lags y[{t-2}:{t}]={lags}, got {actual_mu} vs {expected_mu}")

    def test_mu_equals_base_mu_when_no_lags(self):
        """For t < k, mu should equal base_mu (no AR contribution)."""
        phi = jnp.array([[0.5, 0.5, 0.5],
                                  [0.3, 0.3, 0.3]])  # (k=2, num_states=3)
        emission = AutoregressiveGaussEmission.from_params(self.mean, self.sigma, phi)
        ys = jnp.array([1.0, 2.0, 3.0, 4.0])

        for t in range(len(phi)):
            base_mu = emission.mu_vals(t, ys)
            actual_mu = emission.mu(t, ys)
            self.assertTrue(jnp.allclose(actual_mu, base_mu),
                            f"At t={t} < k, mu should equal base_mu (no AR term)")
            

    def test_zero_phi_gives_base_mu(self):
        """When all phi are zero, AR term vanishes — mu should equal base_mu for all t."""
        phi = jnp.array([[0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0]])  # (k=2, num_states=3)
        emission = AutoregressiveGaussEmission.from_params(self.mean, self.sigma, phi)
        ys = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        for t in range(len(ys)):
            base_mu = emission.mu_vals(t, ys)
            actual_mu = emission.mu(t, ys)
            self.assertTrue(jnp.allclose(actual_mu, base_mu, atol=1e-5),
                            f"With phi=0, mu at t={t} should equal base_mu")

    def test_single_lag(self):
        """With k=1, only y[t-1] should contribute to the AR term."""
        phi = jnp.array([[0.8, 0.8, 0.8]])  # k=1
        emission = AutoregressiveGaussEmission.from_params(self.mean, self.sigma, phi)
        ys = jnp.array([5.0, 10.0, 15.0])
        t = 2

        base_mu = emission.mu_vals(t, ys)
        actual_mu = emission.mu(t, ys)

        phi_val = emission.phi()
        expected_mu = base_mu + phi_val * (ys[1] - base_mu)
        self.assertTrue(jnp.allclose(actual_mu, expected_mu.squeeze()),
                        f"With k=1 at t=2, AR should use only y[1]={ys[1]}")

    def test_density_shape_consistent(self):
        """Density shape should be (1, num_states) regardless of t < k or t >= k."""
        phi = jnp.array([[0.5, 0.3, 0.2],])
        emission = AutoregressiveGaussEmission.from_params(self.mean, self.sigma, phi)
        ys = jnp.array([1.0, 2.0, 3.0, 4.0])
        expected_shape = (1, len(self.mean))

        for t in range(len(ys)):
            density = emission.density(t, ys)
            self.assertEqual(density.shape, expected_shape,
                             f"Density shape at t={t} should be {expected_shape}, got {density.shape}")


class TestSecondOrderMuTilingAlignment(TestCase):
    """For second-order HMMs the emission must be tied across s_{t-1}: at
    augmented-state index i = s_{t-1}*K + s_t the mean must equal the per-s_t
    base mean. These tests verify that mu_vals tiles correctly AND that its
    indexing matches the augmented-state ordering used by
    StaticTransitionHigherOrder, so the forward algorithm reads the right μ
    at the right state.
    """

    K = 4  # number of base latent states
    ORDER = 2

    def setUp(self) -> None:
        # 4 distinct base means + 16 per-pair sigmas/phi — same shape pattern
        # the second-order drivers use.
        self.base_mu = jnp.array([10.0, 20.0, 30.0, 40.0])
        self.sigma_16 = jnp.linspace(1.0, 4.0, self.K ** self.ORDER)
        self.phi_16 = jnp.linspace(0.1, 0.9, self.K ** self.ORDER).reshape(1, -1)

        self.emission = AutoregressiveGaussEmission.from_params(
            mu=self.base_mu, sigma=self.sigma_16, phi=self.phi_16
        )

        # Match how the second-order drivers construct the transition: 4
        # rows of logits per augmented state, tiled to K^ORDER = 16 rows.
        base_logits = jnp.zeros((self.K, self.K - 1))
        all_logits = jnp.tile(base_logits, (self.K, 1))  # shape (16, 3)
        self.transition = StaticTransitionHigherOrder(all_logits, order=self.ORDER)

    def test_mu_vals_has_correct_shape(self):
        mu_vals = self.emission.mu_vals(0, jnp.zeros(1))
        self.assertEqual(mu_vals.shape, (self.K ** self.ORDER,),
                         "mu_vals must be sized to the augmented state space (K^order)")

    def test_mu_vals_depends_only_on_s_t(self):
        """At augmented index i, mu_vals[i] == base_mu[i % K] = base_mu[s_t]."""
        mu_vals = self.emission.mu_vals(0, jnp.zeros(1))
        expected = jnp.tile(self.base_mu, self.K)
        self.assertTrue(jnp.allclose(mu_vals, expected),
                        f"mu_vals should be tile(base_mu, K)={expected}, got {mu_vals}")

        # Rows with the same s_t must share their μ, regardless of s_{t-1}.
        for s_t in range(self.K):
            vals = jnp.array([mu_vals[s_prev * self.K + s_t] for s_prev in range(self.K)])
            self.assertTrue(jnp.allclose(vals, self.base_mu[s_t]),
                            f"μ at s_t={s_t} should be tied to {float(self.base_mu[s_t])} "
                            f"across all s_{{t-1}}, got {vals}")

    def test_transition_matrix_shape_matches_mu_vals(self):
        T = self.transition.transition_matrix(0, jnp.zeros(1))
        mu_vals = self.emission.mu_vals(0, jnp.zeros(1))
        self.assertEqual(T.shape, (self.K ** self.ORDER, self.K ** self.ORDER))
        self.assertEqual(mu_vals.shape[0], T.shape[0],
                         "mu_vals length must equal the transition-matrix row count "
                         "so the forward algorithm indexes the same augmented state in both")

    def test_indexing_convention_matches_transition(self):
        """Transition rows are indexed by idx = s_{t-1}*K + s_t (the convention
        used by decode_possible_transitions). Verify by checking that, from
        every row i, the only non-zero next-state columns j are those whose
        decoded s_{t-1}' equals s_t — the second-order Markov constraint.
        Because mu_vals uses i % K = s_t, that's the same indexing, so the
        emission at row i corresponds to base_mu[s_t].
        """
        T = self.transition.transition_matrix(0, jnp.zeros(1))
        for i in range(self.K ** self.ORDER):
            s_t = i % self.K
            valid_next = {s_t * self.K + j for j in range(self.K)}
            nonzero = {int(j) for j in range(self.K ** self.ORDER)
                       if float(T[i, j]) > 1e-4}
            self.assertTrue(nonzero.issubset(valid_next),
                            f"Row {i} (s_prev={i // self.K}, s_t={s_t}) leaks "
                            f"probability to {nonzero - valid_next}; this would "
                            f"mean transition indexing disagrees with mu_vals "
                            f"indexing")

