import jax
jax.config.update("jax_enable_x64", True)

from unittest import TestCase
import numpy as np
import jax.numpy as jnp

from src.api.v5.inference.backward_sampler import BackwardSampler


class TestBackwardSampler(TestCase):
    """
    Unit tests for BackwardSampler (backward sampling of state paths from
    precomputed transition matrices and filtered state distributions).
    """

    def setUp(self) -> None:
        self.K, self.T = 3, 8
        self.P = jnp.array([[0.8, 0.1, 0.1],
                            [0.1, 0.8, 0.1],
                            [0.1, 0.1, 0.8]])
        self.Gammas = jnp.tile(self.P, (self.T, 1, 1))  # (T, K, K)
        rng = np.random.default_rng(0)
        self.utt = jnp.asarray(rng.dirichlet(np.ones(self.K), self.T))  # (T, K)

    # --- output structure -------------------------------------------------

    def test_step_output_shape_and_dtype(self):
        path = BackwardSampler.step(jax.random.key(0), self.Gammas, self.utt)
        self.assertEqual(path.shape, (self.T,))
        self.assertTrue(jnp.issubdtype(path.dtype, jnp.integer))

    def test_run_output_shape(self):
        paths = BackwardSampler.run(5, self.Gammas, self.utt, seed=0)
        self.assertEqual(paths.shape, (5, self.T))

    def test_states_within_range(self):
        paths = BackwardSampler.run(200, self.Gammas, self.utt, seed=1)
        self.assertTrue(bool(jnp.all(paths >= 0)))
        self.assertTrue(bool(jnp.all(paths < self.K)))

    def test_single_time_step(self):
        paths = BackwardSampler.run(10, self.Gammas[:1], self.utt[:1], seed=0)
        self.assertEqual(paths.shape, (10, 1))

    def test_single_state(self):
        paths = BackwardSampler.run(10, jnp.ones((self.T, 1, 1)), jnp.ones((self.T, 1)), seed=0)
        self.assertTrue(bool(jnp.all(paths == 0)))

    # --- randomness -------------------------------------------------------

    def test_same_seed_is_reproducible(self):
        a = BackwardSampler.run(20, self.Gammas, self.utt, seed=3)
        b = BackwardSampler.run(20, self.Gammas, self.utt, seed=3)
        self.assertTrue(bool(jnp.array_equal(a, b)))

    def test_different_seeds_give_different_paths(self):
        a = BackwardSampler.run(20, self.Gammas, self.utt, seed=3)
        b = BackwardSampler.run(20, self.Gammas, self.utt, seed=4)
        self.assertFalse(bool(jnp.array_equal(a, b)))

    def test_paths_within_a_batch_differ(self):
        paths = BackwardSampler.run(20, self.Gammas, self.utt, seed=0)
        self.assertGreater(len(np.unique(np.asarray(paths), axis=0)), 1)

    def test_jit_matches_eager(self):
        key = jax.random.key(7)
        eager = BackwardSampler.step(key, self.Gammas, self.utt)
        jitted = jax.jit(BackwardSampler.step)(key, self.Gammas, self.utt)
        self.assertTrue(bool(jnp.array_equal(eager, jitted)))

    # --- sampling correctness ---------------------------------------------

    def test_identity_transition_gives_constant_paths(self):
        Gammas = jnp.tile(jnp.eye(self.K), (self.T, 1, 1))
        paths = BackwardSampler.run(100, Gammas, self.utt, seed=0)
        self.assertTrue(bool(jnp.all(paths == paths[:, -1:])))

    def test_deterministic_cycle_is_followed_backwards(self):
        # 0 -> 1 -> 2 -> 0: given z_{t+1}, z_t must be (z_{t+1} - 1) mod K
        cycle = jnp.roll(jnp.eye(self.K), shift=1, axis=1)
        Gammas = jnp.tile(cycle, (self.T, 1, 1))
        paths = BackwardSampler.run(100, Gammas, self.utt, seed=0)
        self.assertTrue(bool(jnp.all(paths[:, :-1] == (paths[:, 1:] - 1) % self.K)))

    def test_zero_probability_states_never_sampled(self):
        utt = self.utt.at[:, 2].set(0.0)
        utt = utt / utt.sum(axis=1, keepdims=True)
        paths = BackwardSampler.run(500, self.Gammas, utt, seed=0)
        self.assertFalse(bool(jnp.any(paths == 2)))

    def test_degenerate_filtered_distribution_is_respected(self):
        utt = self.utt.at[3].set(jnp.array([0.0, 1.0, 0.0]))
        paths = BackwardSampler.run(200, self.Gammas, utt, seed=0)
        self.assertTrue(bool(jnp.all(paths[:, 3] == 1)))

    def test_last_state_marginal_matches_filtered(self):
        n = 20_000
        paths = BackwardSampler.run(n, self.Gammas, self.utt, seed=0)
        freq = np.bincount(np.asarray(paths[:, -1]), minlength=self.K) / n
        np.testing.assert_allclose(freq, np.asarray(self.utt[-1]), atol=0.015)

    def test_conditional_matches_backward_formula(self):
        # P(z_{T-2} = i | z_{T-1} = j) = utt[T-2, i] * Gamma[T-1, i, j] / normaliser
        n = 40_000
        P = jnp.array([[0.6, 0.3, 0.1],
                       [0.2, 0.5, 0.3],
                       [0.3, 0.3, 0.4]])
        Gammas = jnp.tile(P, (self.T, 1, 1))
        paths = np.asarray(BackwardSampler.run(n, Gammas, self.utt, seed=0))
        for j in range(self.K):
            mask = paths[:, -1] == j
            freq = np.bincount(paths[mask, -2], minlength=self.K) / mask.sum()
            expected = np.asarray(self.utt[-2] * P[:, j])
            np.testing.assert_allclose(freq, expected / expected.sum(), atol=0.03)

    def test_uses_gamma_of_next_time_step(self):
        # Gamma[t+1] (not Gamma[t]) links z_t to z_{t+1}: make only Gamma[T-1] an identity
        Gammas = jnp.tile(jnp.roll(jnp.eye(self.K), shift=1, axis=1), (self.T, 1, 1))
        Gammas = Gammas.at[-1].set(jnp.eye(self.K))
        paths = BackwardSampler.run(100, Gammas, self.utt, seed=0)
        self.assertTrue(bool(jnp.all(paths[:, -2] == paths[:, -1])))

    # --- run: input handling and validation -------------------------------

    def test_run_matches_run_algorithm(self):
        a = BackwardSampler.run(10, self.Gammas, self.utt, seed=2)
        b = BackwardSampler.run_algorithm(10, self.Gammas, self.utt, seed=2)
        self.assertTrue(bool(jnp.array_equal(a, b)))

    def test_run_accepts_forward_shaped_utt(self):
        # ForwardAlgorithm returns utt with shape (T, 1, K)
        a = BackwardSampler.run(10, self.Gammas, self.utt[:, None, :], seed=2)
        b = BackwardSampler.run(10, self.Gammas, self.utt, seed=2)
        self.assertTrue(bool(jnp.array_equal(a, b)))

    def test_run_default_seed(self):
        a = BackwardSampler.run(10, self.Gammas, self.utt)
        b = BackwardSampler.run(10, self.Gammas, self.utt, seed=0)
        self.assertTrue(bool(jnp.array_equal(a, b)))

    def test_invalid_num_paths_raises(self):
        for num_paths in (0, -1, 2.5):
            with self.subTest(num_paths=num_paths), self.assertRaises(ValueError):
                BackwardSampler.run(num_paths, self.Gammas, self.utt)  # type: ignore[arg-type]

    def test_invalid_seed_raises(self):
        for seed in (-1, 1.5):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                BackwardSampler.run(5, self.Gammas, self.utt, seed=seed)  # type: ignore[arg-type]

    def test_invalid_gammas_raise(self):
        bad = {
            "2d": self.P,
            "non_square": jnp.ones((self.T, self.K, self.K + 1)),
            "numpy": np.asarray(self.Gammas),
        }
        for name, Gammas in bad.items():
            with self.subTest(name), self.assertRaises(ValueError):
                BackwardSampler.run(5, Gammas, self.utt)  # type: ignore[arg-type]

    def test_invalid_utt_raises(self):
        bad = {
            "4d": self.utt[:, None, None, :],
            "3d_wrong_middle": jnp.tile(self.utt[:, None, :], (1, 2, 1)),
            "numpy": np.asarray(self.utt),
        }
        for name, utt in bad.items():
            with self.subTest(name), self.assertRaises(ValueError):
                BackwardSampler.run(5, self.Gammas, utt)  # type: ignore[arg-type]

    def test_mismatched_time_dimension_raises(self):
        with self.assertRaises(ValueError):
            BackwardSampler.run(5, self.Gammas[:-1], self.utt)
