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
from src.api.v5.inference.backward_sampler import BackwardSampler


def brute_force_path_posterior(params: Params, u0: jnp.ndarray, ys: jnp.ndarray,
                               xs: jnp.ndarray | None = None) -> dict[tuple[int, ...], float]:
    """
    p(z_0, ..., z_{T-1} | y_0, ..., y_{T-1}) for every state path, by enumeration.

    Uses the forward recursion's convention: Gamma_0 is applied to u0 before y_0.
    """
    T = len(ys)
    Gammas = np.asarray(params.transition_matrices(T, xs))
    gs = np.asarray(params.densities(ys))[:, 0, :]  # (T, K)
    delta = np.asarray(u0).ravel()
    K = len(delta)

    joint = {}
    for path in itertools.product(range(K), repeat=T):
        prob = (delta @ Gammas[0])[path[0]] * gs[0, path[0]]
        for t in range(1, T):
            prob *= Gammas[t, path[t - 1], path[t]] * gs[t, path[t]]
        joint[path] = prob
    total = sum(joint.values())
    return {path: p / total for path, p in joint.items()}


def sample_paths(params: Params, u0: jnp.ndarray, ys: jnp.ndarray, num_paths: int,
                 xs: jnp.ndarray | None = None, seed: int = 0) -> np.ndarray:
    utts, _, _ = ForwardAlgorithm.run(params, u0, ys, xs)
    Gammas = params.transition_matrices(len(ys), xs)
    return np.asarray(BackwardSampler.run(num_paths, Gammas, utts, seed=seed))


def empirical_path_distribution(paths: np.ndarray) -> dict[tuple[int, ...], float]:
    uniq, counts = np.unique(paths, axis=0, return_counts=True)
    return {tuple(int(s) for s in row): c / len(paths) for row, c in zip(uniq, counts)}


class TestForwardBackwardIntegration(TestCase):
    """
    Runs ForwardAlgorithm -> BackwardSampler and checks that the sampled
    state paths follow the exact posterior p(z_{0:T-1} | y_{0:T-1}).
    """

    num_paths = 40_000

    def setUp(self) -> None:
        self.K, self.T, self.C = 2, 5, 2
        self.P = jnp.array([[0.85, 0.15],
                            [0.25, 0.75]])
        self.mean = jnp.array([0.0, 1.5])
        self.sigma = jnp.array([1.0, 0.7])
        self.u0 = jnp.array([[0.6, 0.4]])

        rng = np.random.default_rng(2)
        self.ys = jnp.asarray(rng.normal(0.75, 1.0, self.T))
        self.xs = jnp.asarray(rng.integers(0, 3, (self.T, self.C)), dtype=float)
        self.beta = jnp.asarray(rng.normal(0, 0.3, (self.C, self.K, self.K - 1)))

        self.params = Params(transition=StaticTransition.from_params(self.P),
                             emission=GaussEmission.from_params(self.mean, self.sigma))

    def assert_matches_posterior(self, params: Params, xs: jnp.ndarray | None = None) -> None:
        exact = brute_force_path_posterior(params, self.u0, self.ys, xs)
        empirical = empirical_path_distribution(sample_paths(params, self.u0, self.ys, self.num_paths, xs))
        for path, p in exact.items():
            # ~4 standard errors of a binomial proportion
            tol = 4 * np.sqrt(p * (1 - p) / self.num_paths) + 1e-3
            self.assertAlmostEqual(empirical.get(path, 0.0), p, delta=tol, msg=f"path {path}")

    def test_static_transition_paths_follow_posterior(self):
        self.assert_matches_posterior(self.params)

    def test_dynamic_transition_paths_follow_posterior(self):
        params = Params(transition=DynamicTransition.from_params(self.P, self.beta),
                        emission=self.params.emission)
        self.assert_matches_posterior(params, self.xs)

    def test_smoothed_marginals_match_posterior(self):
        exact = brute_force_path_posterior(self.params, self.u0, self.ys)
        expected = np.zeros((self.T, self.K))
        for path, p in exact.items():
            expected[np.arange(self.T), path] += p
        paths = sample_paths(self.params, self.u0, self.ys, self.num_paths)
        empirical = np.stack([np.bincount(paths[:, t], minlength=self.K) for t in range(self.T)]) / self.num_paths
        np.testing.assert_allclose(empirical, expected, atol=0.015)

    def test_last_state_follows_last_filtered_distribution(self):
        utts, _, _ = ForwardAlgorithm.run(self.params, self.u0, self.ys)
        paths = sample_paths(self.params, self.u0, self.ys, self.num_paths)
        freq = np.bincount(paths[:, -1], minlength=self.K) / self.num_paths
        np.testing.assert_allclose(freq, np.asarray(utts[-1, 0]), atol=0.015)
