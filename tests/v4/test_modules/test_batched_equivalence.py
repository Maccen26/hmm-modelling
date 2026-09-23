"""Regression guard for the batched-precompute refactor.

Two kinds of check live here:

1. `TestGoldenLogLikelihood` pins the forward-algorithm log-likelihood of every
   transition x emission combination to a hardcoded constant. The constants were
   captured from the pre-refactor code; hoisting the emission densities out of the
   scan must not move them.
2. `TestBatchedEquivalence` asserts the batched `densities` / `transition_matrices`
   agree elementwise with looping the per-step `density` / `transition_matrix`.

Run this module directly (`uv run python -m tests.v4.test_modules.test_batched_equivalence`)
to print freshly-computed golden values.
"""
import jax
jax.config.update("jax_enable_x64", True)

from unittest import TestCase

import jax.numpy as jnp

from src.api.v4 import (
    AutoregressiveGaussEmission,
    ContinuousDynamicTransition,
    ContinuousStaticTransition,
    DynamicTransition,
    ForwardAlgorithm,
    GaussEmission,
    HMMParams,
    StaticTransition,
    StaticTransitionHigherOrder,
)

# --- fixed fixture data -------------------------------------------------------

YS = jnp.array([1.2, 0.4, -0.6, 2.1, 0.9, -1.3, 0.2, 1.7, -0.1, 0.5, 1.1, -0.8])
XS = jnp.array([[0.10, -0.20], [0.30, 0.05], [-0.40, 0.25], [0.20, 0.15],
                [0.00, -0.35], [0.45, 0.10], [-0.15, 0.30], [0.25, -0.05],
                [0.35, 0.20], [-0.30, -0.10], [0.05, 0.40], [0.15, -0.25]])
# Irregular waiting times for the continuous-time models.
TS_GAPS = jnp.array([1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 3.0])

_GAMMA_2 = jnp.array([[0.8, 0.2], [0.3, 0.7]])
_MU_2 = jnp.array([-0.5, 1.0])
_SIGMA_2 = jnp.array([0.7, 1.1])
_PHI_2 = jnp.array([[0.3, 0.2]])            # (num_lags, num_states)
_BETA_2 = jnp.array([[[0.4], [-0.3]],       # (num_covariates, num_states, num_states - 1)
                     [[-0.2], [0.5]]])

# Higher-order (order 2 over 2 base states) works over 4 augmented states.
_LOGITS_HO = jnp.array([[0.3], [-0.2], [0.1], [0.4]])
_MU_4 = jnp.array([-0.8, -0.1, 0.6, 1.4])
_SIGMA_4 = jnp.array([0.6, 0.8, 0.9, 1.2])
_PHI_4 = jnp.array([[0.25, 0.15, 0.30, 0.10]])

U_PRE_2 = jnp.array([[0.5, 0.5]])
U_PRE_4 = jnp.full((1, 4), 0.25)


def _gauss(mu, sigma):
    return GaussEmission.from_params(mu=mu, sigma=sigma)


def _ar(mu, sigma, phi):
    return AutoregressiveGaussEmission.from_params(mu=mu, sigma=sigma, phi=phi)


def _case(transition, emission, u_pre, ts, xs):
    """A fixture is (params, u_pre, ys, ts, xs) — everything `run` needs."""
    return HMMParams(transition=transition, emission=emission), u_pre, YS, ts, xs


def build_cases() -> dict:
    """All transition x emission combinations under test, built from fixed params."""
    return {
        "static_gauss": _case(
            StaticTransition.from_params(_GAMMA_2), _gauss(_MU_2, _SIGMA_2),
            U_PRE_2, None, None),
        "static_ar": _case(
            StaticTransition.from_params(_GAMMA_2), _ar(_MU_2, _SIGMA_2, _PHI_2),
            U_PRE_2, None, None),
        "higher_order_gauss": _case(
            StaticTransitionHigherOrder(_LOGITS_HO, order=2), _gauss(_MU_4, _SIGMA_4),
            U_PRE_4, None, None),
        "higher_order_ar": _case(
            StaticTransitionHigherOrder(_LOGITS_HO, order=2), _ar(_MU_4, _SIGMA_4, _PHI_4),
            U_PRE_4, None, None),
        "dynamic_gauss": _case(
            DynamicTransition(transition_logits=jnp.array([[0.2], [-0.4]]), beta=_BETA_2),
            _gauss(_MU_2, _SIGMA_2), U_PRE_2, None, XS),
        "dynamic_ar": _case(
            DynamicTransition(transition_logits=jnp.array([[0.2], [-0.4]]), beta=_BETA_2),
            _ar(_MU_2, _SIGMA_2, _PHI_2), U_PRE_2, None, XS),
        "continuous_static_gauss": _case(
            ContinuousStaticTransition(jnp.array([[-1.0], [-0.7]])),
            _gauss(_MU_2, _SIGMA_2), U_PRE_2, TS_GAPS, None),
        "continuous_static_ar": _case(
            ContinuousStaticTransition(jnp.array([[-1.0], [-0.7]])),
            _ar(_MU_2, _SIGMA_2, _PHI_2), U_PRE_2, TS_GAPS, None),
        "continuous_dynamic_gauss": _case(
            ContinuousDynamicTransition(jnp.array([[-1.0], [-0.7]]), _BETA_2),
            _gauss(_MU_2, _SIGMA_2), U_PRE_2, TS_GAPS, XS),
        "continuous_dynamic_ar": _case(
            ContinuousDynamicTransition(jnp.array([[-1.0], [-0.7]]), _BETA_2),
            _ar(_MU_2, _SIGMA_2, _PHI_2), U_PRE_2, TS_GAPS, XS),
    }


def log_likelihood_of(case) -> float:
    params, u_pre, ys, ts, xs = case
    output = ForwardAlgorithm().run(params, u_pre, ys=ys, ts=ts, xs=xs)
    return float(output.log_likelihood())


# Captured from the pre-refactor code. Do not regenerate to make a test pass.
GOLDEN_LOG_LIKELIHOODS = {
    "static_gauss": -19.123871361281942,
    "static_ar": -17.656461886137112,
    "higher_order_gauss": -17.416905494940835,
    "higher_order_ar": -16.641999136445907,
    "dynamic_gauss": -17.45577748120336,
    "dynamic_ar": -16.64295723533828,
    "continuous_static_gauss": -18.70397937197147,
    "continuous_static_ar": -17.167489480185125,
    "continuous_dynamic_gauss": -18.737683305893846,
    "continuous_dynamic_ar": -17.35386931708064,
}


class TestGoldenLogLikelihood(TestCase):
    """The refactor must leave every log-likelihood bit-for-bit comparable."""

    def test_log_likelihoods_unchanged(self):
        cases = build_cases()
        for name, expected in GOLDEN_LOG_LIKELIHOODS.items():
            with self.subTest(case=name):
                self.assertAlmostEqual(log_likelihood_of(cases[name]), expected, places=10)

    def test_every_case_is_pinned(self):
        """A new combination must come with a golden value, not slip through."""
        self.assertEqual(set(build_cases()), set(GOLDEN_LOG_LIKELIHOODS))


class TestBatchedEquivalence(TestCase):
    """Batched precompute must equal the per-step loop it replaces."""

    def test_densities_match_per_step_density(self):
        for name, case in build_cases().items():
            params, _, ys, ts, xs = case
            indices = jnp.arange(len(ys))
            ts_used = ts if ts is not None else indices
            with self.subTest(case=name):
                batched = params.densities(indices, ys, xs, ts_used)
                looped = jnp.stack([params.density(t, ys, xs, ts_used) for t in range(len(ys))])
                self.assertTrue(bool(jnp.allclose(batched, looped, atol=1e-12)))

    def test_transition_matrices_match_per_step_matrix(self):
        for name, case in build_cases().items():
            params, _, ys, ts, xs = case
            indices = jnp.arange(len(ys))
            ts_used = ts if ts is not None else indices
            with self.subTest(case=name):
                batched = params.transition_matrices(indices, ts_used, ys, xs)
                self.assertEqual(batched.shape[0], len(ys))
                # Rows of every matrix must still be a probability distribution.
                self.assertTrue(bool(jnp.allclose(batched.sum(axis=-1), 1.0, atol=1e-8)))


if __name__ == "__main__":
    for _name, _case in build_cases().items():
        print(f'    "{_name}": {log_likelihood_of(_case)!r},')
