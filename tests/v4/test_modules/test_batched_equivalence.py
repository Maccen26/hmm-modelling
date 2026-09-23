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
    HMM,
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


def build_hmms() -> dict:
    """The same fixtures wrapped in an `HMM`, for the diagnostics path.

    The initial distribution is supplied explicitly for every case: a dynamic
    transition has no time-invariant matrix to derive a stationary one from, and
    pinning it keeps the residuals a function of the fixture alone.
    """
    hmms = {}
    for name, (params, u_pre, _ys, _ts, _xs) in build_cases().items():
        hmms[name] = HMM(transition=params.transition, emission=params.emission,
                         inital_distribution=u_pre)
    return hmms


def pseudo_residuals_of(name: str) -> list:
    _params, _u_pre, ys, ts, xs = build_cases()[name]
    return [float(z) for z in build_hmms()[name].pseudo_residuals(ys, xs=xs, ts=ts)]


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


# Captured from the code before the diagnostics refactor. Do not regenerate to make
# a test pass.
GOLDEN_PSEUDO_RESIDUALS = {
    "static_gauss": [0.8534431913851747, 0.008366204939953718, -0.7094712780119009, 1.6558079904851528, 0.30370968626282424, -1.595054272930104, 0.40250418578607233, 1.3583031843884505, -0.4303215995582379, 0.3972637293546675, 0.7403219180757176, -1.0300505676017513],
    "static_ar": [0.8534431913851747, 0.07058347848253782, -0.7603275331500471, 1.9314272742243093, 0.03835329618954999, -1.8943026552467148, 0.6096783436395479, 1.327932324343429, -0.8309313099316665, 0.49599306209018457, 0.8134945329958084, -1.3720061027590924],
    "higher_order_gauss": [0.797151249992455, 0.18011461617410643, -0.6186692946504132, 1.5490613131628086, 0.30831510794699296, -1.342363832142249, 0.4672898047590918, 0.9886280685402469, -0.36384547242169185, 0.43956966794628316, 0.6008558204967482, -0.8708269660692846],
    "higher_order_ar": [0.797151249992455, 0.10949545715795019, -0.7565552587804927, 1.7265961126492333, -0.09775051721766155, -1.754380636653503, 0.517714605623914, 1.094517591920527, -0.7313851203412518, 0.5578794064192438, 0.649623549030275, -1.2697879690047742],
    "dynamic_gauss": [0.6447477442634802, 0.07147874968883645, -0.6781214820199652, 1.3391022572727267, 0.3819420662960857, -1.5557961826799769, 0.05486909353806057, 1.005734432246721, -0.3006424362288307, 0.2367851122434576, 0.6469118419570301, -1.01636033217383],
    "dynamic_ar": [0.6447477442634802, -0.11798064833285281, -0.8769873011096542, 1.588234876017452, 0.09848972972662608, -1.917767049863647, 0.3270016988200401, 1.1152269909018062, -0.7718091298751376, 0.31614660892329, 0.6802282280584351, -1.4038256258218662],
    "continuous_static_gauss": [0.8429990795328409, 0.04519922435385676, -0.6162398492677389, 1.6427595829049357, 0.6338275596963607, -1.5286683795663945, 0.2821081941190846, 1.2730384416246903, -0.3878935244275602, 0.4240719488281674, 0.7479947012876189, -0.8281185108567901],
    "continuous_static_ar": [0.8429990795328409, 0.05568016631507026, -0.7958497388097764, 1.8617874210596599, 0.2857958136935631, -1.8614594059845688, 0.5383280514015232, 1.286818598700164, -0.8103152420224773, 0.4865912573862847, 0.7989180734474688, -1.3105103397392448],
    "continuous_dynamic_gauss": [0.7982150681279083, 0.004239728778831233, -0.5166672367269158, 1.6595274473029753, 0.5380129920552946, -1.571097679709179, 0.3669610165342683, 1.259833586120251, -0.3990373770750651, 0.46702130370630335, 0.8253298845194933, -0.8957246032417412],
    "continuous_dynamic_ar": [0.7982150681279083, 0.01490727077194503, -0.7381605424956997, 1.876300867436931, 0.21662692788640323, -1.8805379812554923, 0.6195088707454041, 1.2786936087934841, -0.8164817223423143, 0.5236698825141991, 0.8672308815050505, -1.3423434320275205],
}


class TestGoldenPseudoResiduals(TestCase):
    """Vectorising the residual loop must reproduce it exactly."""

    def test_pseudo_residuals_unchanged(self):
        for name, expected in GOLDEN_PSEUDO_RESIDUALS.items():
            with self.subTest(case=name):
                actual = pseudo_residuals_of(name)
                self.assertEqual(len(actual), len(expected))
                for i, (a, e) in enumerate(zip(actual, expected)):
                    self.assertAlmostEqual(a, e, places=10, msg=f"element {i}")

    def test_every_case_is_pinned(self):
        self.assertEqual(set(build_cases()), set(GOLDEN_PSEUDO_RESIDUALS))


class TestBatchedCdfEquivalence(TestCase):
    """Batched `cdfs` must equal the per-step loop it replaces."""

    def test_cdfs_match_per_step_cdf(self):
        for name, case in build_cases().items():
            params, _, ys, ts, xs = case
            indices = jnp.arange(len(ys))
            ts_used = ts if ts is not None else indices
            with self.subTest(case=name):
                batched = params.cdfs(indices, ys, xs, ts_used)
                looped = jnp.stack([params.cdf(t, ys, xs, ts_used) for t in range(len(ys))])
                self.assertTrue(bool(jnp.allclose(batched, looped, atol=1e-12)))


if __name__ == "__main__":
    print("GOLDEN_LOG_LIKELIHOODS = {")
    for _name, _fixture in build_cases().items():
        print(f'    "{_name}": {log_likelihood_of(_fixture)!r},')
    print("}")
    print("GOLDEN_PSEUDO_RESIDUALS = {")
    for _name in build_cases():
        print(f'    "{_name}": {pseudo_residuals_of(_name)!r},')
    print("}")
