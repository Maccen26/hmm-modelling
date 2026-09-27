"""Tests for the opt-in batched path of the forward algorithm.

Batching is a likelihood-only feature: `run(..., batched=True)` treats the leading
axis of `ys` as a batch of independent sequences, and the log-likelihood of the batch
must equal the sum of the per-sequence log-likelihoods. That additivity is the
property worth pinning -- it is what makes fitting one parameter set to several
series meaningful -- so most of these tests are a variation on it.

The other half of the module guards the boundary: a 2-D `ys` with `batched=False` is
still one sequence (a single sequence is legitimately spelled `(T, 1)`), and the
per-sequence diagnostics refuse batched input rather than reading the batch axis as
time.
"""
import jax
jax.config.update("jax_enable_x64", True)

from unittest import TestCase

import jax.numpy as jnp

from src.api.v4 import (
    HMM,
    AutoregressiveGaussEmission,
    ContinuousStaticTransition,
    DynamicTransition,
    ForwardAlgorithm,
    GaussEmission,
    HMMParams,
    StaticTransition,
)

# Two independent sequences of equal length, plus their covariates and gaps.
YS_A = jnp.array([1.2, 0.4, -0.6, 2.1, 0.9, -1.3])
YS_B = jnp.array([-0.3, 1.5, 0.8, -0.9, 0.1, 1.9])
YS_BATCH = jnp.stack([YS_A, YS_B])                       # (2, 6)

XS_A = jnp.array([[0.10, -0.20], [0.30, 0.05], [-0.40, 0.25],
                  [0.20, 0.15], [0.00, -0.35], [0.45, 0.10]])
XS_B = jnp.array([[-0.15, 0.30], [0.25, -0.05], [0.35, 0.20],
                  [-0.30, -0.10], [0.05, 0.40], [0.15, -0.25]])
XS_BATCH = jnp.stack([XS_A, XS_B])                       # (2, 6, 2)

TS_A = jnp.array([1.0, 1.0, 2.0, 1.0, 3.0, 1.0])
TS_B = jnp.array([2.0, 1.0, 1.0, 3.0, 1.0, 2.0])
TS_BATCH = jnp.stack([TS_A, TS_B])                       # (2, 6)

_GAMMA_2 = jnp.array([[0.8, 0.2], [0.3, 0.7]])
_MU_2 = jnp.array([-0.5, 1.0])
_SIGMA_2 = jnp.array([0.7, 1.1])
_PHI_2 = jnp.array([[0.3, 0.2]])
_BETA_2 = jnp.array([[[0.4], [-0.3]], [[-0.2], [0.5]]])

U_PRE_2 = jnp.array([[0.5, 0.5]])


def _static_params():
    return HMMParams(transition=StaticTransition.from_params(_GAMMA_2),
                     emission=GaussEmission.from_params(mu=_MU_2, sigma=_SIGMA_2))


def _ll(params, ys, ts=None, xs=None, batched=False):
    output = ForwardAlgorithm().run(params, U_PRE_2, ys=ys, ts=ts, xs=xs, batched=batched)
    return float(output.log_likelihood())


class TestBatchedAdditivity(TestCase):
    """The batch log-likelihood is the sum of the per-sequence log-likelihoods."""

    def test_static_gauss(self):
        params = _static_params()
        expected = _ll(params, YS_A) + _ll(params, YS_B)
        self.assertAlmostEqual(_ll(params, YS_BATCH, batched=True), expected, delta=1e-10)

    def test_autoregressive_emission_does_not_bleed_across_sequences(self):
        # The AR emission reaches back to earlier lags of whatever `ys` it is handed,
        # so this is the case that would break if the batch were flattened into one
        # long sequence instead of sliced per sequence.
        params = HMMParams(
            transition=StaticTransition.from_params(_GAMMA_2),
            emission=AutoregressiveGaussEmission.from_params(
                mu=_MU_2, sigma=_SIGMA_2, phi=_PHI_2),
        )
        expected = _ll(params, YS_A) + _ll(params, YS_B)
        self.assertAlmostEqual(_ll(params, YS_BATCH, batched=True), expected, delta=1e-10)

    def test_with_covariates(self):
        params = HMMParams(
            transition=DynamicTransition(
                transition_logits=jnp.array([[0.2], [-0.4]]), beta=_BETA_2),
            emission=GaussEmission.from_params(mu=_MU_2, sigma=_SIGMA_2),
        )
        expected = (_ll(params, YS_A, xs=XS_A) + _ll(params, YS_B, xs=XS_B))
        got = _ll(params, YS_BATCH, xs=XS_BATCH, batched=True)
        self.assertAlmostEqual(got, expected, delta=1e-10)

    def test_with_waiting_times(self):
        params = HMMParams(
            transition=ContinuousStaticTransition(jnp.array([[-1.0], [-0.7]])),
            emission=GaussEmission.from_params(mu=_MU_2, sigma=_SIGMA_2),
        )
        expected = (_ll(params, YS_A, ts=TS_A) + _ll(params, YS_B, ts=TS_B))
        got = _ll(params, YS_BATCH, ts=TS_BATCH, batched=True)
        self.assertAlmostEqual(got, expected, delta=1e-10)

    def test_per_sequence_gaps_are_not_shared(self):
        # Swapping one sequence's gaps must move the batch likelihood, otherwise
        # `ts` is being broadcast rather than mapped over the batch axis.
        params = HMMParams(
            transition=ContinuousStaticTransition(jnp.array([[-1.0], [-0.7]])),
            emission=GaussEmission.from_params(mu=_MU_2, sigma=_SIGMA_2),
        )
        shared = jnp.stack([TS_A, TS_A])
        self.assertNotAlmostEqual(
            _ll(params, YS_BATCH, ts=TS_BATCH, batched=True),
            _ll(params, YS_BATCH, ts=shared, batched=True),
            places=6,
        )


class TestSingleSequenceUnchanged(TestCase):
    """`batched=False` keeps its old meaning for every spelling of one sequence."""

    def test_column_array_is_one_sequence(self):
        # (T, 1) is a single sequence, not a batch of T length-1 sequences. Inferring
        # from ys.ndim used to read it as the latter and silently return a different
        # likelihood.
        params = _static_params()
        flat = _ll(params, YS_A)
        column = _ll(params, YS_A.reshape(-1, 1))
        self.assertAlmostEqual(column, flat, delta=1e-10)

    def test_batch_shaped_input_without_the_flag_does_not_silently_batch(self):
        # A (B, T) array read as one sequence makes the emission return one density
        # per row, which the forward recursion cannot carry. It raises rather than
        # quietly returning the batched likelihood -- the point of the opt-in flag.
        params = _static_params()
        with self.assertRaises(TypeError):
            _ll(params, YS_BATCH)


class TestBatchedShapes(TestCase):
    def test_outputs_carry_a_leading_batch_axis(self):
        params = _static_params()
        output = ForwardAlgorithm().run(params, U_PRE_2, ys=YS_BATCH, batched=True)
        batch, time = YS_BATCH.shape
        num_states = _MU_2.shape[0]
        self.assertEqual(output.utt.shape, (batch, time, 1, num_states))
        self.assertEqual(output.ut.shape, (batch, time, 1, num_states))
        self.assertEqual(output.ft.shape, (batch, time))

    def test_per_sequence_slices_match_the_single_sequence_run(self):
        params = _static_params()
        batched = ForwardAlgorithm().run(params, U_PRE_2, ys=YS_BATCH, batched=True)
        single = ForwardAlgorithm().run(params, U_PRE_2, ys=YS_B)
        self.assertTrue(jnp.allclose(batched.utt[1], single.utt, atol=1e-12))
        self.assertTrue(jnp.allclose(batched.ft[1], single.ft, atol=1e-12))


class TestBatchedValidation(TestCase):
    def test_one_dimensional_ys_is_rejected(self):
        params = _static_params()
        with self.assertRaises(ValueError):
            ForwardAlgorithm().run(params, U_PRE_2, ys=YS_A, batched=True)

    def test_mismatched_sequence_length_is_rejected(self):
        params = _static_params()
        with self.assertRaises(ValueError):
            ForwardAlgorithm().run(params, U_PRE_2, ys=YS_BATCH,
                                   xs=XS_BATCH[:, :-1], batched=True)

    def test_mismatched_batch_size_is_rejected(self):
        params = _static_params()
        with self.assertRaises(ValueError):
            ForwardAlgorithm().run(params, U_PRE_2, ys=YS_BATCH,
                                   ts=TS_BATCH[:1], batched=True)


class TestBatchedFit(TestCase):
    def _hmm(self):
        return HMM(transition=StaticTransition.from_params(_GAMMA_2),
                   emission=GaussEmission.from_params(mu=_MU_2, sigma=_SIGMA_2),
                   inital_distribution=U_PRE_2)

    def test_fit_improves_the_batch_log_likelihood(self):
        hmm = self._hmm()
        before = hmm.log_likelihood(YS_BATCH, batched=True)
        hmm.fit(ys=YS_BATCH, tol=1e-6, batched=True)
        after = hmm.log_likelihood(YS_BATCH, batched=True)
        self.assertGreater(after, before)
        self.assertAlmostEqual(after, hmm.ll_fits[-1], delta=1e-6)

    def test_fitted_log_likelihood_is_still_additive(self):
        hmm = self._hmm()
        hmm.fit(ys=YS_BATCH, tol=1e-6, batched=True)
        expected = hmm.log_likelihood(YS_A) + hmm.log_likelihood(YS_B)
        self.assertAlmostEqual(hmm.log_likelihood(YS_BATCH, batched=True),
                               expected, delta=1e-8)

    def test_state_results_refuses_a_batched_fit(self):
        hmm = self._hmm()
        hmm.fit(ys=YS_BATCH, tol=1e-2, batched=True)
        with self.assertRaises(NotImplementedError):
            hmm.state_results

    def test_pseudo_residuals_refuse_batched_ys(self):
        hmm = self._hmm()
        with self.assertRaises(ValueError):
            hmm.pseudo_residuals(YS_BATCH)

    def test_single_sequence_fit_still_exposes_state_results(self):
        hmm = self._hmm()
        hmm.fit(ys=YS_A, tol=1e-2)
        self.assertIsNotNone(hmm.state_results)
        self.assertEqual(hmm.state_results.pseudo_residuals.shape, YS_A.shape)
