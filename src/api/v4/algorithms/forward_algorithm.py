from src.base import BaseInference
from src.api.v4.hmm_models.hmm_params import HMMParams
import jax.numpy as jnp
from src.api.v4.algorithms.forward_outout import ForwardOutput
from typing import Any
from jax import lax
import jax


@jax.jit
def normalize_probs(probs: jax.Array) -> jax.Array:
    total = jnp.sum(probs)
    probs = probs / total
    return probs


class ForwardAlgorithm(BaseInference):
    def run(self, hmm_params: HMMParams, carry_pre: Any, ys: jnp.ndarray, ts: jnp.ndarray | None = None, xs: jnp.ndarray | None = None, batched: bool = False) -> ForwardOutput:
        """
        Run the forward algorithm over one sequence, or over a batch of them.

        Batching is opt-in via `batched`, never inferred from the shape of `ys`: a
        single sequence is legitimately spelled either `(T,)` or `(T, 1)`, and the
        latter is indistinguishable from a batch of T length-1 sequences. Guessing
        would silently return a different likelihood rather than an error.

        With `batched=False` (the default) `ys` is one sequence of any rank and
        nothing changes. With `batched=True` the leading axis of `ys` — and of `ts`
        and `xs` when given — is the batch axis, and the result carries that axis in
        front of every field: `utt`/`ut` of shape (B, T, 1, num_states) and `ft` of
        shape (B, T). Since `ForwardOutput.log_likelihood` sums log(ft) over every
        axis, the batch log-likelihood is the sum of the per-sequence ones, which is
        the likelihood of independent sequences under shared parameters.
        """
        if not batched:
            self._validate_inputs(hmm_params, ys, ts, xs, carry_pre)
            return self.run_sequence(hmm_params, carry_pre, ys, ts, xs)

        self._validate_batched_inputs(hmm_params, ys, ts, xs, carry_pre)

        # `carry_pre` is closed over rather than mapped: every sequence starts from
        # the same initial state distribution. `ys`/`ts`/`xs` are sliced per
        # sequence, which is also what an autoregressive emission needs -- its
        # `densities` closes over the whole `ys` it is handed, so each sequence sees
        # only its own history and no lag reaches across a sequence boundary.
        # `None` is an empty pytree, so the optional arguments need no special case.
        return jax.vmap(
            lambda y, t, x: self.run_sequence(hmm_params, carry_pre, y, t, x),
            in_axes=(0, None if ts is None else 0, None if xs is None else 0),
        )(ys, ts, xs)

    def run_sequence(self, hmm_params: Any, carry_pre: Any, ys: jnp.ndarray, ts: jnp.ndarray| None, xs: jnp.ndarray | None = None) -> Any:
        """
        Run the forward algorithm over a sequence using jax.lax.scan.

        Everything model-specific is precomputed in two batched calls before the
        scan: one transition matrix and one emission density per observation. The
        scan body is then pure linear algebra over those arrays, with no reference
        to the model at all.

        `indices` and `ts` are kept distinct throughout: `indices` identifies which
        observation (and so which covariate row) a step refers to, while `ts` holds
        the waiting times the continuous-time transitions need. They coincide only
        by accident for discrete models with unit spacing.
        """

        indices = jnp.arange(len(ys))
        if ts is None:
            ts = indices

        Gammas = hmm_params.transition_matrices(indices, ts, ys, xs)  # (T, num_states, num_states)
        gs = hmm_params.densities(indices, ys, xs, ts)                # (T, 1, num_states)

        carry_final, outputs = jax.lax.scan(self.step, carry_pre, (Gammas, gs))
        return self.postprocess(carry_pre, carry_final, outputs)

    def step(self, carry: Any, step_input: Any) -> Any:
        """One forward recursion step over a precomputed (Gamma, density) pair."""
        Gamma, g_t = step_input
        ut_prev = carry

        u_t = ut_prev @ Gamma
        f_t = jnp.sum(u_t * g_t)

        # To do: Make 1 if f_t is zero. This results in Density being zero, which is
        # correct. But we cannot divide by zero
        f_t = jnp.clip(f_t, a_min=1e-10)

        u_tt = u_t * g_t / f_t

        return u_tt, (u_tt, f_t, u_t)

    def postprocess(self, carry_0, carry_final, outputs) -> ForwardOutput:
        utt, ft, ut = outputs
        return ForwardOutput(ft=ft, utt=utt, ut=ut)
