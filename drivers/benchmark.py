"""Wall-clock benchmark for the v4 fitting path.

Run this before and after a refactor and diff the two JSON files:

    uv run python -m drivers.benchmark

Every run writes a new file under ``results/benchmarks/`` — nothing is ever
overwritten, so baselines survive.

Design notes, so the numbers stay comparable across runs:

* Models are built from hardcoded literals, never loaded from ``results/models/``.
  The fitted values are irrelevant here; only the amount of work matters.
* Fitting runs a fixed ``num_iters`` with ``tol=0.0`` so the convergence check can
  never fire early. A refactor changes floating-point association, so a
  convergence-based benchmark would compare different iteration counts and call the
  difference a speedup.
* Every timed JAX call is followed by ``block_until_ready()``. JAX dispatches
  asynchronously; without the block you time dispatch, not computation.
* The first call is recorded separately as the compile-inclusive number; the
  steady-state figure is the median of ``N_REPEATS`` further calls.

Caveat when reading ``fit_seconds``: ``HMM.fit`` ends by calling
``_compute_state_results``, whose per-observation Python ``cdf`` loop is a large
constant unaffected by the density-batching refactor. The ``forward_*`` and
``grad_*`` figures isolate the part that does change.
"""
import jax

jax.config.update("jax_enable_x64", True)

import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

import equinox as eqx
import jax.numpy as jnp
import jax.random as random

from src.api.v4 import (
    HMM,
    AutoregressiveGaussEmission,
    ForwardAlgorithm,
    GaussEmission,
    StaticTransition,
    StaticTransitionHigherOrder,
)
from src.api.v4.likelihoods import negative_log_likelihood
from src.api.v4.transitions.dynamic_transition import DynamicTransition
from drivers.utils import load_train_data

DATA_NAME = "b1"
TAG = "jans-split"
NUM_ITERS = 10          # fixed outer iterations
TOL = 0.0               # never converge early
N_REPEATS = 11          # steady-state repeats; median is reported
FROZEN = {"mu0": False}
OUT_DIR = "results/benchmarks"


# --- model builders (no pickle loading) ---------------------------------------

def _base_transition_matrix() -> jnp.ndarray:
    return jnp.array([[0.7, 0.1, 0.1, 0.1],
                      [0.1, 0.7, 0.1, 0.1],
                      [0.1, 0.1, 0.7, 0.1],
                      [0.1, 0.1, 0.1, 0.7]])


def _base_mu_sigma(ys: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """The same data-derived starting point `init_ordinary_hmm` uses."""
    q = jnp.quantile(ys, jnp.array([0.40, 0.60, 0.80]))
    mu = jnp.array([400.0, q[0], q[1], q[2]])
    sigma = jnp.std(ys) * jnp.ones_like(mu)
    return mu, sigma


def build_ordinary_hmm(ys: jnp.ndarray, Xs: jnp.ndarray) -> HMM:
    mu, sigma = _base_mu_sigma(ys)
    return HMM(transition=StaticTransition.from_params(_base_transition_matrix()),
               emission=GaussEmission.from_params(mu=mu, sigma=sigma))


def build_ar_2_hmm(ys: jnp.ndarray, Xs: jnp.ndarray) -> HMM:
    mu, sigma = _base_mu_sigma(ys)
    phi = jnp.array([[0.2, 0.2, 0.2, 0.2],
                     [0.2, 0.2, 0.2, 0.2]])
    return HMM(transition=StaticTransition.from_params(_base_transition_matrix()),
               emission=AutoregressiveGaussEmission.from_params(mu=mu, sigma=sigma, phi=phi))


def build_covariate_hmm(ys: jnp.ndarray, Xs: jnp.ndarray) -> HMM:
    mu, sigma = _base_mu_sigma(ys)
    base_transition = StaticTransition.from_params(_base_transition_matrix())
    transition_logits = base_transition.transition_logits
    beta = random.uniform(random.PRNGKey(0),
                          shape=(Xs.shape[1],) + transition_logits.shape,
                          minval=-1.0, maxval=1.0)
    # A DynamicTransition has no single time-invariant matrix, so the stationary
    # distribution is seeded from the covariate-free base transition instead.
    initial_distribution = HMM(transition=base_transition,
                               emission=GaussEmission.from_params(mu=mu, sigma=sigma)).u_pre
    return HMM(transition=DynamicTransition(transition_logits=transition_logits, beta=beta),
               emission=GaussEmission.from_params(mu=mu, sigma=sigma),
               inital_distribution=initial_distribution)


def build_ar_2_second_order_hmm(ys: jnp.ndarray, Xs: jnp.ndarray) -> HMM:
    mu, sigma = _base_mu_sigma(ys)
    base_logits = StaticTransition.from_params(_base_transition_matrix()).transition_logits
    # order 2 over 4 base states -> 16 augmented states, each with 3 free logits.
    transition = StaticTransitionHigherOrder(jnp.concatenate([base_logits] * 4), order=2)
    sigma_16 = jnp.sort(jnp.repeat(sigma, 4))
    phi = jnp.repeat(jnp.array([[0.2, 0.2, 0.2, 0.2],
                                [0.2, 0.2, 0.2, 0.2]]), 4, axis=1)
    return HMM(transition=transition,
               emission=AutoregressiveGaussEmission.from_params(mu=mu, sigma=sigma_16, phi=phi))


BUILD_MODELS = {
    "ordinary_hmm": build_ordinary_hmm,
    "ar_2_hmm": build_ar_2_hmm,
    "covariate_hmm": build_covariate_hmm,
    "ar_2_second_order_hmm": build_ar_2_second_order_hmm,
}


# --- timing helpers -----------------------------------------------------------

def _block(value):
    """Force JAX's async dispatch to complete so the timing is real."""
    return jax.block_until_ready(value)


def time_call(fn, n_repeats: int = N_REPEATS) -> tuple[float, float]:
    """Returns (first_call_seconds, median_steady_state_seconds).

    The first call carries JIT compilation; the rest are steady state.
    """
    start = time.perf_counter()
    _block(fn())
    first = time.perf_counter() - start

    samples = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        _block(fn())
        samples.append(time.perf_counter() - start)
    return first, statistics.median(samples)


def _loss_fns(model: HMM, ys, xs):
    """A forward-pass closure and a gradient closure over the model's parameters."""

    def nll(params):
        output = ForwardAlgorithm().run(params, model.u_pre, ys=ys, ts=None, xs=xs)
        return negative_log_likelihood(output, params)

    forward = eqx.filter_jit(nll)
    grad = eqx.filter_jit(eqx.filter_grad(nll))
    params = model.params
    return (lambda: forward(params)), (lambda: grad(params))


def benchmark_model(name: str, build, ys, Xs) -> dict:
    print(f"  {name}: building...", flush=True)
    model = build(ys, Xs)

    forward_fn, grad_fn = _loss_fns(model, ys, Xs)
    print(f"  {name}: timing forward pass...", flush=True)
    forward_first, forward_median = time_call(forward_fn)
    print(f"  {name}: timing gradient...", flush=True)
    grad_first, grad_median = time_call(grad_fn)

    print(f"  {name}: fitting ({NUM_ITERS} iterations)...", flush=True)
    fit_model = build(ys, Xs)  # fresh model, untouched by the timing calls above
    start = time.perf_counter()
    fit_model.fit(ys=ys, xs=Xs, frozen=FROZEN, num_iters=NUM_ITERS, tol=TOL)
    fit_seconds = time.perf_counter() - start

    record = {
        "model": name,
        "fit_seconds": fit_seconds,
        "num_iterations": len(fit_model.ll_fits),
        "final_log_likelihood": float(fit_model.ll_fits[-1]),
        "forward_first_call_seconds": forward_first,
        "forward_median_seconds": forward_median,
        "grad_first_call_seconds": grad_first,
        "grad_median_seconds": grad_median,
    }
    print(f"  {name}: fit {fit_seconds:.2f}s over {record['num_iterations']} iters, "
          f"LL={record['final_log_likelihood']:.6f}, "
          f"forward {forward_median * 1e3:.2f}ms, grad {grad_median * 1e3:.2f}ms",
          flush=True)
    return record


# --- run metadata -------------------------------------------------------------

def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _run_label() -> str:
    sha = _git("rev-parse", "--short", "HEAD")
    dirty = "-dirty" if _git("status", "--porcelain") else ""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{sha}{dirty}_{stamp}"


def _metadata(ys, Xs, label: str) -> dict:
    return {
        "label": label,
        "git_sha": _git("rev-parse", "HEAD"),
        "git_short_sha": _git("rev-parse", "--short", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_name": DATA_NAME,
        "tag": TAG,
        "num_observations": int(len(ys)),
        "num_covariates": int(Xs.shape[1]),
        "num_iters": NUM_ITERS,
        "tol": TOL,
        "n_repeats": N_REPEATS,
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "platform": platform.platform(),
    }


def main() -> None:
    ys, Xs = load_train_data(data_name=DATA_NAME, tag=TAG)
    label = _run_label()
    print(f"Benchmarking {len(BUILD_MODELS)} models on {DATA_NAME}/{TAG} "
          f"(T={len(ys)}) as {label}", flush=True)

    results = [benchmark_model(name, build, ys, Xs)
               for name, build in BUILD_MODELS.items()]

    payload = {"metadata": _metadata(ys, Xs, label), "results": results}
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"benchmark_{label}.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
