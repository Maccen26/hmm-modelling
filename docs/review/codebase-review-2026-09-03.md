# Codebase Review — HMM Modelling

**Date:** 2026-09-03
**Scope:** `src/` (source) and `tests/` (test code)
**Focus:** modularity, system design, readability, and patterns worth learning
**Reviewer's lens:** what will make you a stronger system designer, not just a bug list

---

## TL;DR

The core architecture in `src/api/v4` + `src/base` is genuinely good. You have applied the
right instincts: abstract base classes for each component (emission, transition, inference,
solver), composition of an HMM out of a transition + emission, immutable functional parameter
updates, and Equinox modules so everything is a JAX pytree. This is a solid, extensible spine.

The problems are mostly around the **edges**: a dead/broken parallel module, one component
that broke out of its own abstraction (`DynamicTransition`), an over-loaded base class doing
too much, a stateful "god" controller (`HMM`), pervasive typos in public names, and misleading
comments/docstrings. None of these are hard to fix, and fixing them will teach the exact
skills you are asking about.

Priorities are labelled **[P1]** (do soon), **[P2]** (should do), **[P3]** (polish).

---

## 1. What is well designed (keep doing this)

Understanding *why* these are good is as important as the fixes.

1. **Dependency inversion via `src/base`.** Abstractions live in `src/base` and concrete
   implementations depend on them (`GaussEmission(BaseEmission)`, `ForwardAlgorithm(BaseInference)`,
   `GradientSolver(BaseSolver)`). High-level code (`HMM`, `ForwardAlgorithm`) talks to
   `BaseTransition`/`BaseEmission`, never to a concrete class. This is textbook and it is the
   reason you can add an emission type without touching the forward algorithm.

2. **Composition over inheritance for the model.** `HMMParams` *has a* transition and *has an*
   emission (`src/base/base_hmm.py:17-18`) rather than trying to be one. That is the correct
   axis of variation for this domain.

3. **Strategy pattern for solvers and inference.** `HMM.fit(solver=...)` and
   `_set_inference_algorithm` let you swap algorithms without changing the model. Good.

4. **Immutable, functional parameter updates.** `update_param` returns a *new* instance
   (`src/base/base_hmm.py:61-74`) instead of mutating. This is exactly right for JAX and makes
   reasoning about state easy.

5. **Shared test suite via inheritance.** `TestBaseHMM` with `__test__ = False`, subclassed by
   `TestStaticHMM` and `TestAutoRegressiveHMM` (`tests/v4/test_modules/test_hmm.py:88-106`), runs
   the same contract against multiple model variants. This is a professional pattern — one
   behavioural spec, many implementations.

Hold on to these. Everything below is about protecting them from erosion.

---

## 2. Structural / modularity issues

### 2.1 [P1] `src/optim/` is dead, broken, duplicated code — delete or quarantine it
`src/optim/base.py:6`, `src/optim/loss.py:4` (and `lbfgs.py`, `minimizer.py`) all import
`from src.deprecated.base.hmm import HMM`. **`src/deprecated/` does not exist** — these modules
cannot be imported at all. They also *duplicate* the working solver stack in
`src/api/v4/solvers/` and define a second `negative_log_likelihood` that competes conceptually
with `src/api/v4/likelihoods.py`.

**Why it matters:** dead parallel modules are the single biggest source of confusion for a new
reader (or future you). "Which minimizer is real?" is a question nobody should have to ask.

**Fix:** delete `src/optim/` entirely, or if you want the `profile_likelihood` idea preserved,
move that one function into the live tree and rewrite it against `src/api/v4`. Do not keep
broken imports in the repo.

### 2.2 [P1] `DynamicTransition` broke out of its abstraction
`src/api/v4/transitions/dynamic_transition.py` subclasses `eqx.Module` **directly**, not
`BaseTransition` (the base import is even commented out on line 3). Consequences:

- It re-implements `__iter__` by hand (line 64) — duplicating `BaseTransition.__iter__`.
- It has **no** `from_params`, no `update_param`, no `transition_logits` contract guarantee —
  so the solver's freeze-by-name machinery and `BaseHMM.update_param` may silently not apply to it.
- **Argument order diverges from the interface.** `BaseTransition.step(self, t, ys, xs)` vs
  `DynamicTransition.step(self, t, xs, ys=None)` (line 29) — `ys` and `xs` are swapped. Same for
  `transition_matrix` (line 46). Today it happens to work only because `HMMParams.transition_matrix`
  calls with keyword args; the moment anything calls positionally, it silently uses covariates as
  observations. This is a latent, hard-to-find bug.
- **Copy-paste docstring rot:** the class docstring says *"The transition matrix does not depend on
  the covariates"* (line 11) — the opposite of what this class does.
- **Junk import:** `from matplotlib.pylab import beta` (line 3) pulls a stats function through
  matplotlib and shadows nothing useful. Leftover; remove it.

**Fix:** make `DynamicTransition(BaseTransition)`, keep the `step(self, t, ys, xs)` signature
identical to the base (read `xs` inside), store `beta` as an extra field, and delete the hand-rolled
`__iter__`. This restores Liskov substitutability — the whole point of having `BaseTransition`.

### 2.3 [P2] `BaseSolver` is doing three jobs — extract the freezing concern
`src/base/base_solver.py` is ~130 lines, of which ~100 are parameter-freezing machinery
(`_parse_frozen`, `_build_filter_spec`, `_freeze_elements`, `_restore_frozen_elements`,
`_build_loss_fn`). The three concrete solvers are thin by comparison. The base class is really
two things wearing one hat: *"a thing that optimises"* and *"a thing that knows how to freeze
parameters."*

**Why it matters:** Single Responsibility. Freezing is a self-contained concept with its own
data (`whole` vs `element`) and its own transformations. It should be a collaborator, not baked
into every solver's ancestry.

**Fix:** introduce a small `ParameterFreezer` (or `FreezeSpec`) class:
```python
freezer = ParameterFreezer(frozen)          # parses once
trainable, static = freezer.partition(params)
loss = freezer.wrap_loss(base_loss, static, params)
result = freezer.restore(optimised_params, params)
```
Solvers then only own the optimisation loop. This also makes freezing unit-testable in
isolation (you already have those tests — they'd move cleanly).

### 2.4 [P2] The abstract solver knows about the concrete forward algorithm
`BaseSolver._build_loss_fn` (`src/base/base_solver.py:124-133`) hardcodes
`from ...forward_algorithm import ForwardAlgorithm` and `negative_log_likelihood`. The *abstract*
optimiser is coupled to *one concrete* inference algorithm. That is a layering inversion — and
it forces the local (function-body) import to dodge a circular dependency, which is itself a
smell telling you the arrow points the wrong way.

**Fix:** inject the inference algorithm and loss. The thing that builds the loss (the `HMM`, or a
dedicated `Objective`) should hand the solver a ready-made `loss(params) -> scalar`. The solver
should not import `ForwardAlgorithm` at all. Notice `HMM` already holds `self.negative_log_likelihood`
and could own an inference algorithm too — that's the natural home for objective construction.

### 2.5 [P2] `HMM` is a stateful "god" controller
`src/api/v4/hmm_models/hmm.py` (the `HMM` facade) currently owns: parameters, the initial
distribution, the loss function, the running list of fit likelihoods, the fitted results, state
results, fitting, log-likelihood evaluation, pseudo-residual computation, and stationary-distribution
solving. That is a lot of responsibilities for one object, and it is mutable (`self.params` is
reassigned during `fit`).

This is a reasonable *facade* to start from, but as it grows it will become hard to test and reason
about. Think about separating along these seams:
- **Model container** (immutable): transition + emission + initial distribution. (`HMMParams` is
  already most of this.)
- **Fitter/Trainer**: takes a model + data + solver, returns a fitted model + `HMMResults`.
- **Diagnostics**: log-likelihood, pseudo-residuals, AIC/BIC — pure functions of a fitted model + data.

You don't have to do this now, but naming the seams helps you resist piling more onto `HMM`.

---

## 3. Correctness / behavioural concerns

### 3.1 [P1] `HMM.fit` runs a confusing nested optimisation and resets the optimiser each pass
`src/api/v4/hmm_models/hmm.py:84-109`: the outer loop runs up to `num_iters=200` times, and each
pass calls `solver.fit(...)` which *itself* runs `n_iter` internal steps (500 for `GradientSolver`).
So the worst case is ~100,000 gradient steps. Worse, `GradientSolver.fit` creates a **fresh**
`opt_state = optimizer.init(...)` on every call (`src/api/v4/solvers/gradient_solver.py:25`), so
Adam's momentum is reset every outer iteration — you are not running one smooth optimisation, you
are running 200 short ones from cold. The convergence check compares the loss *between* these cold
restarts.

**Fix:** pick one level of iteration. Either (a) let the solver own convergence and call it once, or
(b) make the solver do a *single* step per call and drive iteration/convergence from `HMM`. Mixing
both, plus resetting optimiser state, muddies both semantics and performance.

### 3.2 [P2] `data.py:save_model` looks broken against the real `HMMParams` signature
`src/data.py:116` does `params = HMMParams(model)`. But `HMMParams` is a `BaseHMM`/`eqx.Module`
whose fields are `emission` and `transition` (`src/base/base_hmm.py:17-18`), so its constructor
expects `HMMParams(transition=..., emission=...)`, not a single positional `model`. If `model` is
already an `HMMParams` this wrapping is wrong; if it's an `HMM` facade it's also wrong. Verify and
fix — persistence code that's wrong is the kind of thing you only discover when you desperately
need to load a result.

### 3.3 [P2] Test builds one HMM but asserts on another
`tests/v4/test_results/test_textbook_results.py:207-232`
(`test_phi_convergence_notebooks_from_params`): it constructs `hmm` from loaded notebook params,
then fits and asserts on **`self.hmm`** (the AR hmm from `setUp`), never using the `hmm` it just
built. The test passes but doesn't test what its name claims. Either fit/assert on `hmm`, or delete
the dead construction.

### 3.4 [P3] `MODELTYPES` has a duplicated value
`src/config/modelnames.py:13-14`: `AR_HMM_UNCONSTRAINED` and `AR_HMM_UNCONSTRAINED_PRIOR` are both
`"ar_hmm_unconstrained-v2"`. Almost certainly a copy-paste bug — two logical models mapping to the
same on-disk tag will collide in persistence.

### 3.5 [P3] Dead code inside live modules
- `normalize_probs` in `src/api/v4/algorithms/forward_algorithm.py:10-14` is defined (and `@jax.jit`'d)
  but never used.
- Commented-out test blocks in `test_textbook_results.py` and `test_optimizers.py`. If they matter,
  fix and enable them; otherwise delete. Commented-out code rots and misleads.

---

## 4. Readability & naming (cheap wins, high impact)

### 4.1 [P1] Typos baked into public names
These hurt grep-ability and look unpolished, and some are in the **public API** so fixing them later
is a breaking change — do it now while the blast radius is small:
- `inital_distribution` → `initial_distribution` (public `HMM.__init__` kwarg, `hmm.py:17` etc.)
- `forward_outout.py` → `forward_output.py` (module filename, imported in several places)
- `tests/v4/intergrations/` → `integrations/`
- `trainaled_parameters` (`src/optim/base.py:17`, if you keep any of it)
- Minor: `troughout`, `covarites`, `transtions`, `inital` in docstrings throughout.

### 4.2 [P2] Misleading comments/docstrings on `ForwardOutput`
`src/api/v4/algorithms/forward_outout.py:9-11`:
- `ft` is documented "shape (T, num_states)" but it is the scalar per-timestep normaliser, shape `(T,)`.
- `utt` is documented "Observed sequence, shape (T,)" but it is the filtered probability `u_{t|t}`,
  shape `(T, num_states)`.

Wrong comments are worse than no comments — they actively mislead. Fix them to describe what the
fields actually hold. (The `log_likelihood` implementation summing `log(ft)` confirms `ft` is the
scalar likelihood contribution per step.)

### 4.3 [P2] The freeze API's `False`-means-frozen is counter-intuitive
`frozen={"mu0": False}` meaning *"mu0 is frozen"* reads backwards — a reader sees `False` and
thinks "not frozen." And overloading the dict value to be either `False` (whole freeze) or a tuple
(element freeze) packs two concepts into one field (`base_solver.py:29-35`). Consider an explicit
small type, e.g. `Freeze.whole("mu0")`, `Freeze.element("phi_tilde", (0, 1))`, or at least document
the convention loudly at the API boundary. Self-documenting inputs prevent a class of user error.

### 4.4 [P3] Hidden coupling via tiling in the AR emission
`AutoregressiveGaussEmission.mu_vals` (`autoregressive_gauss_emission.py:56-62`) computes
`n_tiles = self.log_sigma.shape[0] // base.shape[0]` and tiles the means. The relationship between
number of states, number of means, and number of sigmas is implicit and hard to follow. Either make
`num_states` an explicit (static) field, or add a comment spelling out the invariant. Magic reshaping
is where shape bugs hide.

### 4.5 [P3] Terse math names are fine — but only if comments are correct
`u_pre`, `utt`, `ut`, `ft`, `g_t`, `Gamma` match the textbook and are acceptable in this domain.
The cost is that a wrong comment (see 4.2) does real damage because the names alone don't rescue the
reader. Keep the names; keep the comments honest.

---

## 5. Duplication (DRY)

### 5.1 [P2] Pseudo-residual loop duplicated
`HMM._compute_state_results` (`hmm.py:112-122`) and `HMM.pseudo_residuals` (`hmm.py:142-153`)
contain the *same* loop computing `z_t`. There's even a `# Todo: Refactor this method` on line 141.
Extract one private helper `_pseudo_residuals(output, ys, xs)` and call it from both.

### 5.2 [P3] `update_param` duplicated between emission and transition bases
`base_emission.py:65-78` and `base_transition.py:70-83` are byte-for-byte identical. Same for the
`__iter__`/`__eq__` pairs. Consider a shared mixin (e.g. `NamedFieldsModule`) that both inherit, so
the field-iteration/equality/update logic lives in exactly one place. Right now a fix to one won't
propagate to the other.

### 5.3 [P3] Inconsistent `__eq__` semantics
`BaseHMM.__eq__` (`base_hmm.py:53-58`) compares with `==` recursively, while `BaseEmission`/
`BaseTransition.__eq__` use `jnp.allclose` with tolerances. So equality means different things at
different levels of the same object graph. Pick one policy (probably tolerance-based, factored into
the shared mixin from 5.2).

---

## 6. Smaller notes

- **[P3] `BaseInference` docstring is stale.** `src/base/base_inference.py:9-13` says "The HMM is
  stored as a regular field so gradients flow through it," but `ForwardAlgorithm` is stateless and
  receives `hmm_params` as an argument — there is no such field. Update the comment.
- **[P3] `data.py` mixes four concerns** (aggregation, plotting, model pickling, experiment-data
  pickling) in one 197-line module. Split into `data_loading.py`, `persistence.py`, `plotting.py`.
  Also the `df["day"] - 74` magic offset (`data.py:21`) needs a named constant + comment.
- **[P3] `StaticTransitionHigherOrder`**: `-1000.0` sentinel logits (line 36) and
  `int(num_augmented ** (1/order))` (line 47) are fragile — float rounding on the latter can pick the
  wrong K for larger state counts. The `#Todo: Make it more generic` (line 105) is honest; when you
  return to it, prefer an explicit mask over a large-negative sentinel.
- **[P2] Two codebases coexist:** `special_course/` (older procedural drivers) and `src/`. If
  `special_course` is archived, say so in a README and keep it out of the import graph; if it's live,
  it deserves its own review. Ambiguity here is a maintenance tax.
- **[P3] Weak "does it crash" tests.** `test_default_hmm_fit` and `test_forward_algorithm_integration`
  only assert "no exception." Smoke tests are fine as a floor, but adding one real assertion
  (a known likelihood, a shape, a monotonic-improvement check) turns them into actual specs.

---

## 7. Suggested order of attack

A sequence that front-loads the highest learning-to-effort ratio:

1. **[P1] Delete `src/optim/`** (or salvage `profile_likelihood` into the live tree). Removes broken
   imports and the "which one is real?" confusion instantly.
2. **[P1] Make `DynamicTransition` inherit `BaseTransition`** with the correct `step`/`transition_matrix`
   signatures, drop the junk import, fix the docstring. Restores substitutability.
3. **[P1] Fix `HMM.fit`'s nested-loop / optimiser-reset behaviour.** Decide who owns convergence.
4. **[P1] Rename the public typos** (`inital_distribution`, `forward_outout.py`, `intergrations/`)
   while the surface area is small.
5. **[P2] Extract a `ParameterFreezer`** out of `BaseSolver`; inject the inference/loss so the base
   solver stops importing `ForwardAlgorithm`.
6. **[P2] De-duplicate the pseudo-residual loop; fix the misleading `ForwardOutput` docs; fix
   `save_model`.**
7. **[P3] Everything in §5–6** as polish.

Each of steps 1–5 is a concrete exercise in a named design principle (dead-code hygiene, Liskov
substitution, single-level-of-abstraction, naming as interface, single responsibility + dependency
injection). Doing them deliberately — and noticing *which principle* each one is — is how the
system-design intuition gets built.
