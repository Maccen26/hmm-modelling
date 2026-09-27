import equinox as eqx
from abc import abstractmethod, ABC
from typing import Any
import jax.numpy as jnp
import jax
from jaxtyping import Int, Array
from src.base.base_hmm import BaseHMM


class BaseInference(ABC):
    """
    Base class for inference algorithms for HMMs.

    Subclasses implement `run` (which precomputes whatever the whole sequence needs
    and drives a `jax.lax.scan`) and `step` (a single scan iteration).

    The split is deliberate: everything model-specific — transition matrices,
    emission densities — is computed up front in `run`, so `step` is left with pure
    linear algebra over already-materialised arrays.
    """

    @abstractmethod
    def step(self, carry: Any, step_input: Any) -> Any:
        """
        Single iteration of the algorithm, in `jax.lax.scan` form.

        Args:
            carry: Algorithm-specific state from the previous step
            step_input: The precomputed quantities for this step, sliced from the
                arrays `run` scans over (e.g. a transition matrix and a density)

        Returns:
            (new_carry, output) tuple compatible with jax.lax.scan
        """
        ...

    @abstractmethod
    def run(self, hmm_params: Any, carry_pre: Any, ys: jnp.ndarray, ts: Int[Array, " n"] | None = None, xs: jnp.ndarray | None = None, batched: bool = False) -> Any:
        """
        Run the full algorithm over a sequence, or over a batch of sequences.

        Implementations precompute the per-step inputs in batched form, scan `step`
        over them, and hand the result to `postprocess`.

        :param batched: when True, the leading axis of `ys` (and of `ts`/`xs` when
            given) is a batch of independent sequences and every output field
            carries that axis in front. Batching is always opt-in: it is never
            inferred from the shape of `ys`, because a single sequence may be
            spelled (T,) or (T, 1) and the latter is indistinguishable from a batch
            of T length-1 sequences.
        """
        ...

    def _validate_common(self, hmm_params: Any, ys: jnp.ndarray, ts: Int[Array, " n"] | None, xs: jnp.ndarray | None, carry_pre: Any):
        """The type checks that hold whether or not the inputs are batched."""
        if carry_pre is None:
            raise ValueError("carry_pre cannot be None. Use the initialize() method to compute the initial carry.")
        if not isinstance(ys, jnp.ndarray):
            raise ValueError(f"ys must be a jnp.ndarray, got {type(ys)}")
        if xs is not None and not isinstance(xs, jnp.ndarray):
            raise ValueError(f"xs must be a jnp.ndarray if provided, got {type(xs)}")
        if len(ys) == 0:
            raise ValueError("ys cannot be empty")
        if not isinstance(hmm_params, BaseHMM):
            raise ValueError(f"hmm_params must be an instance of HMMParams, got {type(hmm_params)}")
        if (ts is not None and not isinstance(ts, jnp.ndarray)):
            raise ValueError(f"ts must be a jnp.ndarray of floats if provided, got {type(ts)}")

    def _validate_inputs(self, hmm_params:Any, ys: jnp.ndarray, ts: Int[Array, " n"] | None, xs: jnp.ndarray | None, carry_pre: Any):
        """Validate that the inputs to run() have compatible shapes and types.

        This is the single-sequence path: `ys` is one sequence, so its leading axis
        is time and `xs`/`ts` must line up with it.
        """
        self._validate_common(hmm_params, ys, ts, xs, carry_pre)
        if xs is not None and len(xs) != len(ys):
            raise ValueError(f"xs and ys must have the same length, got {len(xs)} and {len(ys)}")
        if ts is not None and len(ts) != len(ys):
            raise ValueError(f"ts and ys must have the same length, got {len(ts)} and {len(ys)}") 

    def _validate_batched_inputs(self, hmm_params: Any, ys: jnp.ndarray, ts: Int[Array, " n"] | None, xs: jnp.ndarray | None, carry_pre: Any):
        """Validate the batched path, where ys is (B, T, ...).

        Both the batch size and the sequence length are checked against `ts`/`xs`.
        The single-sequence checks compare only leading-axis lengths, which for
        batched input would compare batch sizes and never notice a per-sequence
        length mismatch.
        """
        self._validate_common(hmm_params, ys, ts, xs, carry_pre)
        if ys.ndim < 2:
            raise ValueError(
                f"batched=True expects ys of shape (B, T, ...), got a {ys.ndim}-D array "
                f"of shape {ys.shape}. Pass batched=False for a single sequence."
            )
        batch_shape = ys.shape[:2]
        if ts is not None and ts.shape[:2] != batch_shape:
            raise ValueError(
                f"ts must match ys on the batch and time axes, got ts shape {ts.shape} "
                f"and ys shape {ys.shape}"
            )
        if xs is not None and xs.shape[:2] != batch_shape:
            raise ValueError(
                f"xs must match ys on the batch and time axes, got xs shape {xs.shape} "
                f"and ys shape {ys.shape}"
            )
    
    @abstractmethod
    def postprocess(self, carry_0, carry_final, outputs) -> Any:
        """
        Method to post-process scan outputs (e.g., compute log-likelihood from final carry).
        """
        ...
