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
    def run(self, hmm_params: Any, carry_pre: Any, ys: jnp.ndarray, ts: Int[Array, " n"] | None = None, xs: jnp.ndarray | None = None) -> Any:
        """
        Run the full algorithm over a sequence.

        Implementations precompute the per-step inputs in batched form, scan `step`
        over them, and hand the result to `postprocess`.
        """
        ...

    def _validate_inputs(self, hmm_params:Any, ys: jnp.ndarray, ts: Int[Array, " n"] | None, xs: jnp.ndarray | None, carry_pre: Any):
        """Validate that the inputs to run() have compatible shapes and types.
        """
        if carry_pre is None:
            raise ValueError("carry_pre cannot be None. Use the initialize() method to compute the initial carry.")
        if not isinstance(ys, jnp.ndarray):
            raise ValueError(f"ys must be a jnp.ndarray, got {type(ys)}")
        if xs is not None and not isinstance(xs, jnp.ndarray):
            raise ValueError(f"xs must be a jnp.ndarray if provided, got {type(xs)}")
        if len(ys) == 0:
            raise ValueError("ys cannot be empty")
        if xs is not None and len(xs) != len(ys):
            raise ValueError(f"xs and ys must have the same length, got {len(xs)} and {len(ys)}")
        if not isinstance(hmm_params, BaseHMM):
            raise ValueError(f"hmm_params must be an instance of HMMParams, got {type(hmm_params)}")
        if ts is not None and len(ts) != len(ys):
            raise ValueError(f"ts and ys must have the same length, got {len(ts)} and {len(ys)}") 
        if (ts is not None and not isinstance(ts, jnp.ndarray)):
            raise ValueError(f"ts must be a jnp.ndarray of floats if provided, got {type(ts)}")
    
    @abstractmethod
    def postprocess(self, carry_0, carry_final, outputs) -> Any:
        """
        Method to post-process scan outputs (e.g., compute log-likelihood from final carry).
        """
        ...
