from abc import abstractmethod, ABC
from typing import Any
import jax.numpy as jnp
from src.api.v5.hmm.params import Params


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
    def run(self,
            params: Params, 
            initial_dist: jnp.ndarray, 
            ys: jnp.ndarray, 
            xs: jnp.ndarray | None = None) -> Any:
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
        :param mask: for a padded ragged batch, a (B, T) boolean array that is True
            where a step is a real observation. Masked steps must contribute nothing
            to the likelihood.
        """
        ...

  
