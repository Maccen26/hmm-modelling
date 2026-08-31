import equinox as eqx
from abc import abstractmethod, ABC
from typing import Any
import jax.numpy as jnp
import jax
from src.base.base_hmm import BaseHMM
class BaseInference(ABC):
    """
    Base class for inference algorithms for HMMs.
    
    Subclasses implement `step` (single iteration) and `run` (full sequence).
    The HMM is stored as a regular field so gradients flow through it.
    """



    @abstractmethod
    def step(self, hmm_params:Any, carry: Any, t: int, ys: jnp.ndarray, xs: jnp.ndarray | None = None) -> Any:
        """
        Single iteration of the algorithm.
        
        Args:
            hmm_params: The HMM parameters
            carry: Algorithm-specific state from previous step
            t: Current time index
            ys: Full observation sequence (indexed by t inside)
            xs: Optional full covariate sequence
            
        Returns:
            (new_carry, output) tuple compatible with jax.lax.scan
        """
        ...



    def run(self, hmm_params:Any, carry_pre: Any, ys: jnp.ndarray, xs: jnp.ndarray | None = None) -> Any:
        """
        Run the full algorithm over a sequence using jax.lax.scan.
        """

        self._validate_inputs(hmm_params,ys, xs, carry_pre)

        def scan_fn(carry, t):
            return self.step(hmm_params, carry, t, ys, xs)

        carry_final, outputs = jax.lax.scan(scan_fn, carry_pre, jnp.arange(0, len(ys)))
        return self.postprocess(carry_pre, carry_final, outputs)
    

    def _validate_inputs(self, hmm_params:Any, ys: jnp.ndarray, xs: jnp.ndarray | None, carry_pre: Any):
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
    
    @abstractmethod
    def postprocess(self, carry_0, carry_final, outputs) -> Any:
        """
        Method to post-process scan outputs (e.g., compute log-likelihood from final carry).
        """
        ... 