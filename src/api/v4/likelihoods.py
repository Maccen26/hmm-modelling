from src.base.base_hmm import BaseHMM 
from src.base.base_output import BaseOutput
import jax.numpy as jnp

def negative_log_likelihood(output: BaseOutput, hmm_params: BaseHMM) -> jnp.ndarray:
    """
    Compute the negative log-likelihood of the observed sequence given the model using the forward algorithm.
    """
    total_log_likelihood = output.log_likelihood()
    return -total_log_likelihood 


