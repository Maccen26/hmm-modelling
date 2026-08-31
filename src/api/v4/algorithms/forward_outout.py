from src.base.base_output import BaseOutput 
import jax.numpy as jnp


class ForwardOutput(BaseOutput):
    """
    Output class for the forward algorithm. This class is meant to store the output of the forward algorithm and compute the likelihood and negative log-likelihood of the observed sequence given the model. 
    """
    ft: jnp.ndarray  # Forward probabilities at each time step, shape (T, num_states) 
    utt: jnp.ndarray # Observed sequence, shape (T,) 
    ut: jnp.ndarray  # Filtered state probabilities at each time step, shape (T, num_states)


    def log_likelihood(self) -> jnp.ndarray:
        """
        Compute the log-likelihood of the observed sequence given the model using the forward algorithm.
        """
        total_log_likelihood = jnp.sum(jnp.log(self.ft))
        return total_log_likelihood 