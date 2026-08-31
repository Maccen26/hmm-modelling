
import jax
from jax.scipy.stats import norm 

@jax.jit
def _gaussian_pdf(x, mean, std):
    """
    x    : (T,) vector of observations
    mean : (num_states,) vector
    std  : (num_states,) vector
    returns: (num_states, T) matrix
    """
    return norm.pdf(x[:, None], loc=mean, scale=std)
