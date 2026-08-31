import jax.numpy as jnp

def logits_to_transition_matrix(logits: jnp.ndarray) -> jnp.ndarray:
    """
    Inverse of transition_matrix_to_logits.
    
    Places exp(pars) in off-diagonal positions of an identity matrix
    (diagonal = 1 = exp(0), the reference), then row-normalizes.
    This is just softmax per row.
    """
    m = logits.shape[0]
    exp_pars = jnp.exp(logits.flatten())
    Gamma = jnp.eye(m)
    rows, cols = jnp.where(~jnp.eye(m, dtype=bool), size=m * (m - 1))
    Gamma = Gamma.at[rows, cols].set(exp_pars)
    return Gamma / Gamma.sum(axis=1, keepdims=True)

def transition_matrix_to_logits(Gamma: jnp.ndarray) -> jnp.ndarray:
    """
    Direct translation of the R Markov.link function.
    
    Maps a transition matrix to unconstrained logit parameters
    by computing log(gamma_ij / gamma_ii) for off-diagonal entries.
    """
    m = Gamma.shape[0]
    # Zero diagonal — rowSums then gives 1 - gamma_ii
    Gamma = Gamma.at[jnp.diag_indices(m)].set(0.0)
    # 1 - rowSums recovers the original diagonal
    diag_vals = 1.0 - Gamma.sum(axis=1)
    # log-ratio: each entry divided by its row's diagonal value
    beta = jnp.log(Gamma / diag_vals[:, None])
    # Extract off-diagonal (R does transpose then extract —
    # transpose changes extraction order to column-major)
    mask = ~jnp.eye(m, dtype=bool)
    return beta[mask].reshape(m, m - 1)