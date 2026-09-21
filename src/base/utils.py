import jax.numpy as jnp
import jax

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


def logits_to_transition_matrix_continuous(logits: jnp.ndarray, t : int) -> jnp.ndarray:
    """
    Createas a continuous transition matrix from the off-diagonal logits by

    T = exp(Q * t) where Q is the transition rate matrix and t is the waiting time. 
    The diagonal of Q is set such that the rows sum to 0
    """
    Q = get_Q_from_logits(logits)
    return jax.scipy.linalg.expm(Q * t)

def get_Q_from_logits(logits: jnp.ndarray) -> jnp.ndarray:
    """
    Createas a continuous transition matrix from the off-diagonal logits by

    T = exp(Q * t) where Q is the transition rate matrix and t is the waiting time. 
    The diagonal of Q is set such that the rows sum to 0
    """
    m = logits.shape[0]
    Q = jnp.zeros((m, m)) 
    rows, cols = jnp.where(~jnp.eye(m, dtype=bool), size=m * (m - 1))
    logits = jnp.exp(logits.flatten())
    Q = Q.at[rows, cols].set(logits) 

    row_sums = jnp.sum(Q, axis=1)
    Q = Q.at[jnp.arange(m), jnp.arange(m)].set(-row_sums)

    return Q


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


def transtion_matrix_to_logits_continuous(Gamma: jnp.ndarray, t: int) -> jnp.ndarray:
    """
    Inverse of `get_Q_from_logits`: maps a continuous transition matrix to the
    unconstrained logits.

    Since the generator's off-diagonal rates are parameterized as
    q_ij = exp(logit_ij) (enforcing non-negativity), the inverse takes the log of
    the off-diagonal entries of Q = logm(Gamma). The matrix log of a stochastic
    matrix may have slightly-negative off-diagonals (the CTMC embeddability issue),
    so we clamp to a tiny positive value before the log.
    """
    eigvals, eigvecs = jnp.linalg.eig(Gamma)
    Q = (eigvecs @ jnp.diag(jnp.log(eigvals)) @ jnp.linalg.inv(eigvecs)).real
    m = Q.shape[0]
    mask = ~jnp.eye(m, dtype=bool)
    off_diag = jnp.clip(Q[mask], a_min=1e-8)
    return jnp.log(off_diag).reshape(m, m - 1)

