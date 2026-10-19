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


def pad_sequence_batch(seqs, lengths: list[int] | None = None) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pad a ragged list of per-sequence arrays into one rectangular batch.

    `jax.vmap` maps over an axis of a rectangular array, so sequences of different
    lengths have to be padded to a common length and the padding then excluded from
    the likelihood. This returns the padded batch and the boolean mask that marks
    which steps are real:

        padded: (B, max_T, ...)   mask: (B, max_T), True where the step is observed

    Padding repeats each sequence's **last row** rather than inserting zeros or NaNs.
    The padded steps are masked out of the likelihood, but their densities are still
    computed, and `jnp.where` propagates a NaN from its unselected branch straight
    into the gradient -- so the padding values have to stay in a range the emission
    can evaluate. Repeating the last row also keeps an autoregressive emission's lags
    finite. Padding always goes at the end, so the lags at every *real* step still
    read only real observations.

    :param seqs: per-sequence arrays, each with time on axis 0 and matching
        trailing dimensions.
    :param lengths: expected sequence lengths; if given, they are checked against
        `seqs` so a mismatched `xs`/`ts` batch is caught here rather than producing a
        silently misaligned mask.
    """
    arrays = [jnp.asarray(s) for s in seqs]
    if not arrays:
        raise ValueError("cannot pad an empty batch of sequences")

    got = [a.shape[0] for a in arrays]
    if min(got) == 0:
        raise ValueError(f"every sequence must have at least one observation, got lengths {got}")
    if lengths is not None and got != list(lengths):
        raise ValueError(
            f"sequence lengths must match across ys, ts and xs, got {got} where "
            f"{list(lengths)} was expected"
        )

    trailing = {a.shape[1:] for a in arrays}
    if len(trailing) > 1:
        raise ValueError(
            f"every sequence must share the same trailing shape, got {sorted(trailing)}"
        )

    max_len = max(got)
    padded = [
        a if a.shape[0] == max_len
        else jnp.concatenate([a, jnp.repeat(a[-1:], max_len - a.shape[0], axis=0)], axis=0)
        for a in arrays
    ]
    mask = jnp.arange(max_len)[None, :] < jnp.asarray(got)[:, None]
    return jnp.stack(padded), mask
