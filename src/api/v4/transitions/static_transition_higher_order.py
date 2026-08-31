import jax
from src.base import BaseTransition 
import jax.numpy as jnp 
import equinox as eqx


def _make_transition_logits(logits, order):
    """
    Constructs the full (K^order, K^order) transition logit matrix for a higher-order HMM.
    """
    num_augmented = logits.shape[0]   # K^order
    K = logits.shape[1] + 1          # base number of states

    def decode(i):
        """Decode augmented state index i into a tuple of base states (oldest first)."""
        result = []
        for _ in range(order):
            result.append(i % K)
            i //= K
        return tuple(reversed(result))

    state_tuples = [decode(i) for i in range(num_augmented)]

    given_rows, given_cols = [], []
    zero_rows, zero_cols = [], []

    for i, tup in enumerate(state_tuples):
        suffix = tup[1:]  # last (order-1) base states
        valid_next = sorted(j for j, t in enumerate(state_tuples) if t[:-1] == suffix)
        for k, j in enumerate(valid_next[:-1]):
            given_rows.append(i)
            given_cols.append(j)
        zero_rows.append(i)
        zero_cols.append(valid_next[-1])

    full_logits = jnp.full((num_augmented, num_augmented), -1000.0)
    full_logits = full_logits.at[jnp.array(given_rows), jnp.array(given_cols)].set(logits.flatten())
    full_logits = full_logits.at[jnp.array(zero_rows), jnp.array(zero_cols)].set(0.0)
    return full_logits

def decode_possible_transitions(Gamma, order = 2):
    """
    Given a transition matrix Gamma of shape (K^order, K^order), decode the possible transitions between augmented states.
    Returns a list of tuples (from_state_tuple, to_state_tuple) for valid transitions.
    """
    num_augmented = Gamma.shape[0]
    K = int(num_augmented ** (1/order))

    def decode(i):
        result = []
        for _ in range(order):
            result.append(i % K)
            i //= K
        return tuple(reversed(result))

    state_tuples = [decode(i) for i in range(num_augmented)]

    transitions = {} 

    for i in range(num_augmented):
        state_list = []
        for j in range(num_augmented):
            if not jnp.isclose(Gamma[i, j], 0.0):  # Assuming -1000 indicates impossible transition
                state_list.append((state_tuples[j], Gamma[i, j]))

        transitions[state_tuples[i]] = state_list
    return transitions




def logits_to_transition_matrix_higher_order(logits):
    """
    Expects a 2d build matrix with 0 entries
    """
    Gamma = jnp.exp(logits)
    return Gamma / Gamma.sum(axis=1, keepdims=True)  # Normalize rows to sum to 1


class StaticTransitionHigherOrder(BaseTransition):
    """
    Static transition model for an HMM. The transition matrix does not depend on the covariates at time step t. 

    transition_matrix_: jnp.ndarray is of dim (num_states, num_states - 1) and contains the off-diagonal elements of the transition matrix. 
    """
    order: int = eqx.field(static=True, default=2)  # Order of the Markov chain, static field since it doesn't change during training

    def __init__(self, transition_logits, order=2, num_states = 2):
        super().__init__(transition_logits)
        self.order = order 
        self._validate_transition_logits(transition_logits.shape[1] + 1)  # Validate the shape of transition_logits based on the order

    def _validate_transition_logits(self, num_states):
        """
        Should validate that the number of possible transitions match the number of possible transition logits. 

        """
        expected_num_logits = num_states**self.order * (num_states - 1)  # K^order augmented states, each with K-1 free logits
        actual_num_logits = self.transition_logits.size
        if actual_num_logits != expected_num_logits:
            raise ValueError(f"Expected {expected_num_logits} transition logits for order {self.order} and {num_states} states, but got {actual_num_logits}.")
        

    #Todo: Make it more geneeric. Right now we only implement for lags of 2 or 3 states.  

    def step(self, t: int | None, ys: jnp.ndarray | None, xs: jnp.ndarray | None = None) -> jnp.ndarray:
        """
        computes new transitiongs logits based on the order of the markov chain. 
        The goal is to make non possible transition have almost 0 probability. 

        
        :param self: Description
        :param xt: Description
        :return: Description
        :rtype: ndarray
        """

        return _make_transition_logits(self.transition_logits, self.order)
    
    def transition_matrix(self, t:int| None = None, ys: jnp.ndarray | None = None, xs: jnp.ndarray | None = None) -> jnp.ndarray: 
        """
        Builds the transition matrix at time step t given the covariates at time step t.
        
        :param xt: covarites at time step t. 

        :return: transition matrix at time step t of dim (num_states, num_states) 
        """
        logits = self.step(t, ys, xs)


        return logits_to_transition_matrix_higher_order(logits)
    

