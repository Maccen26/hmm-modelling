import jax.numpy as jnp
from src.deprecated.hmm.utils import recursive_filter 




class StaticHMM:
    def __init__(self, transition_matrix: jnp.ndarray, emission_distributions: jnp.ndarray):
        self.transition_matrix = transition_matrix
        self.emission_distributions = emission_distributions
        self.num_states = transition_matrix.shape[0]
        self.num_obs = emission_distributions.shape[1]

        self.ut     = jnp.zeros((self.num_obs, self.num_states))
        self.u_norm = jnp.zeros((self.num_obs, self.num_states))
        self.ft     = jnp.zeros(self.num_obs)

    def forward(self, method: str = "filter"):
        if method == "filter":
            return self._forward_filter()
        else:
            raise ValueError("Invalid method. Must be 'filter'.")

    def _forward_filter(self):
        # --- t=0: initialize ---
        ut0   = self.init_stationary_distribution()         
        g0    = self.emission_distributions[:, 0]        
        ft0   = jnp.sum(ut0 * g0)
        utt0  = ut0 * g0 / ft0

        # --- t=1..T-1: scan ---
        # Transpose so scan iterates over time: (T-1, num_states)
        g_rest = self.emission_distributions[:, 1:].T

        Ut, ft_rest, Utt = recursive_filter(utt0, g_rest, self.transition_matrix)

        # --- Concatenate t=0 with t=1..T-1 ---
        self.ut     = jnp.concatenate([ut0[None, :],  Ut],  axis=0)  # (T, num_states)
        self.u_norm = jnp.concatenate([utt0[None, :], Utt], axis=0)  # (T, num_states)
        self.ft     = jnp.concatenate([ft0[None],     ft_rest], axis=0)  # (T,)

    def init_stationary_distribution(self):
        I = jnp.eye(self.num_states)
        E = jnp.ones((self.num_states, self.num_states))
        e = jnp.ones((self.num_states, 1))
        delta = e.T @ jnp.linalg.inv(I - self.transition_matrix + E)  # (1, num_states)
        return delta.flatten()  # (num_states,)
    






if __name__ == "__main__":
    # Simple 2-state HMM, 5 observations
    transition_matrix = jnp.array([
        [0.7, 0.3],
        [0.4, 0.6]
    ])

    # emission_distributions shape: (num_states, T)
    # Each column is the emission probabilities for all states at time t
    emission_distributions = jnp.array([
        [0.9, 0.2, 0.8, 0.1, 0.7],  # state 0 emissions over time
        [0.1, 0.8, 0.2, 0.9, 0.3],  # state 1 emissions over time
    ])

    hmm = StaticHMM(transition_matrix, emission_distributions)
    hmm.forward(method="filter")

    print("ut     (un-normalized):", hmm.ut)
    print("u_norm (normalized):   ", hmm.u_norm)
    print("ft     (likelihoods):  ", hmm.ft)
    print(emission_distributions)

    # Sanity checks
    print("\nSanity checks:")
    print("u_norm rows sum to 1:", jnp.sum(hmm.u_norm, axis=1))   # should all be ~1.0
    print("ft all positive:     ", jnp.all(hmm.ft > 0))