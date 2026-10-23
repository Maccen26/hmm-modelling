import jax.numpy as jnp
import jax 
from functools import partial

class BackwardSampler: 

    @classmethod
    def step(cls, key: jax.Array, Gammas: jax.Array, utt: jax.Array) -> jax.Array:
        """
        Returns 1 path via backward recursion  over a precomputed transition matrix and filtered state distribution.

        :param key: A JAX PRNG key used for sampling the path

        :param Gammas: An Array of transition matrixes of shape (T, num_states, num_states) for each time step t

        :param utt: An Array of filtered state distributions to time t of shape (T, num_states) for each time step t

        :return: A path of states of shape (T,) for each time step t, where each state is an integer in [0, num_states-1]
        """
        T, num_states = utt.shape
        key, subkey = jax.random.split(key)

        # Sample the last state from the filtered distribution at time T
        z_last = jax.random.choice(subkey, num_states, p=utt[T - 1])
        path = jnp.zeros(T, dtype=jnp.int64).at[T - 1].set(z_last)

        def body(i, carry):
            path, key = carry
            t = T - 2 - i  # i = 0, 1, ..., T-2  maps to  t = T-2, ..., 0

            key, subkey = jax.random.split(key)
            z_next = path[t + 1]
            probs = utt[t] * Gammas[t + 1, :, z_next]  # unnormalized
            probs = probs / jnp.sum(probs)
            z_t = jax.random.choice(subkey, num_states, p=probs)

            path = path.at[t].set(z_t)
            return path, key

        path, _ = jax.lax.fori_loop(0, T - 1, body, (path, key))
        return path
    
    @classmethod
    @partial(jax.jit, static_argnames=("cls", "num_paths"))
    def run_algorithm(cls, num_paths: int, Gammas: jax.Array, utt: jax.Array, seed: int) -> jax.Array:
        """
        Returns num_paths paths via backward recursion  over a precomputed transition matrix and filtered state distribution.

        :param num_paths: Number of paths to sample

        :param Gammas: An Array of transition matrixes of shape (T, num_states, num_states) for each time step t

        :param utt: An Array of filtered state distributions to time t of shape (T, num_states) for each time step t

        :param seed: Random seed used to generate one PRNG key per path

        :return: An Array of shape (num_paths, T) containing the sampled paths
        """
        keys = jax.random.split(jax.random.key(seed), num_paths)
        paths = jax.vmap(cls.step, in_axes=(0, None, None))(keys, Gammas, utt)
        return paths  
    
    @classmethod
    def run(cls, num_paths: int, Gammas: jax.Array, utt: jax.Array, seed: int = 0) -> jax.Array:
        """
        We validate args and then call run_algorithm to sample num_paths paths via backward recursion over a precomputed transition matrix and filtered state distribution.
        """
        if not isinstance(num_paths, int) or num_paths <= 0:
            raise ValueError("num_paths must be a positive integer.")
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        if not isinstance(Gammas, jax.Array) or Gammas.ndim != 3 or Gammas.shape[1] != Gammas.shape[2]:
            raise ValueError("Gammas must be a 3D JAX array of shape (T, num_states, num_states). Had shape: {}".format(Gammas.shape))
        if not isinstance(utt, jax.Array) or utt.ndim > 3: 
            raise ValueError("utt must be a JAX array and can be 2D or 3D of shape (T, num_states) or (T, 1, num_states).")
    
        if Gammas.shape[0] != utt.shape[0]:
            raise ValueError("The first dimension of Gammas and utt must match (number of time steps T).")
        
        if (utt.ndim == 3): 
            if utt.shape[1] == 1:
                utt = utt[:, 0, :]  # Reshape to (T, num_states)
            else:
                raise ValueError("If utt is 3D, its second dimension must be 1 (shape (T, 1, num_states)).")
        
        return cls.run_algorithm(num_paths=num_paths, Gammas=Gammas, utt=utt, seed=seed)