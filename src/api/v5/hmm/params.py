import jax.numpy as jnp
import equinox as eqx
from src.api.v5.hmm.transitions.base_transition import BaseTransition
from src.api.v5.hmm.emissions.base_emission import BaseEmission

class Params(eqx.Module):
    """
    HMM class that combines a transition model and an emission model. 
    Holds trainable parameters for both the transition and emission models.
    """

    transition: BaseTransition
    emission: BaseEmission

    def transition_matrix(self, xt: jnp.ndarray | None = None) -> jnp.ndarray:
        """
        Builds the transition matrix for the observation at index `t`.

        :param t: observation index, used for covariate lookup.
        :param dt: waiting time since the previous observation, used by the
            continuous-time transitions.

        :return: transition matrix of dim (num_states, num_states)
        """
        return self.transition.transition_matrix(xt=xt)

    def transition_matrices(self, N: int, xs: jnp.ndarray | None = None) -> jnp.ndarray:
        """
        Builds one transition matrix per observation in a single batched call.
        Returns an array of shape (T, num_states, num_states).
        """
        return self.transition.transition_matrices(N = N, xs = xs)


    def density(self, t:int, ys: jnp.ndarray):
        """
        y is the observation at time step t.
        x is the covariates at time step t.
        ts is the optional per-observation waiting-time sequence, forwarded to the
        emission (used by continuous-time emissions; ignored by the others).
        Returns the emission density p(y_t | z_t, x_t) at time step t with dimensions (num_states,).
        """
        return self.emission.density(t = t, ys = ys)

    def densities(self, ys: jnp.ndarray):
        """
        Emission densities for every observation in a single batched call.

        :param indices: observation indices, i.e. jnp.arange(T).
        :param ts: per-observation waiting times, forwarded to the emission.
        Returns an array whose leading axis is T.
        """
        indices = jnp.arange(len(ys)) 
        return self.emission.densities(indices = indices, ys = ys)

    def cdf(self, t:int, ys: jnp.ndarray):
        """
        y is the observation at time step t.
        x is the covariates at time step t.
        ts is the optional per-observation waiting-time sequence, forwarded to the emission.
        Returns the emission cdf P(Y_t <= y | z_t, x_t) at time step t with dimensions (num_states,).
        """
        return self.emission.cdf(t = t, ys = ys)
    
    def cdfs(self, ys: jnp.ndarray):
        """
        Emission CDFs for every observation in a single batched call.

        :param indices: observation indices, i.e. jnp.arange(T).
        :param ts: per-observation waiting times, forwarded to the emission.
        Returns an array whose leading axis is T.
        """
        indices = jnp.arange(len(ys))
        return self.emission.cdfs(indices=indices, ys=ys)

    def __iter__(self):
        """Make the class iterable with names. This is useful for the forward and backward algorithms, where we need to iterate over the states and compute the transition and emission probabilities."""
        yield 'transition', self.transition
        yield 'emission', self.emission 

    def __len__(self):
        """Returns the number of trainable parameters in the model."""
        #num_transition_params = self.transition.transition_logits.size
        # Skip static/non-array fields (e.g. the higher-order transition's `order` int),
        # which are not trainable parameters and have no `.size`.
        num_transition_params = sum(param.size for _, param in self.transition.__iter__() if hasattr(param, "size"))
        num_emission_params = sum(param.size for _, param in self.emission.__iter__() if hasattr(param, "size"))
        return num_transition_params + num_emission_params


    


        
    
    