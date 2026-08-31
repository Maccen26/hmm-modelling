import jax
import jax.numpy as jnp
import jaxopt
import equinox as eqx

from src.deprecated.base.hmm import HMM
from src.optim.base import BaseOptimizer


class Minimizer(BaseOptimizer):
    def __init__(self, model: HMM, loss_fn, spec=None):
        super().__init__(model, loss_fn, spec)
        self.optimizer = jaxopt.ScipyMinimize(fun=self._loss_wrapper, maxiter=1000)

    def _loss_wrapper(self, trainaled_parameters, y : jnp.ndarray, x: jnp.ndarray| None =None):
        # Reconstruct the full model with the current parameters
        full_model = eqx.combine(trainaled_parameters, self.frozen_parameters)
        return self.loss_fn(full_model, y, x)

    def run(self, y, x=None):
        # Run the optimizer
        result = self.optimizer.run(self.trainaled_parameters, y=y, x=x)

        # Update the model with the optimized parameters
        self.trainaled_parameters = result.params 
        self.model = eqx.combine(self.trainaled_parameters, self.frozen_parameters)
        return self.model
    