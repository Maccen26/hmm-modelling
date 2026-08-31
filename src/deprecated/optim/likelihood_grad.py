from src.deprecated.hmm.models.stationary_hmm import StationaryHMM 

import jax.numpy as jnp
import jaxopt 
import jax 


@jax.jit
def likelihood_grad(model, y):

    def loss(model, y):
        return -model.log_likelihood(y)

    solver = jaxopt.LBFGS(fun=loss, maxiter=100)
    (opt_model, state) = solver.run(model, y)
    return opt_model, state


@jax.jit
def likelihood_grad_dynamic(model, y, x):

    def loss(model, y, x):
        return -model.log_likelihood(y, x)

    solver = jaxopt.LBFGS(fun=loss, maxiter=100)
    (opt_model, state) = solver.run(model, y, x)
    return opt_model, state
