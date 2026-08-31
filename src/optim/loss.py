import equinox as eqx
import numpy as np
import jax.numpy as jnp
from src.deprecated.base.hmm import HMM
from typing import Callable 
from src.optim.minimizer import Minimizer
from src.optim.lbfgs import LFBGSOptimizer


def negative_log_likelihood(model: HMM, y: jnp.ndarray, x: jnp.ndarray | None = None):
    _, ft, _ = model.forward(y, x)
    log_likelihood = jnp.sum(jnp.log(ft))
    return -log_likelihood 

def negative_log_likelihood_v2(model: HMM, y: jnp.ndarray, x: jnp.ndarray | None = None):
    return -model.log_likelihood(y, x) 


def profile_likelihood(model: HMM, y: jnp.ndarray, phi_range: tuple, loss_generator: Callable, num_points: int = 20, X: jnp.ndarray | None = None):
    param_space = jnp.linspace(phi_range[0], phi_range[1], num=num_points)
    log_likes = np.zeros(num_points)

    for i in range(num_points):
        phi0 = float(param_space[i])
        loss_fn = loss_generator(phi0)

        optimizer = LFBGSOptimizer(model=model, loss_fn=loss_fn)
        opt_model = optimizer.run(y, X)

        #log_likes.append(float(-loss_fn(opt_model, y, X)))
        log_likes[i] = float(-loss_fn(opt_model, y, X))


    return param_space, jnp.array(log_likes)


#def profile_likelihood(model: HMM, optimizer:  y : jnp.ndarray, phi_range: tuple,loss_fn : Callable,  num_points: int = 20, X : jnp.ndarray | None = None):
#    phi_space = jnp.linspace(phi_range[0], phi_range[1], num=num_points)
#    log_likes = []
#
#    for i in range(num_points):
#        phi0 = float(phi_space[i])
#        loss_fn = loss_fn(phi0)
#
#        optimizer = Minimizer(loss_fn=loss_fn, model=model)
#        opt_model = optimizer.run(y, X)
#
#        log_likes.append(float(-loss_fn(opt_model, y, X)))
#        print(f"phi0={phi0:.3f}  log-lik={log_likes[-1]:.2f}")
#
#    return phi_space, jnp.array(log_likes)
#
#