import jax
jax.config.update("jax_enable_x64", True)

from typing import Callable
import jax.numpy as jnp
import jax.random as random

from src.api.v4 import (
    HMM,
    StaticTransition,
    GaussEmission,
    AutoregressiveGaussEmission,
    StaticTransitionHigherOrder,
)
from src.api.v4.transitions.dynamic_transition import DynamicTransition
from drivers.utils import load_train_data, load_test_data, save_model, load_model


def fit_model(model_name: str,
            data_name: str,
            tag: str,
            init_model: Callable,
            frozen_param: dict|None = None):

    print(f"Starting {model_name} model run...")
    ys, Xs = load_train_data(data_name=data_name, tag=tag)
    if (len(ys) != Xs.shape[0]):
        raise ValueError(f"Mismatch in number of samples: ys has length {len(ys)}, but Xs has shape {Xs.shape}.")

    model = init_model(model_name=model_name, ys=ys, Xs=Xs, data_name=data_name, tag=tag)
    model.fit(ys=ys, xs=Xs, frozen=frozen_param)
    path = f"results/models/{data_name}/{tag}/{model_name}.pkl"
    save_model(model=model, save_path=path)


def _base_model_path(data_name: str, tag: str, model_name: str) -> str:
    """Path to a previously fitted model used to seed the current one."""
    return f"results/models/{data_name}/{tag}/{model_name}.pkl"


def init_ordinary_hmm(model_name: str, ys: jnp.ndarray, Xs: jnp.ndarray, data_name: str, tag: str) -> HMM:
    # Initiating parameters
    mu0 = 400
    q = jnp.quantile(ys, jnp.array([0.40, 0.60, 0.80]))
    mu = jnp.array([mu0, q[0], q[1], q[2]])
    std = jnp.std(ys) * jnp.ones_like(mu)
    transition_matrix = jnp.array([[0.7, 0.1, 0.1, 0.1],
                                   [0.1, 0.7, 0.1, 0.1],
                                   [0.1, 0.1, 0.7, 0.1],
                                   [0.1, 0.1, 0.1, 0.7]])
    # Initiating the HMM model
    transition = StaticTransition.from_params(transition_matrix)
    emission = GaussEmission.from_params(mu=mu, sigma=std)
    model = HMM(transition=transition, emission=emission)
    return model


def init_ar_hmm(model_name: str, ys: jnp.ndarray, Xs: jnp.ndarray, data_name: str, tag: str) -> HMM:
    # Initiating parameters from the fitted ordinary HMM
    ordinary_model = load_model(_base_model_path(data_name, tag, "ordinary_hmm"))
    mu = ordinary_model.emission.mu(0, 0)
    std = ordinary_model.emission.sigma(0, 0)
    transition_matrix = ordinary_model.transition.transition_matrix()
    phi_vals = jnp.array([0.20, 0.20, 0.20, 0.20])

    # Initiating the HMM model
    transition = StaticTransition.from_params(transition_matrix)
    emission = AutoregressiveGaussEmission.from_params(mu=mu, sigma=std, phi=phi_vals)
    model = HMM(transition=transition, emission=emission)
    return model


def init_ar_2_hmm(model_name: str, ys: jnp.ndarray, Xs: jnp.ndarray, data_name: str, tag: str) -> HMM:
    # Initiating parameters from the fitted AR(1) HMM
    ar_model = load_model(_base_model_path(data_name, tag, "ar_hmm"))
    mu = ar_model.emission.mu_vals(0, 0)
    std = ar_model.emission.sigma(0, 0)
    transition_matrix = ar_model.transition.transition_matrix()
    phi_vals = ar_model.emission.phi()
    phi_vals = jnp.concatenate([phi_vals, jnp.array([[0.20, 0.20, 0.20, 0.20]])])

    # Initiating the HMM model
    transition = StaticTransition.from_params(transition_matrix)
    emission = AutoregressiveGaussEmission.from_params(mu=mu, sigma=std, phi=phi_vals)
    model = HMM(transition=transition, emission=emission)
    return model


def init_covariate_hmm(model_name: str, ys: jnp.ndarray, Xs: jnp.ndarray, data_name: str, tag: str) -> HMM:
    # Seed the dynamic transition from the fitted base (ordinary) HMM.
    base = load_model(_base_model_path(data_name, tag, "ordinary_hmm"))
    transition_logits = base.transition.transition_logits
    initial_distribution = base._compute_stationary_distribution()
    emission = base.emission

    # beta has shape (num_covariates,) + transition_logits.shape, e.g. (2, 4, 3).
    num_covariates = Xs.shape[1]
    beta = random.uniform(
        random.PRNGKey(0),
        shape=(num_covariates,) + transition_logits.shape,
        minval=-1.0,
        maxval=1.0,
    )
    transition = DynamicTransition(transition_logits=transition_logits, beta=beta)

    # DynamicTransition.transition_matrix needs (t, xs), so the HMM can't compute the
    # stationary distribution itself -> seed it from the base model.
    model = HMM(transition=transition, emission=emission,
                inital_distribution=initial_distribution)
    return model


def init_second_order_hmm(model_name: str, ys: jnp.ndarray, Xs: jnp.ndarray, data_name: str, tag: str) -> HMM:
    # Initiating parameters from the fitted AR(1) HMM
    ar_model = load_model(_base_model_path(data_name, tag, "ar_hmm"))
    params = ar_model.params
    transition_logits = list(params.transition.transition_logits)

    transition = StaticTransitionHigherOrder(jnp.asarray(transition_logits * 4), order=2)

    mu_init = jnp.asarray(params.emission.mu(0, ys))  # length 4, tied across s_{t-1}
    sigma_init = jnp.sort(jnp.repeat(params.emission.sigma(0, ys), 4))
    phi_vals = jnp.repeat(params.emission.phi(), 4, axis=1)
    emission = AutoregressiveGaussEmission.from_params(mu=mu_init, sigma=sigma_init, phi=phi_vals)

    model = HMM(transition=transition, emission=emission)
    return model


def init_ar_1_covariate_hmm(model_name: str, ys: jnp.ndarray, Xs: jnp.ndarray, data_name: str, tag: str) -> HMM:
    # Seed the dynamic transition from the fitted covariate HMM.
    base = load_model(_base_model_path(data_name, tag, "covariate_hmm"))
    transition_logits = base.transition.transition_logits
    initial_distribution = base.u_pre  # Use the initial distribution from the covariate HMM
    beta = base.transition.beta  # Use the beta from the covariate HMM as a starting point
    transition = DynamicTransition(transition_logits=transition_logits, beta=beta)

    # Use the emission from the AR(1) HMM as a starting point
    ar_model = load_model(_base_model_path(data_name, tag, "ar_hmm"))
    emission = ar_model.emission

    model = HMM(transition=transition, emission=emission,
                inital_distribution=initial_distribution)
    return model


def init_ar_2_covariate_hmm(model_name: str, ys: jnp.ndarray, Xs: jnp.ndarray, data_name: str, tag: str) -> HMM:
    # Seed the dynamic transition from the fitted AR(1) covariate HMM.
    base = load_model(_base_model_path(data_name, tag, "ar_1_covariate_hmm"))
    transition_logits = base.transition.transition_logits
    initial_distribution = base.u_pre  # Use the initial distribution from the covariate HMM
    beta = base.transition.beta  # Use the beta from the covariate HMM as a starting point
    transition = DynamicTransition(transition_logits=transition_logits, beta=beta)

    # Use the emission from the AR(2) HMM as a starting point
    ar_model = load_model(_base_model_path(data_name, tag, "ar_2_hmm"))
    emission = ar_model.emission

    model = HMM(transition=transition, emission=emission,
                inital_distribution=initial_distribution)
    return model


def init_ar_2_second_order_hmm(model_name: str, ys: jnp.ndarray, Xs: jnp.ndarray, data_name: str, tag: str) -> HMM:
    # Transition and means from the fitted (AR(1)) second-order HMM
    so_model = load_model(_base_model_path(data_name, tag, "second_order_hmm"))
    transition = so_model.params.transition
    mu = jnp.asarray(so_model.emission.mu_vals(0, ys))[:4]  # 4 tied means
    std = so_model.emission.sigma(0, ys)

    # Phi from the fitted AR(2) HMM
    ar2_model = load_model(_base_model_path(data_name, tag, "ar_2_hmm"))
    phi_vals = ar2_model.emission.phi()
    phi_vals = jnp.repeat(phi_vals, 4, axis=1)
    emission = AutoregressiveGaussEmission.from_params(mu=mu, sigma=std, phi=phi_vals)

    model = HMM(transition=transition, emission=emission)
    return model


# Registry so a model can be run just by name.
INIT_MODELS: dict[str, Callable] = {
    "ordinary_hmm": init_ordinary_hmm,
    "ar_hmm": init_ar_hmm,
    "ar_2_hmm": init_ar_2_hmm,
    "covariate_hmm": init_covariate_hmm,
    "second_order_hmm": init_second_order_hmm,
    "ar_1_covariate_hmm": init_ar_1_covariate_hmm,
    "ar_2_covariate_hmm": init_ar_2_covariate_hmm,
    "ar_2_second_order_hmm": init_ar_2_second_order_hmm,
}


if __name__ == "__main__":
    DATA_NAME = "b1"
    TAG = "jans-split"
    MODEL_NAME = "all"

    frozen_params = {
        "mu0": False
    }
    if (MODEL_NAME == "all"): 
        for model_name in INIT_MODELS.keys():
            fit_model(model_name=model_name,
                      data_name=DATA_NAME,
                      tag=TAG,
                      init_model=INIT_MODELS[model_name],
                      frozen_param=frozen_params)
    else:
        fit_model(model_name=MODEL_NAME,
              data_name=DATA_NAME,
              tag=TAG,
              init_model=INIT_MODELS[MODEL_NAME],
              frozen_param=frozen_params)
