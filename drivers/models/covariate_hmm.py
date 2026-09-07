import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as random

from src.api.v4 import HMM
from src.api.v4.transitions.dynamic_transition import DynamicTransition
from src.api.v4.utils import load_y_data
from drivers.utils import save_model, load_model, format_transition_matrix, load_time_covariates


def run_covariate_hmm(model_name: str):
    print(f"Starting {model_name} model run...")
    ys = load_y_data()
    xs = load_time_covariates()  # (T, 2) cyclic time-of-day covariates

    # Seed the dynamic transition from the fitted base (ordinary) HMM.
    BASE_MODEL_PATH = "results/models/ordinary_hmm.pkl"
    base = load_model(BASE_MODEL_PATH)
    transition_logits = base.transition.transition_logits
    initial_distribution = base._compute_stationary_distribution()
    emission = base.emission

    # beta has shape (num_covariates,) + transition_logits.shape, e.g. (2, 4, 3).
    num_covariates = xs.shape[1]
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

    # Fitting the model
    frozen_params = {
        "mu0": False
    }
    print(f"Fitting {model_name} model...")
    model.fit(ys=ys, xs=xs, frozen=frozen_params)

    print(f"Finished fitting {model_name} model! The following parameters were found")
    print("Base transition matrix (no covariate effect):")
    print(format_transition_matrix(model.transition.base_transition_matrix()))

    print("Emission means:")
    print(model.emission.mu(0, 0))
    print("Emission stds:")
    print(model.emission.sigma(0, 0))
    print("Beta (covariate coefficients):")
    print(model.transition.beta)
    print("------------------------------------")
    return model


if __name__ == "__main__":
    model_name = "covariate_hmm"
    PATH = f"results/models/{model_name}.pkl"
    model = run_covariate_hmm(model_name)
    save_model(model, PATH)
