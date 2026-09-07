from src.api.v4.utils import load_y_data
from src.api.v4 import HMM, GaussEmission, StaticTransition
import jax.numpy as jnp
from drivers.utils import save_model, format_transition_matrix


def run_ordinary_hmm(model_name: str):
    print(f"Starting {model_name} model run...")
    ys = load_y_data()

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

    # Fitting the model
    frozen_params = {
        "mu0": False
    }
    print("Fitting ordinary HMM model...")
    model.fit(ys=ys, frozen=frozen_params)
    print("Finished fitting ordinary HMM model! The following parameters were found")
    print("Transition matrix:")
    print(format_transition_matrix(model.transition.transition_matrix()))

    print("Emission means:")
    print(model.emission.mu(0, 0))
    print("Emission stds:")
    print(model.emission.sigma(0, 0))
    print("------------------------------------")
    return model


if __name__ == "__main__":
    model_name = "ordinary_hmm"
    PATH = f"results/models/{model_name}.pkl"
    model = run_ordinary_hmm(model_name)
    save_model(model, PATH)
