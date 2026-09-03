import jax
jax.config.update("jax_enable_x64", True)

from src.api.v4.utils import load_y_data
from src.api.v4 import HMM, AutoregressiveGaussEmission, StaticTransitionHigherOrder
from src.api.v4.transitions.static_transition_higher_order import decode_possible_transitions

import jax.numpy as jnp
from drivers.utils import load_model, save_model


def run_ar_2_second_order_hmm(model_name: str):
    print(f"Starting {model_name} model run...")
    ys = load_y_data()

    # Initiating parameters
    PRE_MODEL_PATH = "results/models/second_order_hmm.pkl"
    PRE_ARR_PATH = "results/models/ar_2_hmm.pkl"

    model = load_model(PRE_MODEL_PATH)
    transition = model.params.transition
    mu = jnp.asarray(model.emission.mu_vals(0, ys))[:4]  # 4 tied means
    std = model.emission.sigma(0, ys)

    ar2_model = load_model(PRE_ARR_PATH)
    phi_vals = ar2_model.emission.phi()
    phi_vals = jnp.repeat(phi_vals, 4, axis=1)
    emission = AutoregressiveGaussEmission.from_params(mu=mu, sigma=std, phi=phi_vals)

    model = HMM(transition=transition, emission=emission)

    # Fitting the model
    frozen_params = {
        "mu0": False
    }

    print(f"Fitting {model_name} model...")
    model.fit(ys=ys, frozen=frozen_params)

    print(f"Finished fitting {model_name} model! The following parameters were found")
    print("Transition matrix:")
    transition_matrix = model.transition.transition_matrix(0, ys)
    state_transions = decode_possible_transitions(transition_matrix, order=2)
    for state, transition_list in state_transions.items():
        print(f"From {state}:")
        for to_state, prob in transition_list:
            print(f"  To {to_state}: {prob:.2f}")
        print("----------")

    print("Emission means:")
    print(model.emission.mu_vals(0, 0))
    print("Emission stds:")
    print(model.emission.sigma(0, 0))
    print("------------------------------------")
    print("Phi values:")
    print(model.emission.phi())
    print(f"Phi Summed across states: {jnp.sum(model.emission.phi(), axis=0)}")
    return model


if __name__ == "__main__":
    model_name = "ar_2_second_order_hmm"
    PATH = f"results/models/{model_name}.pkl"
    model = run_ar_2_second_order_hmm(model_name)
    save_model(model, PATH)
