import jax
jax.config.update("jax_enable_x64", True)

from src.api.v4.utils import load_y_data
from src.api.v4 import HMM, AutoregressiveGaussEmission, StaticTransitionHigherOrder
from src.api.v4.transitions.static_transition_higher_order import decode_possible_transitions

import jax.numpy as jnp
from drivers.utils import load_model, save_model


def run_ar_1_second_order_hmm(model_name: str):
    print(f"Starting {model_name} model run...")
    ys = load_y_data()

    PRE_MODEL_PATH = "results/models/ar_hmm.pkl"
    model = load_model(PRE_MODEL_PATH)

    params = model.params
    transition_logits = list(params.transition.transition_logits)

    transition = StaticTransitionHigherOrder(jnp.asarray(transition_logits * 4), order=2)

    mu_init = jnp.asarray(params.emission.mu(0, ys))  # length 4, tied across s_{t-1}
    sigma_init = jnp.sort(jnp.repeat(params.emission.sigma(0, ys), 4))
    phi_vals = jnp.repeat(params.emission.phi(), 4, axis=1)
    emission = AutoregressiveGaussEmission.from_params(mu=mu_init, sigma=sigma_init, phi=phi_vals)

    model = HMM(transition=transition, emission=emission)

    frozen_params = {"mu0": False}

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
    print("Phi values:")
    print(model.emission.phi())
    print("------------------------------------")

    return model


if __name__ == "__main__":
    model_name = "second_order_hmm"
    PATH = f"results/models/{model_name}.pkl"
    model = run_ar_1_second_order_hmm(model_name)
    save_model(model, PATH)
