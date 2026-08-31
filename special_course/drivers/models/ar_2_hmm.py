from src.api.v4.utils import load_y_data
from src.api.v4 import HMM, AutoregressiveGaussEmission, StaticTransition
import jax.numpy as jnp
from drivers.utils import format_transition_matrix, load_model, save_model

def run_ar_2_hmm(model_name: str):
    print(f"Starting {model_name} model run...")
    ys = load_y_data()

    #Initiing parameters 
    PRE_MODEL_PATH = "results/models/ar_hmm.pkl"
    model = load_model(PRE_MODEL_PATH)
    mu = model.emission.mu_vals(0,0)
    std = model.emission.sigma(0,0)
    transition_matrix = model.transition.transition_matrix()
    phi_vals = model.emission.phi()
    phi_vals = jnp.concatenate([phi_vals, jnp.array([[0.20, 0.20, 0.20, 0.20]]) ])
    #raise NotImplementedError("This model is not fully implemented yet. The phi values are not being updated during fitting. This is a known issue and will be fixed in the future.")
    ## Initiating the HMM model
    transition = StaticTransition.from_params(transition_matrix)
    emission = AutoregressiveGaussEmission.from_params(mu=mu, sigma=std, phi=phi_vals)
    model = HMM(transition=transition, emission=emission) 

    ## Fitting the model
    frozen_params = {
    "mu0": False
    }

    print(f"Fitting {model_name} model...")
    model.fit(ys =ys, frozen=frozen_params)

    print(f"Finished fitting {model_name} model! The following parameters were found")
    print("Transition matrix:")

    print(format_transition_matrix(model.transition.transition_matrix()))
    
    print("Emission means:")
    print(model.emission.mu_vals(0,0))
    print("Emission stds:")
    print(model.emission.sigma(0,0))
    print("Emission phis :")
    print(model.emission.phi()) 
    print(f"Phi Summed across states: {jnp.sum(model.emission.phi(), axis=0)}")
    print("------------------------------------")

    return model


if __name__ == "__main__":
    model_name = "ar_2_hmm"
    PATH = f"results/models/{model_name}.pkl"
    model = run_ar_2_hmm(model_name)
    save_model(model, PATH)