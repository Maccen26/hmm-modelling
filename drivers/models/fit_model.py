from drivers.utils import load_train_data, load_test_data, save_model 
#from drivers.models.ordinary_hmm import run_ordinary_hmm
from typing import Callable
import jax.numpy as jnp
from src.api.v4 import HMM, StaticTransition, GaussEmission

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


if __name__ == "__main__":
    frozen_params = {
        "mu0": False
    }
    fit_model(model_name="ordinary_hmm", 
              data_name="b1", 
              tag="full", 
              init_model=init_ordinary_hmm, 
              frozen_param=frozen_params)
    

