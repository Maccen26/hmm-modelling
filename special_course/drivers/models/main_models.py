import jax 
jax.config.update("jax_enable_x64", True)

from drivers.models.ordinary_hmm import run_ordinary_hmm 
from drivers.models.ar_hmm import run_ar_hmm 
from drivers.models.ar_2_hmm import run_ar_2_hmm
from drivers.models.second_order_hmm import run_ar_1_second_order_hmm
from drivers.models.ar_2_second_order_hmm import run_ar_2_second_order_hmm
from drivers.utils import save_model


def main_models():


    models = {
        "ordinary_hmm": run_ordinary_hmm,
        "ar_hmm": run_ar_hmm,
        "ar_2_hmm": run_ar_2_hmm,
        "second_order_hmm": run_ar_1_second_order_hmm,
        "ar_2_second_order_hmm": run_ar_2_second_order_hmm
    }
    
    for model_name, model_function in models.items():
        model = model_function(model_name)
        save_model(model, f"results/models/{model_name}.pkl")


if __name__ == "__main__":
    main_models()