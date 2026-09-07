import jax
jax.config.update("jax_enable_x64", True)

from drivers.models.ordinary_hmm import run_ordinary_hmm
from drivers.models.ar_hmm import run_ar_hmm
from drivers.models.ar_2_hmm import run_ar_2_hmm
from drivers.models.second_order_hmm import run_ar_1_second_order_hmm
from drivers.models.ar_2_second_order_hmm import run_ar_2_second_order_hmm
from drivers.models.covariate_hmm import run_covariate_hmm
from drivers.models.ar_1_covariate_hmm import run_ar_1_covariate_hmm
from drivers.models.ar_2_covariate_hmm import run_ar_2_covariate_hmm
from drivers.utils import save_model


def main_models():
    # Ordered so that models seeding from a previous fit run after their dependency.
    # covariate_hmm seeds from ordinary_hmm, so it runs after it.
    models = {
        "ordinary_hmm": run_ordinary_hmm,
        "ar_hmm": run_ar_hmm,
        "ar_2_hmm": run_ar_2_hmm,
        "second_order_hmm": run_ar_1_second_order_hmm,
        "ar_2_second_order_hmm": run_ar_2_second_order_hmm,
        "covariate_hmm": run_covariate_hmm,
        "ar_1_covariate_hmm": run_ar_1_covariate_hmm,
        "ar_2_covariate_hmm": run_ar_2_covariate_hmm,
    }

    for model_name, model_function in models.items():
        model = model_function(model_name)
        save_model(model, f"results/models/{model_name}.pkl")


if __name__ == "__main__":
    main_models()
