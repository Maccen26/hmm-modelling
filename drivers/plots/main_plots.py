from drivers.utils import plot_hmm_diagnostics, load_model
import matplotlib.pyplot as plt


def main_plots():
    model_names = [
        "ordinary_hmm",
        "ar_hmm",
        "ar_2_hmm",
        "second_order_hmm",
        "ar_2_second_order_hmm",
        "covariate_hmm",
        "ar_1_covariate_hmm",
        "ar_2_covariate_hmm",
    ]

    for model_name in model_names:
        model = load_model(f"results/models/{model_name}.pkl")
        fig = plot_hmm_diagnostics(model, save_path=f"results/plots/{model_name}_diagnostics.png")
        plt.close(fig)


if __name__ == "__main__":
    main_plots()
