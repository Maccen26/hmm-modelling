from drivers.utils import plot_hmm_diagnostics, load_model
import matplotlib.pyplot as plt


def plot_second_order_hmm_diagnostics():
    model_path = "results/models/second_order_hmm.pkl"
    save_path = "results/plots/second_order_hmm_diagnostics.png"
    model = load_model(model_path)
    fig = plot_hmm_diagnostics(model, save_path=save_path)
    return fig


if __name__ == "__main__":
    fig = plot_second_order_hmm_diagnostics()
    plt.show()
