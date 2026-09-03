from drivers.utils import plot_hmm_diagnostics, load_model
import matplotlib.pyplot as plt


def plot_ordinary_hmm_diagnostics():
    model_path = "results/models/ordinary_hmm.pkl"
    save_path = "results/plots/ordinary_hmm_diagnostics.png"
    model = load_model(model_path)
    fig = plot_hmm_diagnostics(model, save_path=save_path)
    return fig


if __name__ == "__main__":
    fig = plot_ordinary_hmm_diagnostics()
    plt.show()
