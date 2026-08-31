from src.api.v4.utils import load_y_data 
import matplotlib.pyplot as plt
import jax.numpy as jnp
import os

def plot_co2_data(save_path: str | None = None):
    ys = load_y_data()
    plt.plot(ys)
    plt.plot(jnp.ones_like(ys) * 400, "--", label="mu0=400")
    plt.title("Observations")
    plt.xlabel("Time Step")
    plt.ylabel("CO2 Concentration (ppm)")
    plt.legend()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.show()



if __name__ == "__main__":

    plot_co2_data(save_path=os.path.join("results/plots", "co2_data.png"))




