import jax
jax.config.update("jax_enable_x64", True)

from drivers.legacy.models.main_models import main_models
from drivers.legacy.main_plots import main_plots
from drivers.legacy.test_statistics import main_test_statistics


def main():
    main_models()
    main_plots()
    main_test_statistics()


if __name__ == "__main__":
    main()
