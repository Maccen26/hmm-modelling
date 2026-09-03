import jax
jax.config.update("jax_enable_x64", True)

from drivers.models.main_models import main_models


def main():
    main_models()


if __name__ == "__main__":
    main()
