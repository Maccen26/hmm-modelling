import os
import pickle

import jax.numpy as jnp


def save_model(model, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(model, f)


def load_model(save_path: str):
    with open(save_path, "rb") as f:
        model = pickle.load(f)
    return model


def format_transition_matrix(matrix: jnp.ndarray) -> str:
    formatted = "\n".join(["\t" + " ".join(f"{val:.4f}" for val in row) for row in matrix])
    return f"Transition Matrix:\n{formatted}"


def load_time_covariates(period: int = 48) -> jnp.ndarray:
    """
    Build cyclic time-of-day covariates (cos/sin of HalfHour) for the covariate HMM.

    Returns a (T, 2) array aligned with load_y_data(), since both draw from
    load_and_aggregate_data() with default arguments.
    """
    from src.data import load_and_aggregate_data  # lazy import to avoid circular import

    data = load_and_aggregate_data()
    t = jnp.asarray(data["HalfHour"], dtype=float)
    cos = jnp.cos(2 * jnp.pi * t / period)
    sin = jnp.sin(2 * jnp.pi * t / period)
    return jnp.column_stack((cos, sin))
