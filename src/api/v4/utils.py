import jax.numpy as jnp
import jax 

def load_y_data(no_of_days: int | None = None) -> jnp.ndarray:
    """
    Loads the y data from the csv file and returns it as a jnp array. 
    If no_of_days is not None, it returns only the first no_of_days of data. 
    """ 
    from src.data import load_and_aggregate_data
    df = load_and_aggregate_data(no_of_days=no_of_days)
    y_data = jnp.array(df["CO2C"].values)
    return jnp.asarray(y_data) 



def make_lag_matrix(ys: jnp.ndarray, k: int) -> jnp.ndarray:
    """
    Build (T, k) lag matrix where row t = [y_{t-1}, ..., y_{t-k}].
    Rows t < k are zero-padded (valid lags don't exist yet).
    """
    T = len(ys)
    cols = [jnp.concatenate([jnp.zeros(lag), ys[:T - lag]]) for lag in range(1, k + 1)]
    return jnp.stack(cols, axis=1)  # (T, k)



#def phi_to_phi_tilde(phi):
#    return jax.scipy.special.logit(phi)  # constrained → unconstrained
#
#def phi_tilde_to_phi(phi_tilde):
#    return jax.nn.sigmoid(phi_tilde)     # unconstrained → constrained (0, 1)


def phi_to_phi_tilde(phi):
    # (-1, 1) → (0, 1) → (-∞, ∞)
    #
    # Clip off the endpoints first. The inverse, phi_tilde_to_phi, saturates: once
    # phi_tilde exceeds ~37 the float64 result rounds to exactly 1.0. A fitted model
    # whose phi has hit that boundary is then reseeded through this function (see
    # init_ar_2_hmm in drivers/fit_model_b1.py), and logit(1.0) = +inf makes every
    # gradient NaN — poisoning the fit and everything seeded from it.
    phi = jnp.clip(phi, -1.0 + 1e-12, 1.0 - 1e-12)
    return jax.scipy.special.logit((phi + 1) / 2)

def phi_tilde_to_phi(phi_tilde):
    # (-∞, ∞) → (0, 1) → (-1, 1)
    return 2 * jax.nn.sigmoid(phi_tilde) - 1