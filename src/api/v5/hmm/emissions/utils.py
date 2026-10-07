import jax 
import jax.numpy as jnp


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