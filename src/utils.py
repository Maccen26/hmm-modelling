import jax.numpy as jnp
import jax
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_likelihood(
    param_values: jnp.ndarray,
    log_like: jnp.ndarray,
    zoom: float = 1.0,
    xlabel: str = "Parameter",
    ylabel: str = "Normalized Likelihood",
    plot: bool = True,
) -> tuple[np.ndarray, np.ndarray]:

    param_values = jnp.asarray(param_values)
    log_like = jnp.asarray(log_like)

    norm_log_like = log_like - jnp.max(log_like)

    params_np = np.array(param_values)
    norm_np = np.array(jnp.exp(norm_log_like))

    if plot:
        peak_idx = int(np.argmax(norm_np))
        peak_x = params_np[peak_idx]
        half_width = (params_np[-1] - params_np[0]) * zoom / 2
        x_min, x_max = peak_x - half_width, peak_x + half_width

        mask = (params_np >= x_min) & (params_np <= x_max)

        fig, ax = plt.subplots()
        ax.plot(params_np[mask], norm_np[mask])
        ax.axvline(peak_x, color="red", linestyle="--", linewidth=0.8, label=f"peak = {peak_x:.4g}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        plt.tight_layout()
        plt.show()

    return params_np, norm_np


def _init_mu(y, num_states):
    """Initialize mu from evenly spaced quantiles of y, excluding 0 and 100."""
    quantiles = jnp.linspace(25, 75, num_states - 1)
    return jnp.percentile(y, quantiles)

def _init_log_sigma(y, num_states):
    """Initialize log_sigma as log variance of y, shared across states."""
    #return jnp.log(jnp.std(y)) * jnp.ones(num_states)
    return jnp.array([jnp.log(150)] * num_states)  
def _init_transition_logits(key, y, mu, num_states):

    # Diagonal: count observations below each mu, shape (num_states, 1)
    diag = jnp.array([jnp.sum(y < mu[i]) for i in range(num_states)], dtype=float)
    log_diag = jnp.log(diag + 1).reshape(num_states, 1)

    # Off-diagonal: random, shape (num_states, num_states - 2)
    off_diag = jax.random.normal(key, shape=(num_states, num_states - 2))

    # Concatenate along columns -> (num_states, num_states - 1)
    return jnp.concatenate([log_diag, off_diag], axis=1)

def random_init_hmm_params(key, y, num_states):
    """Initialize HMM parameters using data-driven heuristics."""
    mu = _init_mu(y, num_states)
    log_sigma = _init_log_sigma(y, num_states)
    gamma = _init_transition_logits(key, y, mu, num_states)
    return mu, log_sigma, gamma


def professor_init_3state_hmm_params():
    """
    Initialize 3-state HMM parameters exactly as the professor does in R:
        gamma.pars <- rep(-3, 6)
        pars <- c(gamma.pars, 800, 1000, rep(10, 3))

    Professor's sigma convention: sigma2 = exp(pars), sigma = sqrt(sigma2)
    Our convention:               sigma  = exp(log_sigma)
    -> log_sigma = log(sqrt(exp(10))) = 10 / 2 = 5.0
    """
    tgamma0    = jnp.full((3, 2), -3.0)
    mu0        = jnp.array([800.0, 1000.0])
    log_sigma0 = jnp.full(3, 5.0)
    return mu0, log_sigma0, tgamma0


def professor_init_4state_hmm_params():
    """
    Initialize 4-state HMM parameters exactly as the professor does in R:
        gamma.pars <- c(-2,-10,-10, -2,-2,-10, -10,-2,-2, -10,-10,-2)
        pars <- c(gamma.pars, 550, 800, 1200, rep(10, 4))

    Banded structure: adjacent states logit -2 (easy), non-adjacent logit -10 (rare).
    """
    tgamma0 = jnp.array([
        [ -2.0, -10.0, -10.0],
        [ -2.0,  -2.0, -10.0],
        [-10.0,  -2.0,  -2.0],
        [-10.0, -10.0,  -2.0],
    ])
    mu0        = jnp.array([550.0, 800.0, 1200.0])
    log_sigma0 = jnp.full(4, 5.0)
    return mu0, log_sigma0, tgamma0



