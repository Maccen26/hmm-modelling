"""Shared config, data loading, likelihood and seeding helpers for the drivers."""
import argparse
import os
import re
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Callable

import jax.numpy as jnp
import numpy as np
import pandas as pd

from drivers.utils import build_covariates, load_y_df
from src.api.v5 import (
    AutoregressiveGaussEmission,
    ForwardAlgorithm,
    GaussEmission,
    LBFGSSolver,
    MultivariateAutoregressiveGaussEmission,
    MultivariateGaussEmission,
    Params,
    StaticTransition,
)
from src.api.v5.solvers.base_solver import SolverResult

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
RESULTS_DIR = os.path.join(ROOT, "results")
PLOTS_DIR = os.path.join(ROOT, "plots")
HORIZONS = (1, 2, 4, 6, 8)  # rolling-forecast horizons (steps ahead) for the validation set
PLOT_HORIZONS = (6, 8)  # subset of HORIZONS drawn in the residual and first-week plots
TAG = "calibrated_anchor"   # default data tag: the folder of the room CSVs, and a level of the results/plots tree

OBS_COLS = ("co2",)         # ≥2 columns -> multivariate emission
COVARIATE_COLS = ("off_day", "working_hours")  # β models only; beats the tod/week harmonics out of sample (covariates_fit.ipynb)


@dataclass(frozen=True)
class Loss:
    """NLL plus an optional ridge beta_lam * sum(beta^2) on the transition covariate effects (β models only)."""
    beta_lam: float = 0.0

    @property
    def fn(self) -> Callable:
        return partial(penalised_negative_likelihood, beta_lam=self.beta_lam)

    @property
    def penalised(self) -> bool:
        return bool(self.beta_lam)

    @property
    def slug(self) -> str:
        """e.g. nll, ridge_beta_lam0.01"""
        return f"ridge_beta_lam{self.beta_lam:g}" if self.beta_lam else "nll"


@dataclass(frozen=True)
class Config:
    room: str                       # e.g. "room_009"
    obs_cols: tuple[str, ...]       # ≥2 columns -> multivariate emission
    covariate_cols: tuple[str, ...] # only used by the β models
    states: int
    loss: Loss = Loss()
    tag: str = TAG
    n_iter: int = 10_000
    tol: float = 1e-6

    def __post_init__(self):
        object.__setattr__(self, "obs_cols", tuple(self.obs_cols))
        object.__setattr__(self, "covariate_cols", tuple(self.covariate_cols))

    @property
    def multivariate(self) -> bool:
        return len(self.obs_cols) > 1

    def run_dir(self, base: str) -> str:
        """{base}/{room}/{tag}/states_{K}: where this run's fits, metrics and plots live."""
        return os.path.join(base, self.room, self.tag, f"states_{self.states}")


@dataclass(frozen=True)
class Data:
    ys: jnp.ndarray                 # (T,) or (T, D)
    xs: jnp.ndarray | None          # (T, C) covariates for the transition
    init_dist: jnp.ndarray          # (1, K)
    datetime: pd.DatetimeIndex = field(repr=False)

    def static(self) -> tuple:
        return (self.ys, None, self.init_dist)

    def dynamic(self) -> tuple:
        return (self.ys, self.xs, self.init_dist)

    def without_covariates(self) -> "Data":
        """Copy with xs dropped, for saving the data a static model was fitted on."""
        return replace(self, xs=None)


def room_name(room: int) -> str:
    return f"room_{room:03d}"


def add_run_args(parser: argparse.ArgumentParser, default_states: int | None = None) -> argparse.ArgumentParser:
    """--room, --states and --tag, shared by the driver CLIs."""
    parser.add_argument("--room", "-r", type=int, required=True, help="room number, e.g. 9 for room_009")
    parser.add_argument("--states", "-K", type=int, default=default_states, required=default_states is None,
                        help="number of hidden states" + (f" (default {default_states})" if default_states else ""))
    parser.add_argument("--tag", default=TAG, help=f"data tag, i.e. the binning folder of the room data (default {TAG})")
    return parser


# --- Data ---------------------------------------------------------------------------

def working_hours(dt: pd.DatetimeIndex) -> np.ndarray:
    """1 between 08:00 and 17:00 Monday-Friday, else 0."""
    return ((dt.hour >= 8) & (dt.hour < 17) & (dt.dayofweek < 5)).astype(float)


def _harmonics(names, label: str) -> tuple[int, ...]:
    pattern = re.compile(rf"^(?:sin|cos)_{label}_(\d+)$")
    return tuple(sorted({int(m.group(1)) for n in names if (m := pattern.match(n))}))


def covariates(dt: pd.DatetimeIndex, names) -> jnp.ndarray:
    """(T, C) covariate matrix with columns in the order of `names`."""
    X, all_names = build_covariates(
        dt,
        tod_harmonics=_harmonics(names, "tod"),
        week_harmonics=_harmonics(names, "week"),
        use_off_day=True,
        weather_cols=[],
    )
    df = pd.DataFrame(X, columns=all_names)
    df["working_hours"] = working_hours(dt)
    unknown = [n for n in names if n not in df.columns]
    if unknown:
        raise ValueError(f"Unknown covariate columns {unknown}; available: {df.columns.tolist()}")
    return jnp.asarray(df[list(names)].to_numpy())


def load_data(cfg: Config, split: str = "train") -> Data:
    df = load_y_df(cfg.room, cfg.tag, split).sort_values("datetime")
    dt = pd.DatetimeIndex(df["datetime"])
    ys = df[list(cfg.obs_cols)].to_numpy()
    ys = jnp.asarray(ys if cfg.multivariate else ys[:, 0])
    xs = covariates(dt, cfg.covariate_cols)
    init_dist = jnp.ones((1, cfg.states)) / cfg.states
    return Data(ys=ys, xs=xs, init_dist=init_dist, datetime=dt)


# --- Likelihood ---------------------------------------------------------------------

def negative_likelihood(params: Params, data: tuple):
    ys, xs, init_dist = data
    _, f_t, _ = ForwardAlgorithm().run(params=params, ys=ys, xs=xs, initial_dist=init_dist)
    return -jnp.sum(jnp.log(f_t))


def penalised_negative_likelihood(params: Params, data: tuple, beta_lam: float = 0.0):
    """Negative log-likelihood plus the ridge penalty beta_lam * sum(beta^2)."""
    loss = negative_likelihood(params, data)
    if beta_lam:
        loss = loss + beta_lam * jnp.sum(params.transition.beta ** 2)
    return loss


def number_of_clipped_ft(params: Params, data: tuple) -> int:
    ys, xs, init_dist = data
    _, f_t, _ = ForwardAlgorithm().run(params=params, ys=ys, xs=xs, initial_dist=init_dist)
    return int(jnp.sum(f_t <= 1e-100))


# --- Seeding ------------------------------------------------------------------------

def seed_emission(cfg: Config, ys: jnp.ndarray):
    """Quantile means and the overall std for every state."""
    quantiles = jnp.linspace(0.05, 0.95, cfg.states)
    if cfg.multivariate:
        mu = jnp.quantile(ys, quantiles, axis=0).T                      # (D, K)
        sigma = jnp.std(ys, axis=0)[:, None] * jnp.ones((cfg.states,))  # (D, K)
        return MultivariateGaussEmission.from_params(mu=mu, sigma=sigma)
    mu = jnp.quantile(ys, quantiles)
    sigma = jnp.std(ys) * jnp.ones((cfg.states,))
    return GaussEmission.from_params(mu=mu, sigma=sigma)


def floored_transition_matrix(transition, floor: float = 1e-8) -> jnp.ndarray:
    """Γ with entries floored and rows renormalised: entries that underflowed to 0 in a fitted model
    would give -inf logits when seeding a new transition, which L-BFGS turns into NaN on step 1."""
    gamma = jnp.maximum(transition.transition_matrix(), floor)
    return gamma / gamma.sum(axis=1, keepdims=True)


def seed_transition(states: int) -> StaticTransition:
    matrix = jnp.eye(states) * 0.7 + jnp.ones((states, states)) * 0.3 / states
    return StaticTransition.from_params(transition_matrix=matrix)


def ar1_emission_from(emission, cfg: Config, phi0: float = 0.05):
    """AR(1) emission seeded with the state means/stds of a fitted Gaussian emission."""
    mu, sigma = emission.mu(t=0, ys=0), emission.sigma(t=0, ys=0)
    if cfg.multivariate:
        phi = jnp.ones((1, len(cfg.obs_cols), cfg.states)) * phi0
        return MultivariateAutoregressiveGaussEmission.from_params(mu=mu, sigma=sigma, phi=phi)
    phi = jnp.ones(cfg.states) * phi0
    return AutoregressiveGaussEmission.from_params(mu=mu, sigma=sigma, phi=phi)


# --- Fitting ------------------------------------------------------------------------

def fit_and_report(name: str, params: Params, data: tuple, cfg: Config) -> SolverResult:
    print(f"--- Fitting {name} ({cfg.room}, obs={list(cfg.obs_cols)}, K={cfg.states}, loss={cfg.loss.slug})")
    result = LBFGSSolver(n_iter=cfg.n_iter, tol=cfg.tol).fit(
        params=params, loss_fn=cfg.loss.fn, data=data
    )
    print(f"Final loss: {result.loss}")
    if cfg.loss.penalised:
        print(f"Negative log-likelihood (unpenalised): {float(negative_likelihood(result.params, data))}")
    print(f"Converged: {result.converged}")
    print(f"Number of clipped steps in f_t: {number_of_clipped_ft(result.params, data)}")
    return result


def fit_path(room: str, states: int, slug: str, tag: str = TAG) -> str:
    """Path of a saved fit."""
    return os.path.join(RESULTS_DIR, room, tag, f"states_{states}", f"{slug}.pkl")
