from dataclasses import dataclass
import equinox as eqx
import jax
import jax.numpy as jnp
from src.api.v5.hmm.params import Params
from src.api.v5.inference.forward_algorithm import ForwardAlgorithm
from src.api.v5.analysis.diagnostics import uniform_initial_dist


@dataclass(frozen=True)
class Prediction:
    y_pred: jnp.ndarray        # (H,) or (H, K)
    state_probs: jnp.ndarray   # (H, N): predicted state distribution P(z_t | y_{<=origin})

    def rmse(self, y_true: jnp.ndarray) -> float:
        return float(jnp.sqrt(jnp.mean((jnp.asarray(y_true) - self.y_pred) ** 2)))

    def r2(self, y_true: jnp.ndarray) -> float:
        y_true = jnp.asarray(y_true)
        ss_res = jnp.sum((y_true - self.y_pred) ** 2)
        ss_tot = jnp.sum((y_true - jnp.mean(y_true, axis=0)) ** 2)
        return float(1 - ss_res / ss_tot)


@eqx.filter_jit
def _forecast(params: Params, u0: jnp.ndarray, ys_buffer: jnp.ndarray, t0, x_future: jnp.ndarray | None, horizon: int):
    """
    Forecast steps t0+1 .. t0+horizon from the filtered state u0 at t0.

    Each step propagates the state (u <- u @ Gamma(x_{t})) and takes the
    state-weighted emission mean. The prediction is written into `ys_buffer[t]`,
    so an AR emission's lags beyond the data are the predicted values (plug-in
    mean). For AR models with horizon > 1 this is an approximation of E[y_t].
    """
    def step(carry, inp):
        u, buf = carry
        j, xt = inp
        t = t0 + j
        Gamma = params.transition.transition_matrix() if xt is None else params.transition.transition_matrix(xt)
        u = u @ Gamma
        y_hat = params.emission.mu(t, buf) @ u
        buf = buf.at[t].set(y_hat)
        return (u, buf), (y_hat, u)

    _, (y_pred, us) = jax.lax.scan(step, (jnp.ravel(u0), ys_buffer), (jnp.arange(1, horizon + 1), x_future))
    return y_pred, us


class Predictor:
    """Stateless forecaster: every call takes the model and the data it needs."""

    @staticmethod
    def _check_covariates(params: Params, *xs) -> None:
        if hasattr(params.transition, "beta") and any(x is None for x in xs):
            raise ValueError("This model has a covariate-driven transition: pass the covariates "
                             "for both the observed and the forecast steps.")

    @classmethod
    def predict(cls, params: Params, ys: jnp.ndarray, horizon: int, xs: jnp.ndarray | None = None,
                x_pred: jnp.ndarray | None = None, initial_dist: jnp.ndarray | None = None) -> Prediction:
        """
        Absolute forecast of the `horizon` steps after the end of `ys`, with no updates.
        `x_pred[h-1]` holds the covariates of forecast step h.
        """
        cls._check_covariates(params, xs, x_pred)
        initial_dist = uniform_initial_dist(params) if initial_dist is None else initial_dist
        ys = jnp.asarray(ys, dtype=float)
        utts, _, _ = ForwardAlgorithm.run(params=params, initial_dist=initial_dist, ys=ys, xs=xs)

        buffer = jnp.concatenate([ys, jnp.zeros((horizon, *ys.shape[1:]))])
        x_future = None if x_pred is None else jnp.asarray(x_pred)[:horizon]
        y_pred, us = _forecast(params, utts[-1], buffer, len(ys) - 1, x_future, horizon)
        return Prediction(y_pred=y_pred, state_probs=us)

    @classmethod
    def rolling(cls, params: Params, ys: jnp.ndarray, ys_test: jnp.ndarray, xs: jnp.ndarray | None = None,
                xs_test: jnp.ndarray | None = None, horizon: int = 1,
                initial_dist: jnp.ndarray | None = None) -> Prediction:
        """
        Rolling h-step forecasts over the test set. Test point y_t is predicted from
        the filtered state at its origin t-h, so each prediction only uses data up
        to its origin, and the real observations are filtered in as it moves on.
        Returns one prediction (and predicted state distribution) per test point.
        """
        cls._check_covariates(params, xs, xs_test)
        initial_dist = uniform_initial_dist(params) if initial_dist is None else initial_dist
        ys_all = jnp.concatenate([jnp.asarray(ys, dtype=float), jnp.asarray(ys_test, dtype=float)])
        xs_all = None if xs is None else jnp.concatenate([jnp.asarray(xs), jnp.asarray(xs_test)])
        utts, _, _ = ForwardAlgorithm.run(params=params, initial_dist=initial_dist, ys=ys_all, xs=xs_all)

        origins = jnp.arange(len(ys), len(ys_all)) - horizon

        def one(o):
            x_future = None if xs_all is None else jax.lax.dynamic_slice_in_dim(xs_all, o + 1, horizon)
            y_pred, us = _forecast(params, utts[o], ys_all, o, x_future, horizon)
            return y_pred[-1], us[-1]

        y_pred, us = jax.vmap(one)(origins)
        return Prediction(y_pred=y_pred, state_probs=us)
