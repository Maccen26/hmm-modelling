"""Out-of-sample forecasts, metrics and diagnostic plots of a fitted model."""
import os

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf

from drivers.models.utils import HORIZONS, Config, Data
from drivers.utils import mape
from src.api.v5 import Params
from src.api.v5.analysis import Diagnostics, Predictor


def _covariates(params: Params, data: Data) -> jnp.ndarray | None:
    """Only covariate-driven transitions take xs (same check as Predictor)."""
    return data.xs if hasattr(params.transition, "beta") else None


def _columns(a: np.ndarray) -> np.ndarray:
    """(T,) -> (T, 1) so univariate and multivariate share one code path."""
    return a[:, None] if a.ndim == 1 else a


def rolling_predictions(params: Params, train: Data, val: Data, horizons=HORIZONS) -> dict[int, np.ndarray]:
    """k-step rolling forecasts of the validation set: filter up to t, predict t+k."""
    xs, xs_val = _covariates(params, train), _covariates(params, val)
    return {k: _columns(np.asarray(Predictor.rolling(params, train.ys, val.ys, xs=xs, xs_test=xs_val,
                                                     horizon=k, initial_dist=train.init_dist).y_pred))
            for k in horizons}


def history(train: Data, val: Data) -> Data:
    """Train followed by validation: everything the model has seen before the test split starts."""
    xs = None if train.xs is None or val.xs is None else jnp.concatenate([train.xs, val.xs])
    return Data(ys=jnp.concatenate([train.ys, val.ys]), xs=xs, init_dist=train.init_dist,
                datetime=train.datetime.append(val.datetime))


def pseudo_residuals(name: str, params: Params, train: Data, val: Data) -> tuple[np.ndarray, np.ndarray]:
    """One-step forecast pseudo-residuals of one filter pass over train then val, split into (train, val)."""
    ys = jnp.concatenate([train.ys, val.ys])
    xs = None if _covariates(params, train) is None else jnp.concatenate([train.xs, val.xs])
    diag = Diagnostics().add_model(name, params, train.init_dist, ys=ys, xs=xs)
    res = _columns(np.asarray(diag.residuals(name)))
    return res[:len(train.ys)], res[len(train.ys):]


def metrics_table(name: str, slug: str, cfg: Config, y_val: np.ndarray, preds: dict[int, np.ndarray]) -> pd.DataFrame:
    rows = []
    for k, y_pred in preds.items():
        for d, col in enumerate(cfg.obs_cols):
            y, p = y_val[:, d], y_pred[:, d]
            ss_res, ss_tot = np.sum((y - p) ** 2), np.sum((y - y.mean()) ** 2)
            rows.append({"model": name, "slug": slug, "horizon": k, "obs_col": col,
                         "R2": 1 - ss_res / ss_tot, "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
                         "MAPE": mape(y, p), "n_val": len(y)})
    return pd.DataFrame(rows)


def plot_diagnostics(name: str, y: np.ndarray, preds: dict[int, np.ndarray], res: np.ndarray, path: str,
                     split: str = "validation"):
    """Residuals vs predicted, residual histogram, and QQ/ACF of the pseudo-residuals for one obs column."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax_res, ax_hist, ax_qq, ax_acf = axes.ravel()

    for k, p in preds.items():
        ax_res.scatter(p, y - p, s=6, alpha=0.5, label=f"k={k}")
        ax_hist.hist(y - p, bins=50, alpha=0.4, label=f"k={k}")
    ax_res.axhline(0, color="black", lw=1)
    ax_res.set(title="Residuals vs predicted", xlabel="Predicted", ylabel="Observed − predicted")
    ax_res.legend()
    ax_hist.set(title="Histogram of prediction residuals", xlabel="Observed − predicted", ylabel="Count")
    ax_hist.legend()

    _plot_qq_acf(res, ax_qq, ax_acf)
    _save(fig, f"{name}: {split} set", path)


def plot_residual_diagnostics(name: str, res: np.ndarray, split: str, path: str):
    """QQ/ACF of the forecast pseudo-residuals alone, e.g. for the training set."""
    fig, (ax_qq, ax_acf) = plt.subplots(1, 2, figsize=(12, 4.5))
    _plot_qq_acf(res, ax_qq, ax_acf)
    _save(fig, f"{name}: {split} set", path)


def _plot_qq_acf(res: np.ndarray, ax_qq, ax_acf):
    res = res[~np.isnan(res)]
    sm.qqplot(res, line="45", ax=ax_qq)
    ax_qq.set_title("QQ plot of forecast pseudo-residuals")
    plot_acf(res, ax=ax_acf, lags=48, zero=False, markersize=3)
    ax_acf.set_ylim(-0.3, 0.3)
    ax_acf.set_title("ACF of forecast pseudo-residuals")


def _save(fig, title: str, path: str):
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_first_week(name: str, col: str, dt: pd.DatetimeIndex, y: np.ndarray, preds: dict[int, np.ndarray],
                    table: pd.DataFrame, path: str, split: str = "validation"):
    """Observed vs k-step predicted emission over the first week of the split, one subplot per k."""
    week = np.asarray(dt < dt[0] + pd.Timedelta(days=7))
    fig, axes = plt.subplots(len(preds), 1, figsize=(14, 2.6 * len(preds)), sharex=True, squeeze=False)

    for ax, (k, p) in zip(axes[:, 0], preds.items()):
        m = table[(table["horizon"] == k) & (table["obs_col"] == col)].iloc[0]
        ax.plot(dt[week], y[week], color="black", lw=1.2, label="Observed")
        ax.plot(dt[week], p[week], color="tab:orange", lw=1.2, label=f"Predicted (k={k})")
        ax.set_title(f"k={k}   RMSE={m.RMSE:.1f}   R²={m.R2:.3f}   MAPE={m.MAPE:.1f}%", loc="left")
        ax.set_ylabel(col)
        ax.legend(loc="upper right")

    fig.suptitle(f"{name}: first week of the {split} set (metrics over the full {split} set)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_label(name: str, cfg: Config) -> str:
    """Plot-title prefix naming the model, room, number of states and loss."""
    return f"{name} — {cfg.room}, K={cfg.states}, loss={cfg.loss.slug}"
