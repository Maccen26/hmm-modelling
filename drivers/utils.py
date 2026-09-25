import os
import pickle
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
from dotenv import load_dotenv

def load_train_data(data_name: str, tag: str) -> tuple[jnp.ndarray, jnp.ndarray]:
    y_path, X_path = load_data_path(data_name=data_name, tag=tag, arr_type="train")
    ys = load_csv_to_jnp(y_path)
    ys = ys.flatten()  # Ensure ys is a 1D array
    Xs = load_csv_to_jnp(X_path)
    return ys, Xs

def load_val_data(data_name: str, tag: str) -> tuple[jnp.ndarray, jnp.ndarray]:
    y_path, X_path = load_data_path(data_name=data_name, tag=tag, arr_type="val")
    ys = load_csv_to_jnp(y_path)
    ys = ys.flatten()  # Ensure ys is a 1D array
    Xs = load_csv_to_jnp(X_path)
    return ys, Xs

def load_test_data(data_name: str, tag: str) -> tuple[jnp.ndarray, jnp.ndarray]:
    y_path, X_path = load_data_path(data_name=data_name, tag=tag, arr_type="test")
    ys = load_csv_to_jnp(y_path)
    ys.flatten()  # Ensure ys is a 1D array
    Xs = load_csv_to_jnp(X_path)
    return ys, Xs

def load_data_path(data_name: str, tag: str, arr_type: str) -> tuple[str, str]:
    base_path = load_base_data_path()
    y_path = os.path.join(base_path, f"{data_name}/{tag}/y_{arr_type}.csv")
    X_path = os.path.join(base_path, f"{data_name}/{tag}/X_{arr_type}.csv")
    return y_path, X_path

def load_csv_to_jnp(path) -> jnp.ndarray:
    arr = np.loadtxt(path, delimiter=",")
    arr = jnp.asarray(arr)
    return arr

def load_base_data_path() -> str:
    load_dotenv()
    PATH = os.getenv("DATA_PATH")
    if (PATH is None):
        raise ValueError("DATA_PATH environment variable is not set.")
    return PATH


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

    DEPRECATED: hardcoded to period=48 and tied to the legacy
    `src.data.load_and_aggregate_data` path. Use `build_covariates` instead, which takes
    real timestamps and any number of harmonics.
    """
    from src.data import load_and_aggregate_data  # lazy import to avoid circular import

    data = load_and_aggregate_data()
    t = jnp.asarray(data["HalfHour"], dtype=float)
    cos = jnp.cos(2 * jnp.pi * t / period)
    sin = jnp.sin(2 * jnp.pi * t / period)
    return jnp.column_stack((cos, sin))



# --------------------------------------------------------------------------- #
# Covariate construction
#
# Shared by `drivers/clean_dtu_data.py`, `week_5.ipynb` and
# `week_5_optimal_parameters.ipynb` so the three cannot drift apart.
# --------------------------------------------------------------------------- #

WEATHER_COLS = [
    "mean_temp",
    "mean_relative_hum",
    "mean_wind_speed",
    "mean_pressure",
    "mean_cloud_cover",
    "mean_radiation",
]


def weather_path() -> Path:
    return Path(load_base_data_path()) / "raw" / "dtu" / "weather.csv"


def harmonic_range(n_list: list) -> tuple[int, ...]:
    """Cycle count -> harmonic indices. `harmonic_range(0)` drops the block entirely."""
    return tuple(n_list) if n_list else ()


def is_holiday_mask(dt: pd.DatetimeIndex) -> np.ndarray:
    import holidays as _holidays

    dk = _holidays.Denmark(years=sorted(set(dt.year)))
    return np.array([d.date() in dk for d in dt])


def fourier_columns(angle, harmonics, label):
    """sin/cos pairs for each requested harmonic of a 2*pi-normalised angle."""
    cols, names = [], []
    for k in harmonics:
        cols += [np.sin(k * angle), np.cos(k * angle)]
        names += [f"sin_{label}_{k}", f"cos_{label}_{k}"]
    return cols, names


def weather_features(datetimes, weather_cols: list[str]) -> np.ndarray:
    """Match hourly weather (UTC) to each local, tz-naive observation time."""
    if not weather_cols:
        return np.empty((len(datetimes), 0))

    weather_df = pd.read_csv(weather_path(), parse_dates=["DateFrom", "DateTo"])
    weather_df = (
        weather_df[["DateFrom", *weather_cols]]
        .dropna()
        .sort_values("DateFrom")
        .reset_index(drop=True)
    )

    local = pd.DatetimeIndex(datetimes).tz_localize(
        "Europe/Copenhagen", ambiguous="NaT", nonexistent="shift_forward"
    )
    left = pd.DataFrame({"t": local.tz_convert("UTC")})
    left["order"] = np.arange(len(left))
    left = left.sort_values("t")
    merged = pd.merge_asof(
        left, weather_df.rename(columns={"DateFrom": "t"}), on="t", direction="nearest",
    )
    merged = merged.sort_values("order")
    return merged[weather_cols].ffill().bfill().to_numpy()


def build_covariates(
    datetimes,
    tod_harmonics: tuple[int, ...] = (1, 2, 3),
    week_harmonics: tuple[int, ...] = (1,),
    use_off_day: bool = True,
    weather_cols: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Return (X, names): an (N, D) covariate matrix and its D column names."""
    weather_cols = WEATHER_COLS if weather_cols is None else weather_cols
    dt = pd.DatetimeIndex(datetimes)
    cols, names = [], []

    if use_off_day:
        is_weekend = dt.dayofweek >= 5
        cols.append((is_weekend | is_holiday_mask(dt)).astype(float))
        names.append("off_day")

    seconds_into_week = dt.dayofweek * 86400 + dt.hour * 3600 + dt.minute * 60 + dt.second
    c, n = fourier_columns(2 * np.pi * seconds_into_week / (7 * 86400), week_harmonics, "week")
    cols += c
    names += n

    seconds_into_day = dt.hour * 3600 + dt.minute * 60 + dt.second
    c, n = fourier_columns(2 * np.pi * seconds_into_day / 86400, tod_harmonics, "tod")
    cols += c
    names += n

    X = np.column_stack(cols) if cols else np.empty((len(dt), 0))
    return np.column_stack([X, weather_features(datetimes, weather_cols)]), [*names, *weather_cols]


def standardise(X: np.ndarray, train_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score X with training statistics only, so nothing from val/test leaks into the scaling."""
    mean = X[:train_idx].mean(axis=0)
    std = X[:train_idx].std(axis=0)
    std = np.where(std == 0, 1.0, std)           # a flag can be constant in a short segment
    return (X - mean) / std, mean, std


# --------------------------------------------------------------------------- #
# Forecast metrics
# --------------------------------------------------------------------------- #

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def r2_score(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)




def state_means(emission, ys):
    """Per-state base means, whichever emission is asked.

    `GaussEmission` exposes them directly as `mu`; the autoregressive emission splits them
    into `mu_vals` (state base) and `mu` (base + AR term on the previous values).
    """
    return getattr(emission, "mu_vals", emission.mu)(0, ys)


# --------------------------------------------------------------------------- #
# Rolling multi-step forecasting
# --------------------------------------------------------------------------- #

def rolling_forecast(model, ys, xs_std, start: int, end: int, K: int):
    """Fixed K-step-ahead plug-in forecast anchored at every observation in [start, end - K).

    Returns `(target_idx, y_true, y_pred)`: the index of each forecast target, the realised
    value there, and the forecast of it. Anchors stop at `end - K` so every target falls
    inside the block, which keeps the scores clear of the training block.

    General in the number of AR lags: `lags[:, j]` is y_{t-j}, matching the flipped slice in
    `AutoregressiveGaussEmission.mu`, and each prediction is pushed onto the front of that
    window as the next step's lag-1 value. Works unchanged for a higher-order transition,
    whose augmented state space just makes the arrays wider. A non-autoregressive emission
    takes the constant-mean branch.
    """
    from src.api.v4.algorithms import ForwardAlgorithm  # lazy: avoid a circular import

    ys = jnp.asarray(ys)
    T = len(ys)
    out = ForwardAlgorithm().run(model.params, model.u_pre, ys=ys, ts=None, xs=xs_std)
    utt = out.utt.reshape(T, -1)                                          # (T, S)
    # `ts` is ignored by a discrete transition, so unit waiting times are fine here.
    Gammas = model.transition.transition_matrices(jnp.arange(T), jnp.ones(T), ys, xs_std)

    anchors = jnp.arange(start, end - K)
    u = utt[anchors]
    is_ar = hasattr(model.emission, "phi")

    if not is_ar:
        mu = model.emission.mu(0, ys, xs_std)                             # (S,)
        for m in range(1, K + 1):
            u = jnp.einsum("ai,aij->aj", u, Gammas[anchors + m])
        y_pred = u @ mu
    else:
        base = model.emission.mu_vals(0, ys, xs_std)                      # (S,) state base means
        phi = model.emission.phi()                                        # (k, S)
        k = phi.shape[0]
        if start < k - 1:
            raise ValueError(f"start={start} leaves no room for {k} AR lags")

        lags = jnp.stack([ys[anchors - j] for j in range(k)], axis=1)      # (A, k), col 0 = newest
        # mu_s = base_s + sum_j phi_js (y_{t-j} - base_s)
        #      = base_s (1 - sum_j phi_js) + sum_j phi_js y_{t-j}
        intercept = base[None, :] * (1.0 - phi.sum(axis=0))[None, :]       # (1, S)

        y_pred = None
        for m in range(1, K + 1):
            u = jnp.einsum("ai,aij->aj", u, Gammas[anchors + m])
            mu_state = intercept + jnp.einsum("aj,js->as", lags, phi)      # (A, S)
            y_pred = jnp.sum(u * mu_state, axis=1)                         # state-weighted mean
            lags = jnp.concatenate([y_pred[:, None], lags[:, :-1]], axis=1)  # plug-in next lag

    target_idx = np.asarray(anchors) + K
    return target_idx, np.asarray(ys[anchors + K]), np.asarray(y_pred)


def rolling_persistence(ys, start: int, end: int, K: int):
    """Persistence baseline: predict y[anchor + K] as the current value y[anchor]."""
    ys = jnp.asarray(ys)
    anchors = jnp.arange(start, end - K)
    target_idx = np.asarray(anchors) + K
    return target_idx, np.asarray(ys[anchors + K]), np.asarray(ys[anchors])


# --------------------------------------------------------------------------- #
# Diagnostics plotting
# --------------------------------------------------------------------------- #

def plot_hmm_diagnostics(model, save_path: str | None = None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import MaxNLocator
    from scipy import stats
    from statsmodels.graphics.tsaplots import plot_acf

    residuals = np.asarray(model.state_results.pseudo_residuals).ravel()
    residuals = residuals[np.isfinite(residuals)]

    sns.set_theme(style="whitegrid", context="notebook")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ll = np.asarray(model.ll_fits)
    iterations = np.arange(1, len(ll) + 1)
    sns.lineplot(x=iterations, y=ll, ax=axes[0], marker="o")
    axes[0].set_title("Log-likelihood per iteration")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("log L")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].get_lines()[0].set_color(sns.color_palette()[0])
    axes[1].get_lines()[1].set_color(sns.color_palette()[3])
    axes[1].set_title("Normal Q-Q of pseudo-residuals")

    lags = min(40, max(1, len(residuals) // 4))
    plot_acf(residuals, lags=lags, ax=axes[2])
    axes[2].set_title("ACF of pseudo-residuals")
    axes[2].set_ylim(-0.25, 1.05)

    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)

    return fig


# --------------------------------------------------------------------------- #
# LaTeX table rendering
# --------------------------------------------------------------------------- #

_LATEX_HEADER_MAP = {
    "#Params": "\\#Params",
    "ΔAIC": "$\\Delta$AIC",
    "ΔBIC": "$\\Delta$BIC",
    "P-val": "$p$-value",
}


def _fmt_cell(value, col, float_cols_4dp):
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}" if col in float_cols_4dp else f"{value:.2f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value) + 1)
    return str(value)


def write_latex_table(df, path: str, caption: str, label: str, float_cols_4dp=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    float_cols_4dp = set(float_cols_4dp or [])
    col_spec = "l" + "c" * (len(df.columns) - 1)
    header = " & ".join(_LATEX_HEADER_MAP.get(c, c) for c in df.columns) + " \\\\"
    body_rows = []
    for _, row in df.iterrows():
        cells = [_fmt_cell(row[col], col, float_cols_4dp) for col in df.columns]
        body_rows.append(" & ".join(cells) + " \\\\")
    tex = (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\hline\n"
        f"{header}\n"
        "\\hline\n"
        + "\n".join(body_rows) + "\n"
        "\\hline\n"
        "\\end{tabular}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )
    with open(path, "w") as f:
        f.write(tex)


if __name__ == "__main__":
    print(load_train_data("b1", "train-test"))