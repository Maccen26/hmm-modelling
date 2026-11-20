import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv


def load_y_df(data_name: str, tag: str, arr_type: str) -> pd.DataFrame:
    """Headered y CSV written by `clean_room_data_anchor`, with `datetime` parsed."""
    base_path = load_base_data_path()
    y_path = os.path.join(base_path, f"{data_name}/{tag}/y_{arr_type}.csv")
    df = pd.read_csv(y_path)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df

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


# --------------------------------------------------------------------------- #
# Covariate construction
#
# Used by `drivers.models.utils.covariates` to build the transition covariates.
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
    """Match hourly weather (UTC) to each tz-naive observation time, which is also UTC."""
    if not weather_cols:
        return np.empty((len(datetimes), 0))

    weather_df = pd.read_csv(weather_path(), parse_dates=["DateFrom", "DateTo"])
    weather_df = (
        weather_df[["DateFrom", *weather_cols]]
        .dropna()
        .sort_values("DateFrom")
        .reset_index(drop=True)
    )

    left = pd.DataFrame({"t": pd.DatetimeIndex(datetimes).tz_localize("UTC")})
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


# --------------------------------------------------------------------------- #
# Forecast metrics
# --------------------------------------------------------------------------- #

def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)


# --------------------------------------------------------------------------- #
# LaTeX table rendering
# --------------------------------------------------------------------------- #

_LATEX_HEADER_MAP = {
    "#Params": "\\#Params",
    # A bare % opens a LaTeX comment, which would swallow the rest of the header row.
    "MAPE%": "MAPE (\\%)",
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
        return str(int(value))
    return str(value)


def write_latex_table(df, path: str, caption: str, label: str, float_cols_4dp=None, max_width=False):
    """`max_width` shrinks the tabular to `\\textwidth` when wider (needs `adjustbox` in the preamble)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    float_cols_4dp = set(float_cols_4dp or [])
    col_spec = "l" + "c" * (len(df.columns) - 1)
    header = " & ".join(_LATEX_HEADER_MAP.get(c, c.replace("%", "\\%")) for c in df.columns) + " \\\\"
    body_rows = []
    for _, row in df.iterrows():
        cells = [_fmt_cell(row[col], col, float_cols_4dp) for col in df.columns]
        body_rows.append(" & ".join(cells) + " \\\\")
    tex = (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        + ("\\begin{adjustbox}{max width=\\textwidth}\n" if max_width else "")
        + f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\hline\n"
        f"{header}\n"
        "\\hline\n"
        + "\n".join(body_rows) + "\n"
        "\\hline\n"
        "\\end{tabular}\n"
        + ("\\end{adjustbox}\n" if max_width else "")
        + f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )
    with open(path, "w") as f:
        f.write(tex)


THESIS_TABLES_DIR = Path(__file__).resolve().parent.parent / "Thesis" / "Tables"


def save_thesis_table(
    df: pd.DataFrame, name: str, caption: str, label: str, float_cols_4dp=None, max_width=False,
) -> str:
    """Write `df` as a LaTeX table to `Thesis/Tables/<name>.tex`; include it with `\\input{Tables/<name>}`."""
    path = str(THESIS_TABLES_DIR / f"{name}.tex")
    write_latex_table(df, path, caption=caption, label=label, float_cols_4dp=float_cols_4dp, max_width=max_width)
    return path

