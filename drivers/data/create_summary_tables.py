"""Summary tables of the raw DTU CO2 and feedback data, ported from `summary_statistics.ipynb`.

Each table is printed to the terminal and written as LaTeX to `Thesis/Tables/data/<name>.tex`
(include with `\\input{Tables/data/<name>}`). Only rooms with both a CO2 series and a feedback file
are used, as in the notebook.

    uv run python -m drivers.data.create_summary_tables --gap-room 15
"""

import argparse
import os
from pathlib import Path

import pandas as pd

from drivers.data.clean_room_data import BIN, clean_df
from drivers.utils import load_base_data_path, save_thesis_table

DEDUP_WINDOW = pd.Timedelta("60min")
STEP = pd.Timedelta(BIN)
TABLES_SUBDIR = "data"  # tables go to `Thesis/Tables/data/`


# --- Load ---------------------------------------------------------------

def raw_dir() -> Path:
    return Path(load_base_data_path()) / "raw" / "dtu"


def room_names() -> list[str]:
    """Rooms with both a CO2 series and a feedback file, e.g. `Room 015`."""
    series = {f.removesuffix(".csv") for f in os.listdir(raw_dir() / "timeseries") if "Room" in f}
    feedback = {f.removesuffix(".csv") for f in os.listdir(raw_dir() / "feedback")}
    return sorted(series & feedback)


def load_raw(room_name: str) -> pd.DataFrame:
    df = pd.read_csv(raw_dir() / "timeseries" / f"{room_name}.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def load_feedback(room_name: str) -> pd.DataFrame:
    df = pd.read_csv(raw_dir() / "feedback" / f"{room_name}.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.drop_duplicates().dropna(subset=["value"])


def filter_repeat_complaints(df: pd.DataFrame) -> pd.DataFrame:
    """Keep a complaint only if the same user has not made a kept complaint
    in the same category within `DEDUP_WINDOW` before it."""
    keep = pd.Series(False, index=df.index)
    for _, group in df.sort_values("datetime").groupby(["userId", "value"]):
        last_kept = None
        for idx, t in group["datetime"].items():
            if last_kept is None or t - last_kept > DEDUP_WINDOW:
                keep[idx] = True
                last_kept = t
    return df[keep].sort_values("datetime")


def load_all() -> tuple[dict, dict, dict, dict]:
    """Raw, cleaned, feedback and deduplicated feedback frames keyed by room name."""
    rooms = room_names()
    raw = {r: load_raw(r) for r in rooms}
    clean = {r: clean_df(df) for r, df in raw.items()}
    feedback = {r: load_feedback(r) for r in rooms}
    valid = {r: filter_repeat_complaints(df) for r, df in feedback.items()}
    return raw, clean, feedback, valid


def binned_co2(clean: pd.DataFrame) -> pd.Series:
    """`BIN` means of the cleaned readings, empty bins left as NaN."""
    return clean.set_index("datetime")["co2"].resample(BIN).mean()


# --- Tables -------------------------------------------------------------

def option_label(value: str) -> str:
    """`radio_0_opt_2` -> `Opt 2` (underscores would break LaTeX)."""
    return f"Opt {value.rsplit('_', 1)[-1]}"


def summarise(x: pd.Series) -> pd.Series:
    return pd.Series({"N": x.count(), "Min": x.min(), "Q1": x.quantile(0.25), "Median": x.median(),
                      "Mean": x.mean(), "Q3": x.quantile(0.75), "Max": x.max()})


def removed_table(raw: dict, clean: dict) -> pd.DataFrame:
    table = pd.DataFrame({r: {"Raw": len(raw[r]), "Clean": len(clean[r])} for r in raw}).T
    table["Removed"] = table["Raw"] - table["Clean"]
    return table


def feedback_table(feedback: dict) -> pd.DataFrame:
    counts = pd.DataFrame({r: df["value"].value_counts() for r, df in feedback.items()}).T
    counts = counts.fillna(0).astype(int).sort_index(axis=1).rename(columns=option_label)
    counts.insert(0, "Complaints", counts.sum(axis=1))
    return counts


def valid_feedback_table(feedback: dict, valid: dict) -> pd.DataFrame:
    counts = feedback_table(feedback)
    valid_counts = feedback_table(valid).reindex(columns=counts.columns, fill_value=0)
    table = pd.DataFrame({"Complaints": counts["Complaints"], "Valid": valid_counts["Complaints"]})
    table["Removed"] = table["Complaints"] - table["Valid"]
    table = table.join(valid_counts.drop(columns="Complaints").add_prefix("Valid "))
    table.loc["Total"] = table.sum()
    return table.astype(int)


def co2_table(clean: dict, valid: dict) -> pd.DataFrame:
    table = pd.DataFrame({r: summarise(df["co2"]) for r, df in clean.items()}).T
    table["N"] = table["N"].astype(int)
    return table.join(feedback_table(valid)["Complaints"].rename("Valid complaints"))


def waiting_time_table(clean: dict) -> pd.DataFrame:
    """Seconds between consecutive cleaned readings."""
    table = pd.DataFrame(
        {r: summarise(df["datetime"].diff().dropna().dt.total_seconds()) for r, df in clean.items()}
    ).T
    table["N"] = table["N"].astype(int)
    return table


def gap_table(clean: pd.DataFrame, top: int | None = 10) -> tuple[pd.DataFrame, pd.Series]:
    """The `top` longest runs of empty `BIN` bins, with the CO2 level on either side."""
    binned = binned_co2(clean)
    missing = binned.isna()
    gap_id = (missing != missing.shift()).cumsum()[missing]
    gaps = binned.index.to_series()[missing].groupby(gap_id).agg(Start="first", End="last")
    gaps["Bins"] = gap_id.groupby(gap_id).size()
    gaps["Duration"] = (gaps["End"] - gaps["Start"] + STEP).dt.total_seconds() / 3600
    gaps["CO2 before"] = binned.shift(1).loc[gaps["Start"]].values
    gaps["CO2 after"] = binned.shift(-1).loc[gaps["End"]].values
    gaps = gaps.sort_values("Bins", ascending=False)
    return (gaps.head(top) if top else gaps), missing


def longest_complete_run(clean: pd.DataFrame) -> pd.Series:
    """Longest stretch of `BIN` means without an empty bin."""
    binned = binned_co2(clean)
    valid = binned.notna()
    run_id = (~valid).cumsum()[valid]
    return binned[valid][run_id == run_id.value_counts().idxmax()]


def longest_run_table(clean: dict, feedback: dict, valid: dict) -> pd.DataFrame:
    rows = {}
    for r, df in clean.items():
        run = longest_complete_run(df)
        start, end = run.index[0], run.index[-1]
        in_window = lambda t: int(((t >= start) & (t < end + STEP)).sum())
        rows[r] = {
            "Start": start.strftime("%Y-%m-%d %H:%M"),
            "End": end.strftime("%Y-%m-%d %H:%M"),
            "Bins": len(run),
            "Complaints": in_window(feedback[r]["datetime"]),
            "Valid": in_window(valid[r]["datetime"]),
            "Removed": len(df) - in_window(df["datetime"]),
        }
    return pd.DataFrame(rows).T


# --- Output -------------------------------------------------------------

def emit(table: pd.DataFrame, name: str, caption: str, index_name: str = "Room") -> None:
    print(f"\n=== {name} ===")
    print(table.round(1).to_string())
    out = table.rename_axis(index_name).reset_index()
    for col in out.select_dtypes("float").columns:
        out[col] = out[col].round(1)
    path = save_thesis_table(out, name=f"{TABLES_SUBDIR}/{name}", caption=caption, label=f"tab:{name}",
                             max_width=True)
    print(f"wrote {path}")


def create_summary_tables(gap_room: str) -> None:
    raw, clean, feedback, valid = load_all()

    emit(removed_table(raw, clean), "raw_removed_readings",
         "Number of raw CO$_2$ readings per room before and after removing duplicate and missing readings.")
    emit(feedback_table(feedback), "feedback_counts",
         "Number of occupant complaints per room, in total and per answer option.")
    emit(valid_feedback_table(feedback, valid), "valid_feedback_counts",
         f"Complaints per room after removing repeats: a complaint is kept only if the same user has not "
         f"made a kept complaint in the same category within the preceding "
         f"{int(DEDUP_WINDOW.total_seconds() // 60)} minutes.")
    emit(co2_table(clean, valid), "raw_co2_quantiles",
         "Summary statistics of the cleaned raw CO$_2$ readings (ppm) per room, with the number of valid (deduplicated) complaints.")
    emit(waiting_time_table(clean), "raw_waiting_times",
         "Summary statistics of the waiting time (seconds) between consecutive cleaned CO$_2$ readings.")

    gaps, missing = gap_table(clean[gap_room])
    print(f"\n{gap_room}: {missing.sum()} of {len(missing)} bins missing ({missing.mean():.1%})")
    gaps = gaps.assign(Start=gaps["Start"].dt.strftime("%Y-%m-%d %H:%M"),
                       End=gaps["End"].dt.strftime("%Y-%m-%d %H:%M"))
    gaps.index = range(1, len(gaps) + 1)
    emit(gaps.rename(columns={"Duration": "Duration [h]"}),
         f"{gap_room.lower().replace(' ', '_')}_gaps",
         f"The ten longest gaps of empty {BIN} bins in {gap_room}, with the mean CO$_2$ (ppm) in the bin "
         f"just before and just after each gap.", index_name="Rank")

    emit(longest_run_table(clean, feedback, valid), "longest_complete_run",
         f"Longest stretch of {BIN} bins without missing data per room, with the complaints falling inside it "
         f"and the raw readings outside it.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gap-room", type=int, default=15, help="room for the gap table, e.g. 15 -> 'Room 015'")
    args = parser.parse_args()

    create_summary_tables(f"Room {args.gap_room:03d}")
