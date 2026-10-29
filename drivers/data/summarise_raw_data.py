"""Summary statistics of each room's raw CO2 readings, written as a LaTeX table to `Thesis/Tables/`.

Raw readings are those from `{DATA_PATH}/raw/dtu/timeseries/<room>.csv` after the NaN / range /
duplicate filter in `clean_dtu_data.load_raw_series` -- the same series `plot_room_data` plots.

    uv run python -m drivers.data.summarise_raw_data
"""

import pandas as pd

from drivers.data.clean_dtu_data import load_raw_series
from drivers.utils import save_thesis_table

ROOM_LIST = [
    "Room 001",
    "Room 003",
    "Room 004",
    "Room 005",
    "Room 007",
    "Room 008",
    "Room 009",
    "Room 012",
    "Room 013",
    "Room 014",
    "Room 015",
    "Room 016",
    "Room 019",
]


def summarise(co2: pd.Series) -> dict:
    return {
        "N": len(co2),
        "Min": co2.min(),
        "2.5%": co2.quantile(0.025),
        "Median": co2.median(),
        "Mean": co2.mean(),
        "97.5%": co2.quantile(0.975),
        "Max": co2.max(),
    }


def raw_summary_table(rooms: list[str]) -> pd.DataFrame:
    rows = []
    for room_name in rooms:
        df_raw, _ = load_raw_series(room_name, y_cols=("co2",))
        rows.append({"Room": room_name, **summarise(df_raw["co2"].astype(float))})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    table = raw_summary_table(ROOM_LIST)
    print(table.round(1).to_string(index=False))

    path = save_thesis_table(
        table,
        name="raw_co2_summary",
        caption="Summary statistics of the raw CO$_2$ readings (ppm) per room, after removing missing, "
                "saturated and duplicate readings.",
        label="tab:raw_co2_summary",
    )
    print(f"wrote {path}")
