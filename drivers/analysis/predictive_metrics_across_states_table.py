"""LaTeX table of the forecast RMSE / MAPE of every model for every number of states.

Reads the per-K metrics CSVs under one results folder (`states_{K}/val_metrics.csv`, or
`predictive_metrics.csv` for the plain fits; `test_metrics.csv` with `--split test`) and writes one
row per K and one `RMSE / MAPE` column per model to `Thesis/Tables/<name>.tex`.

    uv run python -m drivers.analysis.predictive_metrics_across_states_table \
        results/room_015/calibrated_anchor/random_search --horizon 1 [--split val|test] [--name ...]
"""
import argparse
import os
import re

import pandas as pd

from drivers.models.run import FIT_ORDER
from drivers.utils import save_thesis_table

SPLIT_FILES = {"val": ("val_metrics.csv", "predictive_metrics.csv"), "test": ("test_metrics.csv",)}
SPLIT_NAMES = {"val": "validation", "test": "test"}


def states_dirs(results_dir: str) -> dict[int, str]:
    """`{K: path}` of every `states_{K}` folder, sorted by K."""
    found = {int(m.group(1)): os.path.join(results_dir, d)
             for d in os.listdir(results_dir) if (m := re.fullmatch(r"states_(\d+)", d))}
    return dict(sorted(found.items()))


def load_metrics(results_dir: str, split: str) -> pd.DataFrame:
    """All metrics rows of the split, with a `K` column."""
    frames = []
    for k, path in states_dirs(results_dir).items():
        csv = next((os.path.join(path, f) for f in SPLIT_FILES[split] if os.path.exists(os.path.join(path, f))), None)
        if csv is not None:
            frames.append(pd.read_csv(csv).assign(K=k))
    if not frames:
        raise SystemExit(f"No {split} metrics found under {results_dir}")
    return pd.concat(frames, ignore_index=True)


def model_order(df: pd.DataFrame) -> list[str]:
    """Model labels in fit order; slugs outside FIT_ORDER go last."""
    rank = {slug: i for i, slug in enumerate(FIT_ORDER)}
    slugs = sorted(df["slug"].unique(), key=lambda s: (rank.get(s, len(rank)), s))
    labels = df.drop_duplicates("slug").set_index("slug")["model"]
    return [labels[s] for s in slugs]


def metrics_table(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Rows are K, columns are models, cells are `RMSE / MAPE`; a missing fit is `--`."""
    df = df[df["horizon"] == horizon]
    if df.empty:
        raise SystemExit(f"No rows for horizon {horizon}")
    cells = df.assign(cell=[f"{r:.2f} / {m:.2f}" for r, m in zip(df["RMSE"], df["MAPE"])])
    wide = cells.pivot(index="K", columns="model", values="cell")[model_order(df)].fillna("--")
    return wide.reset_index().rename(columns={"K": "States"})


def default_name(results_dir: str, split: str, horizon: int) -> str:
    """e.g. `predictive_metrics_room_015_calibrated_anchor_random_search_val_h1`."""
    parts = os.path.normpath(results_dir).split(os.sep)
    parts = parts[parts.index("results") + 1:] if "results" in parts else parts[-3:]
    return "_".join(["predictive_metrics", *parts, split, f"h{horizon}"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_dir", help="folder holding the states_{K}/ subfolders")
    parser.add_argument("--horizon", "-H", type=int, default=1, help="forecast horizon in steps (default 1)")
    parser.add_argument("--split", choices=SPLIT_FILES, default="val")
    parser.add_argument("--name", help="table file name in Thesis/Tables/ (default: from the folder, split and horizon)")
    args = parser.parse_args()

    table = metrics_table(load_metrics(args.results_dir, args.split), args.horizon)
    print(table.to_string(index=False))

    name = args.name or default_name(args.results_dir, args.split, args.horizon)
    steps = "step" if args.horizon == 1 else "steps"
    path = save_thesis_table(
        table,
        name=name,
        caption=f"{SPLIT_NAMES[args.split].capitalize()} RMSE (ppm) / MAPE (\\%) of the {args.horizon}-{steps}-ahead "
                f"forecasts for each model and number of states.",
        label=f"tab:{name}",
        max_width=True,
    )
    print(f"wrote {path}")
