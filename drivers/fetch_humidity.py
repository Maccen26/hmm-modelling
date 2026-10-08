"""Download hourly relative humidity from DMI's 10 km climate grid for a coordinate.

The coordinate is mapped to its DMI grid cell (ETRS89 / UTM 32N), the hourly
`mean_relative_hum` series is fetched from the DMI open-data API, de-duplicated,
resampled to a regular hourly grid, and written to `{DATA_PATH}/raw/dtu/humidity.csv`.
"""

from pathlib import Path

import pandas as pd
import requests
from pyproj import Transformer

from drivers.utils import load_base_data_path

DMI_URL = "https://opendataapi.dmi.dk/v2/climateData/collections/10kmGridValue/items"


def find_grid_name(lat: float, lon: float) -> str:
    """DMI 10 km cell id (`10km_<north>_<east>`) containing the WGS84 coordinate."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
    east, north = transformer.transform(lon, lat)
    return f"10km_{int(north / 10000)}_{int(east / 10000)}"


def fetch_weather(parameter: str, cell: str, start: str, end: str) -> pd.Series:
    """Hourly series of one DMI grid parameter, indexed by the interval end time (UTC)."""
    params = {
        "cellId": cell,
        "limit": 300000,
        "timeResolution": "hour",
        "parameterId": parameter,
        "datetime": f"{start}/{end}",
    }
    response = requests.get(DMI_URL, params=params, timeout=60)
    response.raise_for_status()
    json_data = response.json()

    if "features" not in json_data:
        raise RuntimeError(f"No weather data returned for {parameter}")
    data = pd.json_normalize(json_data["features"])
    if data.empty:
        raise RuntimeError(f"Empty weather dataset for {parameter}")

    data["time"] = pd.to_datetime(data["properties.to"], utc=True, errors="coerce")
    data = data.dropna(subset=["time"])
    return data.set_index("time")["properties.value"].astype(float).sort_index()


def fetch_humidity(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Regular hourly relative humidity (%) for the grid cell containing (lat, lon)."""
    cell = find_grid_name(lat, lon)
    print("DMI grid:", cell)
    print("Start:", start)
    print("End:", end)

    humidity = fetch_weather("mean_relative_hum", cell, start, end)
    df = pd.DataFrame({"mean_relative_hum": humidity})

    # Remove duplicate timestamps, then fill gaps on a regular hourly grid
    df = df.sort_index().groupby(level=0).mean()
    df = df.resample("1h").mean().interpolate(method="time").ffill().bfill()
    df.index.name = "DateTo"
    return df


def save_humidity(df: pd.DataFrame, file_name: str = "humidity.csv") -> Path:
    path = Path(load_base_data_path()) / "raw" / "dtu" / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return path


if __name__ == "__main__":
    LAT = 55.78440496059041   # DTU Lyngby campus
    LON = 12.518867022602029

    START = "2023-11-12T00:00:00.000Z"
    END = "2024-02-10T23:00:00.000Z"

    print("\nDownloading humidity data...")
    df_humidity = fetch_humidity(LAT, LON, START, END)
    path = save_humidity(df_humidity)
    print(f"Saved {len(df_humidity)} hourly rows to {path}")
