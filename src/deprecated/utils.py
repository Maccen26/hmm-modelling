## LOAD DATA AS IN EXAMPLE USAGE 
import pandas as pd 
from dotenv import load_dotenv 
import os 
import matplotlib.pyplot as plt
import numpy as np



def load_and_aggregate(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")

    # Drop rows where WindowClosed is NaN and filter to Bedroom
    df = df[df["WindowClosed"].notna()]
    df = df[df["Room"] == "Bedroom"]

    # Take first 5000 rows
    df = df.iloc[:5000]

    # Compute HalfHour: (day + Time) * 24 * 2, rounded to nearest int
    df["HalfHour"] = ((df["day"] + df["Time"]) * 24 * 2).round().astype(int)

    # Aggregate: mean of these columns, grouped by HalfHour
    agg = df.groupby("HalfHour")[["CO2C", "WindowClosed", "Time", "Day", "Month"]].mean().reset_index()

    # HalfHour mod 48 (to get position within day)
    agg["HalfHour"] = agg["HalfHour"] % 48

    # Hour of day
    agg["Hour"] = agg["HalfHour"] / 2

    # Time as fraction of days elapsed
    agg["Time"] = (pd.RangeIndex(1, len(agg) + 1)) / 2 / 24

    return agg 



def load_data_path(file) -> str:
    load_dotenv()  # Load environment variables from .env file
    return os.path.join(os.getenv("DATA_PATH"), file)  # Get the DATA_PATH variable and join with file name



def plot_filtered_states(df, u_norm):
    """
    df     : aggregated dataframe with Time and Hour columns
    u_norm : (T, num_states) array of filtered state probabilities
    """
    u_norm_np = np.array(u_norm)  # convert from jnp to numpy for matplotlib

    # x-axis: Time + Hour[0]/24, replicating R's win1.agr$Time + win1.agr$Hour[1]/24
    x = df["Time"].values + df["Hour"].iloc[0] / 24

    # Cumulative sums for stacking polygons
    cum1 = u_norm_np[:, 0]                        # state 0
    cum2 = u_norm_np[:, 0] + u_norm_np[:, 1]      # state 0 + 1
    ones = np.ones(len(x))                         # top = 1

    col = ["#000000", "#FF4949", "#69E989"]        # gray(c(0, 0.5, 0.9)) in R

    fig, ax = plt.subplots(figsize=(10, 5))

    # State 0: from 0 up to u_norm[:, 0]
    ax.fill_between(x, 0,    cum1, color=col[0], label="State 1")

    # State 1: from u_norm[:, 0] up to u_norm[:, 0] + u_norm[:, 1]
    ax.fill_between(x, cum1, cum2, color=col[1], label="State 2")

    # State 2: from cum2 up to 1
    ax.fill_between(x, cum2, ones, color=col[2], label="State 3")

    ax.set_xlabel("Time [days]")
    ax.set_ylabel(r"$\hat{u}_{t|t}$")
    ax.set_title("Filtered state probabilities")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

