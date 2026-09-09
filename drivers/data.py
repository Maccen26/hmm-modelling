import os 
from dotenv import load_dotenv
import pandas as pd

def create_train_test_data(name: str, train_size: float = 0.8): 
    load_dotenv()

    PATH = load_data_path(name=name)  
    df = load_df(data_path=PATH) 
    df = aggregate_df(df=df) 
    df_train, df_test = split_df(df=df, train_size=train_size) 
    for df_name, df in zip(["train", "test"], [df_train, df_test]): 
        save_df(df=df, data_name=name, df_name=df_name) 

def load_data_path(name: str) -> str:
    base_path = load_base_data_path() 
    return os.path.join(base_path, name) 

def load_base_data_path() -> str:
    load_dotenv()
    PATH = os.getenv("DATA_PATH")
    if (PATH is None):
        raise ValueError("DATA_PATH environment variable is not set.")
    return PATH

def load_df(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path, sep=";")
    return df


def aggregate_df(df:pd.DataFrame) -> pd.DataFrame:

    df = df[df["WindowClosed"].notna()]
    df = df[df["Room"] == "Bedroom"]
    df["day"] = df["day"] - 74  # Shift day so that day 0 corresponds to the first day in the dataset 

    # Compute HalfHour: (day + Time) * 24 * 2, rounded to nearest int
    df["HalfHour"] = ((df["day"] + df["Time"]) * 24 * 2).round().astype(int)

    # Aggregate: mean of these columns, grouped by HalfHour
    agg = df.groupby("HalfHour")[["CO2C", "WindowClosed", "Time", "Day", "Month", "Hour"]].mean().reset_index()

    # HalfHour mod 48 (to get position within day)
    agg["HalfHour"] = agg["HalfHour"] % 48

    # Time as fraction of days elapsed
    agg["Time"] = (pd.RangeIndex(1, len(agg) + 1)) / 2 / 24

    return agg 

def split_df(df: pd.DataFrame, train_size: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_len = int(len(df) * train_size)
    df_train = df.iloc[:train_len]
    df_test = df.iloc[train_len:]
    return df_train, df_test

def save_df(df: pd.DataFrame, data_name: str, df_name: str) -> None:
    base_path = load_base_data_path() 
    data_name = data_name.split(".")[0]  # Remove file extension if present
    save_path = os.path.join(base_path, f"{data_name}/{data_name}_{df_name}.csv") 
    buld_dir = os.path.dirname(save_path)
    if not os.path.exists(buld_dir):
        os.makedirs(buld_dir)
    df.to_csv(save_path, sep=";", index=False)



if __name__ == "__main__":
    create_train_test_data(name="b1.csv", train_size=0.8)

