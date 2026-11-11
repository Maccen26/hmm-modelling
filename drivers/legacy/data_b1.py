import os 
from dotenv import load_dotenv
import pandas as pd
from typing import Callable
import numpy as np
from drivers.utils import load_base_data_path

def create_train_test_data(
        name: list|str, 
        tag: str, 
        train_size: float = 0.8, 
        data_func: Callable|None = None,
        data_path: str|None = None
        ): 
    
    if (isinstance(name, str)):
        name = [name]  # Convert to list for uniform processing

    for n in name:
        create_single_train_test_data(name=n, data_path=data_path, tag=tag, train_size=train_size, data_func=data_func)

def create_single_train_test_data(
        name: str, 
        data_path: str|None,
        tag: str, 
        train_size: float = 0.8, 
        data_func: Callable|None = None
        ): 

    if (data_func is None):
        raise ValueError("data_func must be provided to create_b1_train_test_data.")
    load_dotenv()
    PATH = load_data_path(name=name, data_path=data_path)  
    df = load_df(data_path=PATH) 
    ys, Xs = data_func(df=df, data_path=data_path)  # Use the provided data_func to process the DataFrame into ys and Xs
    meta = {}
    for arr_name,arr in zip(["y", "X"],[ys, Xs]): 
        arr_train, arr_test = split_arr(arr=arr, train_size=train_size)
        meta[f"{arr_name}_train_shape"] = arr_train.shape
        meta[f"{arr_name}_test_shape"] = arr_test.shape
        save_arr(arr=arr_train, data_name=name, tag=tag, arr_name = arr_name, arr_type="train")
        save_arr(arr=arr_test, data_name=name, tag=tag, arr_name=arr_name, arr_type="test")
    df = build_metadata_df(ys=ys, Xs=Xs, name=name, tag=tag, train_size=train_size, meta=meta)

    base_path = load_base_data_path()
    data_name = name.split(".")[0]
    save_path = os.path.join(base_path, f"{data_name}/{tag}/metadata.csv") 
    df.to_csv(save_path, index=False, sep=",")

def load_data_path(name: str, data_path:str|None) -> str:
    if (data_path is not None):
        return os.path.join(data_path, name)
    base_path = load_base_data_path() 
    return os.path.join(base_path, f"raw/{name}") 



def load_df(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path, sep=";")
    return df


def split_arr(arr: np.ndarray, train_size: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    arr = np.atleast_2d(arr)  # Ensure arr is at least 2D
    n, m = arr.shape
    if (n < m): 
        print(f"Warning: arr has shape {arr.shape}, which is not expected. Transposing.")
        arr = arr.T  # Transpose if the number of rows is less than the number of columns
    if (train_size == 1): 
        return arr, np.array([])  # Return empty test set if train_size is 1
    train_len = int(len(arr) * train_size)
    arr_train = arr[:train_len, :]
    arr_test = arr[train_len:, :]
    return arr_train, arr_test

def save_arr(arr: np.ndarray, data_name: str, arr_name: str, arr_type: str, tag: str) -> None:
    if (len(arr) == 0): 
        print(f"Warning: {arr_name}_{arr_type} array is empty. Not saving.")
        return
    
    base_path = load_base_data_path() 
    data_name = data_name.split(".")[0]  # Remove file extension if present
    save_path = os.path.join(base_path, f"{data_name}/{tag}/{arr_name}_{arr_type}.csv") 
    buld_dir = os.path.dirname(save_path)
    if not os.path.exists(buld_dir):
        os.makedirs(buld_dir)
    np.savetxt(save_path, arr, delimiter=",")


def build_metadata_df(ys: np.ndarray, Xs: np.ndarray, name: str, tag: str, train_size: float, meta: dict) -> pd.DataFrame:
    metadata = {
        "data_name": name,
        "tag": tag,
        "train_size": train_size,
        "y_shape": ys.shape,
        "X_shape": Xs.shape,
    }
    metadata.update(meta)  # Add additional metadata from the meta dictionary
    df = pd.DataFrame([metadata])

    return df





def aggregate_b1(df:pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:

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

    ys = np.asarray(agg["CO2C"].values)
    t = np.asarray(agg["HalfHour"], dtype=float)
    cos = np.cos(2 * np.pi * t / 48)
    sin = np.sin(2 * np.pi * t / 48)
    Xs = np.column_stack((cos, sin))
    return ys, Xs


    






if __name__ == "__main__":
    create_train_test_data(
        name="b1.csv", 
        train_size=0.46047540077390825, 
        tag="test", 
        data_func=None
        )

