import json
from pathlib import Path
import numpy as np
import pandas as pd

def load_config(path="config.json"):
    return json.loads(Path(path).read_text())

def time_split(df: pd.DataFrame, train_end: str, val_end: str, test_end: str):
    ts = pd.to_datetime(df["timestamp"])
    train = ts <= pd.to_datetime(train_end)
    val = (ts > pd.to_datetime(train_end)) & (ts <= pd.to_datetime(val_end))
    test = (ts > pd.to_datetime(val_end)) & (ts <= pd.to_datetime(test_end))
    return train, val, test

def regression_metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    eps = 1e-6
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0)

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE_%": mape}
