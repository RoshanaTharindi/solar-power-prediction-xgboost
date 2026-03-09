import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

from .utils import load_config, time_split, regression_metrics

# Saves a time series plot - Actual vs Predicted Irradiance over time
def plot_timeseries(ts, y, yhat, outpath, title):
    plt.figure()
    if len(y) > 2500: # limited to 2500 - hourly data across years is huge
        idx = np.linspace(0, len(y)-1, 2500).astype(int)
        ts, y, yhat = ts.iloc[idx], y.iloc[idx], yhat[idx]
    plt.plot(ts, y, label="Actual")
    plt.plot(ts, yhat, label="Predicted")
    plt.xlabel("Time"); plt.ylabel("Irradiance (W/m²)")
    plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()
    
# Saves a Predicted vs Actual scatter plot
def plot_scatter(y, yhat, outpath, title):
    plt.figure()
    plt.scatter(y, yhat, s=8, alpha=0.4)
    plt.xlabel("Actual"); plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model_in", required=True)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--artifacts_dir", default="reports")
    args = ap.parse_args()

    cfg = load_config(args.config)
    target = cfg["target"]

    df = pd.read_csv(args.data, parse_dates=["timestamp"]).sort_values("timestamp")
    train_m, val_m, test_m = time_split(df, **cfg["splits"])
    feature_cols = [c for c in df.columns if c not in ["timestamp", target]]

    model = XGBRegressor()
    model.load_model(args.model_in)

    def eval_split(mask):
        X = df.loc[mask, feature_cols]
        y = df.loc[mask, target]
        ts = df.loc[mask, "timestamp"]
        yhat = model.predict(X)
        return ts, y, yhat, regression_metrics(y, yhat)

    results = {}
    artifacts = Path(args.artifacts_dir)
    figdir = artifacts / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    for name, mask in [("train", train_m), ("val", val_m), ("test", test_m)]:
        ts, y, yhat, m = eval_split(mask)
        results[name] = m
        plot_timeseries(ts, y, yhat, figdir / f"{name}_timeseries.png", f"{name.upper()} Actual vs Predicted")
        plot_scatter(y, yhat, figdir / f"{name}_scatter.png", f"{name.upper()} Predicted vs Actual")

    # Save metrics in a JSON file
    (artifacts / "evaluation_metrics.json").write_text(json.dumps(results, indent=2))
    print(results)

if __name__ == "__main__":
    main()
