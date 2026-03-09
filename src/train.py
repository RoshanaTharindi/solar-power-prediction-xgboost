import argparse, json # saves metrics/hyperparameters as a JSON file
from pathlib import Path
import pandas as pd
from xgboost import XGBRegressor # The model we train

from .utils import load_config, time_split, regression_metrics 
# time_split - creates train/val/test marks based on timestamps
# regression_metrics - computes MAE, RMSE, R2, MAPE

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)      # processed CSV
    ap.add_argument("--model_out", required=True) # models/xgb_model.json
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--artifacts_dir", default="reports")
    args = ap.parse_args()

    cfg = load_config(args.config)
    target = cfg["target"]

    df = pd.read_csv(args.data, parse_dates=["timestamp"]).sort_values("timestamp")
    # Create time-based train/val/test splits
    train_m, val_m, test_m = time_split(df, **cfg["splits"])
    # Choose input features - ensures model gets only feature columns
    feature_cols = [c for c in df.columns if c not in ["timestamp", target]]

    # build train and validation sets
    X_train, y_train = df.loc[train_m, feature_cols], df.loc[train_m, target]
    X_val, y_val     = df.loc[val_m, feature_cols], df.loc[val_m, target]

    # Create and train XGBoost model
    model = XGBRegressor(**cfg["xgb_params"])
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=30, # if validation performance doesn't improve for 30 boosting rounds, training stops - reduces overfitting and saves time
        verbose=False
    )

    # Created a dictionary containing train metrics(MAE, RMSE, R2, MAPE), validation metrics, hyperparameters used, dataset split sizes and no of features
    metrics = {
        "train": regression_metrics(y_train, model.predict(X_train)),
        "val": regression_metrics(y_val, model.predict(X_val)),
        "hyperparameters": cfg["xgb_params"],
        "n_train": int(train_m.sum()),
        "n_val": int(val_m.sum()),
        "n_test": int(test_m.sum()),
        "feature_count": len(feature_cols)
    }

    # Save metrics to a JSON file
    artifacts = Path(args.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "train_metrics.json").write_text(json.dumps(metrics, indent=2))

    # Save the trained model
    out = Path(args.model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out))

    print("Validation metrics:", metrics["val"])
    print("Saved model:", out)

if __name__ == "__main__":
    main()
