import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

from .utils import load_config, time_split

def feature_importance_gain(model, feature_cols, outpath):
    imp = model.get_booster().get_score(importance_type="gain")
    gains = np.array([imp.get(f, 0.0) for f in feature_cols], dtype=float)
    order = np.argsort(gains)[::-1][:20]

    plt.figure()
    plt.bar(range(len(order)), gains[order])
    plt.xticks(range(len(order)), [feature_cols[i] for i in order], rotation=90)
    plt.ylabel("Gain")
    plt.title("Top-20 Feature Importances (XGBoost Gain)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def pdp_1d(model, X, feature, outpath, grid_size=30):
    x = X[feature].values
    grid = np.linspace(np.quantile(x, 0.02), np.quantile(x, 0.98), grid_size)
    preds = []
    Xtmp = X.copy()
    for v in grid:
        Xtmp[feature] = v
        preds.append(model.predict(Xtmp).mean())

    plt.figure()
    plt.plot(grid, preds)
    plt.xlabel(feature)
    plt.ylabel("Mean prediction (W/m²)")
    plt.title(f"PDP: {feature}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

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

    artifacts = Path(args.artifacts_dir)
    figdir = artifacts / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    # Use validation set for explainability
    X_val = df.loc[val_m, feature_cols].sample(n=min(2000, int(val_m.sum())), random_state=42)

    # 1) Feature importance analysis
    feature_importance_gain(model, feature_cols, figdir / "feature_importance_gain.png")

    # 2) SHAP
    shap_available = False
    try:
        import shap
        shap_available = True
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val)

        plt.figure()
        shap.summary_plot(shap_values, X_val, show=False)
        plt.tight_layout()
        plt.savefig(figdir / "shap_summary.png", dpi=200, bbox_inches="tight")
        plt.close()

        # dependence plot for most influential feature
        top_idx = np.argmax(np.abs(shap_values).mean(axis=0))
        top_feature = X_val.columns[top_idx]

        plt.figure()
        shap.dependence_plot(top_feature, shap_values, X_val, show=False)
        plt.tight_layout()
        plt.savefig(figdir / "shap_dependence_top_feature.png", dpi=200, bbox_inches="tight")
        plt.close()

    except Exception as e:
        top_feature = None

    # 3) PDP (domain-aligned)
    for f in ["hour", "T2M", "RH2M"]:
        if f in X_val.columns:
            pdp_1d(model, X_val.copy(), f, figdir / f"pdp_{f}.png")

    (artifacts / "explainability_summary.json").write_text(json.dumps({
        "shap_available": shap_available,
        "top_feature_from_shap": top_feature
    }, indent=2))

    print("Explainability done. SHAP:", shap_available)

if __name__ == "__main__":
    main()
