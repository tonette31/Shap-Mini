from __future__ import annotations
import argparse, os, json
import numpy as np, pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from utils import save_json, load_columns

def load_data_for_shap(feature_cols, data_path="data/train.csv"):
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        if "target" not in df.columns:
            raise ValueError("Expected 'target' column in data/train.csv")
        X = df[feature_cols]
    else:
        n = 2000
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.normal(size=(n, len(feature_cols))), columns=feature_cols)
    return X

def normalize_shap_values(shap_values):
    """Return 2D array (n_samples, n_features) regardless of SHAP version/model.
    Handles: list-of-arrays (per-class), 2D arrays, and 3D arrays (n_samples, n_features, n_classes).
    Picks the positive class if multi-class/binary.
    """
    import numpy as np
    if isinstance(shap_values, list):
        # e.g., [class0, class1, ...]
        return shap_values[1] if len(shap_values) > 1 else shap_values[0]
    arr = np.asarray(shap_values)
    if arr.ndim == 3:  # (n_samples, n_features, n_classes)
        idx = 1 if arr.shape[-1] > 1 else 0
        return arr[:, :, idx]
    return arr  # assume 2D

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rf", choices=["rf","logreg"])
    ap.add_argument("--feature", default=None, help="feature name for dependence plot")
    args = ap.parse_args()

    model_path = os.path.join("models", f"{args.model}.pkl")
    cols_path = os.path.join("outputs", "train_columns.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Train first: python train.py --model {args.model}")
    if not os.path.exists(cols_path):
        raise FileNotFoundError(f"Columns not found: {cols_path}. Train first to export columns.")

    feature_cols = load_columns(cols_path)
    model = joblib.load(model_path)
    X = load_data_for_shap(feature_cols)

    clf = model.named_steps["clf"]

    os.makedirs("outputs", exist_ok=True)

    if args.model == "rf":
        explainer = shap.TreeExplainer(clf)
        X_sample = X.sample(n=min(200, len(X)), random_state=42)
        shap_values = explainer.shap_values(X_sample)
        sv = normalize_shap_values(shap_values)
    elif args.model == "logreg":
        X_sample = X.sample(n=min(500, len(X)), random_state=42)
        explainer = shap.LinearExplainer(clf, X_sample, feature_perturbation="interventional")
        sv = normalize_shap_values(explainer.shap_values(X_sample))
    else:
        raise ValueError("Unsupported model for SHAP demo")

    # Ensure alignment
    if sv.shape[0] != len(X_sample) or sv.shape[1] != X_sample.shape[1]:
        raise RuntimeError(f"SHAP shape mismatch: sv={sv.shape}, X={X_sample.shape}")

    # Summary plot
    plt.figure()
    shap.summary_plot(sv, X_sample, show=False, plot_type="bar")
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "shap_summary.png"))
    plt.close()

    # Dependence plot for a chosen feature
    dep_feat = feature_cols[0] if args.feature is None else args.feature
    if dep_feat not in X_sample.columns:
        raise ValueError(f"Feature '{dep_feat}' not in columns. Choose from: {list(X_sample.columns)}")

    plt.figure()
    shap.dependence_plot(dep_feat, sv, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", f"shap_dependence_{dep_feat}.png"))
    plt.close()

    print("Saved:", os.path.join("outputs", "shap_summary.png"))
    print("Saved:", os.path.join("outputs", f"shap_dependence_{dep_feat}.png"))

if __name__ == "__main__":
    main()
