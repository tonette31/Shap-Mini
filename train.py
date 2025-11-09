from __future__ import annotations
import argparse, os, json
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import joblib
from utils import save_json, save_columns

def load_or_generate_data(data_path: str = "data/train.csv", n_samples: int = 2000, n_features: int = 12, seed: int = 42):
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        if "target" not in df.columns:
            raise ValueError("Expected 'target' column in data/train.csv")
        return df
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=6, n_redundant=2,
                               n_repeated=0, n_classes=2, random_state=seed, class_sep=1.0)
    cols = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y
    return df

def build_model(name: str):
    if name == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, n_jobs=None)),
        ])
    elif name == "rf":
        return Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ])
    else:
        raise ValueError("Unknown model. Use 'rf' or 'logreg'.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rf", choices=["rf","logreg"])
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = load_or_generate_data()
    y = df["target"].astype(int).to_numpy()
    X = df.drop(columns=["target"])
    feature_cols = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed, stratify=y)

    model = build_model(args.model)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "model": args.model,
    }

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(model, os.path.join("models", f"{args.model}.pkl"))
    save_json(os.path.join("outputs", "metrics.json"), metrics)
    save_columns(os.path.join("outputs", "train_columns.json"), feature_cols)

    print("Saved model:", os.path.join("models", f"{args.model}.pkl"))
    print("Saved metrics:", os.path.join("outputs", "metrics.json"))
    print("Saved columns:", os.path.join("outputs", "train_columns.json"))

if __name__ == "__main__":
    main()
