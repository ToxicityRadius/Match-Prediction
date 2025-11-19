import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import load_teammatch_features


def evaluate_models(X, y, numeric_cols, random_state=42):
    ct = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric_cols)], remainder="passthrough"
    )

    models = {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "rf": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=random_state),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    results = {}
    for name, model in models.items():
        pipe = Pipeline([("preproc", ct), ("clf", model)])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        results[name] = {
            "mean_roc_auc": float(np.mean(scores)),
            "std_roc_auc": float(np.std(scores)),
            "all_scores": scores.tolist(),
        }

    return results


def train_and_save_best(X, y, numeric_cols, out_path: Path, random_state=42):
    ct = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric_cols)], remainder="passthrough"
    )

    candidates = {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "rf": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=random_state),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    best_name = None
    best_score = -1.0
    best_pipe = None

    for name, model in candidates.items():
        pipe = Pipeline([("preproc", ct), ("clf", model)])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        mean_score = float(np.mean(scores))
        print(f"Model {name}: ROC-AUC {mean_score:.4f} (+/- {np.std(scores):.4f})")
        if mean_score > best_score:
            best_score = mean_score
            best_name = name
            best_pipe = pipe

    # fit best pipeline on all data
    print(f"Fitting best model: {best_name}")
    if best_pipe is not None:
        best_pipe.fit(X, y)
    else:
        raise ValueError("No valid model found")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": best_pipe, "numeric_cols": numeric_cols}, out_path)
    print(f"Saved model pipeline to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="TeamMatchTbl.csv", help="Path to TeamMatchTbl.csv")
    ap.add_argument("--summoner", default=None, help="Path to SummonerMatchTbl.csv (optional)")
    ap.add_argument("--stats", default=None, help="Path to MatchStatsTbl.csv (optional)")
    ap.add_argument("--out", default="model_pipeline.joblib", help="Path to save model")
    ap.add_argument("--nrows", type=int, default=None, help="Optional: read only first N rows")
    args = ap.parse_args()

    print("Loading data...")
    X, y, mlb_b, mlb_r, numeric_cols = load_teammatch_features(
        args.csv, args.summoner, args.stats, nrows=args.nrows
    )
    print("Feature matrix shape:", X.shape)
    print(f"Features: {len(numeric_cols)} numeric/player stats, {X.shape[1] - len(numeric_cols)} champion one-hots")

    print("Evaluating candidates with cross-validation...")
    results = evaluate_models(X, y, numeric_cols)
    for k, v in results.items():
        print(f"{k}: mean ROC-AUC={v['mean_roc_auc']:.4f} std={v['std_roc_auc']:.4f}")

    train_and_save_best(X, y, numeric_cols, Path(args.out))


if __name__ == "__main__":
    main()
