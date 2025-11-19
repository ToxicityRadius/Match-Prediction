"""
Model evaluation and visualization script.

Creates comprehensive plots for model performance analysis:
- ROC curves for all models
- Confusion matrices
- Feature importance (for tree-based models)
- Model comparison charts
- Cross-validation score distributions
"""

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore[import-untyped]
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import load_teammatch_features

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


def plot_roc_curves(X, y, numeric_cols, output_dir: Path, random_state=42):
    """Plot ROC curves for all models."""
    print("\n📊 Generating ROC curves...")
    
    ct = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric_cols)], remainder="passthrough"
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=random_state),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        pipe = Pipeline([("preproc", ct), ("clf", model)])
        
        # Get cross-validated predictions
        y_proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y, y_proba)
        auc_score = roc_auc_score(y, y_proba)
        
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc_score:.3f})")
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves - Model Comparison", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    output_path = output_dir / "roc_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_confusion_matrices(X, y, numeric_cols, output_dir: Path, random_state=42):
    """Plot confusion matrices for all models."""
    print("\n📊 Generating confusion matrices...")
    
    ct = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric_cols)], remainder="passthrough"
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=random_state),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, (name, model) in enumerate(models.items()):
        pipe = Pipeline([("preproc", ct), ("clf", model)])
        
        # Get cross-validated predictions
        y_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Plot
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[idx],
            cbar=True,
            square=True,
        )
        axes[idx].set_title(f"{name}", fontsize=12, fontweight="bold")
        axes[idx].set_ylabel("True Label", fontsize=10)
        axes[idx].set_xlabel("Predicted Label", fontsize=10)
        axes[idx].set_xticklabels(["Red Win", "Blue Win"])
        axes[idx].set_yticklabels(["Red Win", "Blue Win"])
    
    plt.tight_layout()
    output_path = output_dir / "confusion_matrices.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_feature_importance(X, y, numeric_cols, output_dir: Path, top_n=20, random_state=42):
    """Plot feature importance for Random Forest model."""
    print("\n📊 Generating feature importance plot...")
    
    ct = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric_cols)], remainder="passthrough"
    )

    model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=random_state)
    pipe = Pipeline([("preproc", ct), ("clf", model)])
    
    # Fit model
    pipe.fit(X, y)
    
    # Get feature names and importances
    feature_names = pipe.named_steps["preproc"].get_feature_names_out()
    importances = pipe.named_steps["clf"].feature_importances_
    
    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df, x="importance", y="feature", palette="viridis")
    plt.title(f"Top {top_n} Most Important Features (Random Forest)", fontsize=14, fontweight="bold")
    plt.xlabel("Feature Importance", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()
    
    output_path = output_dir / "feature_importance.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved: {output_path}")
    plt.close()
    
    # Save to CSV
    csv_path = output_dir / "feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved: {csv_path}")


def plot_cv_scores_comparison(X, y, numeric_cols, output_dir: Path, random_state=42):
    """Plot cross-validation score distributions for model comparison."""
    print("\n📊 Generating CV score comparison...")
    
    ct = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric_cols)], remainder="passthrough"
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=random_state),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    results = []
    
    for name, model in models.items():
        pipe = Pipeline([("preproc", ct), ("clf", model)])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        
        for fold, score in enumerate(scores, 1):
            results.append({"Model": name, "Fold": fold, "ROC-AUC": score})
    
    results_df = pd.DataFrame(results)
    
    # Create comparison plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot
    sns.boxplot(data=results_df, x="Model", y="ROC-AUC", ax=axes[0], palette="Set2")
    axes[0].set_title("Cross-Validation Score Distribution", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("ROC-AUC Score", fontsize=10)
    axes[0].set_ylim([0.9, 1.0])
    axes[0].grid(True, alpha=0.3)
    
    # Bar plot with error bars
    summary = results_df.groupby("Model")["ROC-AUC"].agg(["mean", "std"]).reset_index()
    axes[1].bar(summary["Model"], summary["mean"], yerr=summary["std"], 
                capsize=5, alpha=0.7, color=["#66c2a5", "#fc8d62"])
    axes[1].set_title("Mean CV Scores with Standard Deviation", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("ROC-AUC Score", fontsize=10)
    axes[1].set_ylim([0.9, 1.0])
    axes[1].grid(True, alpha=0.3, axis="y")
    
    # Add value labels on bars
    for idx, row in summary.iterrows():
        axes[1].text(idx, row["mean"] + row["std"] + 0.002, 
                    f'{row["mean"]:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / "cv_scores_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved: {output_path}")
    plt.close()
    
    # Save summary to CSV
    csv_path = output_dir / "cv_scores_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"  ✓ Saved: {csv_path}")


def plot_class_distribution(y, output_dir: Path):
    """Plot target class distribution."""
    print("\n📊 Generating class distribution plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Count plot
    class_counts = y.value_counts()
    axes[0].bar(["Red Win (0)", "Blue Win (1)"], class_counts.values, color=["#e74c3c", "#3498db"], alpha=0.7)
    axes[0].set_title("Class Distribution (Counts)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Number of Matches", fontsize=10)
    axes[0].grid(True, alpha=0.3, axis="y")
    
    # Add value labels
    for i, count in enumerate(class_counts.values):
        axes[0].text(i, count + 5, str(count), ha='center', va='bottom', fontsize=10, fontweight="bold")
    
    # Pie chart
    axes[1].pie(class_counts.values, labels=["Red Win", "Blue Win"], autopct='%1.1f%%',
                colors=["#e74c3c", "#3498db"], startangle=90)
    axes[1].set_title("Class Distribution (Percentage)", fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    output_path = output_dir / "class_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def generate_classification_report(X, y, numeric_cols, output_dir: Path, random_state=42):
    """Generate detailed classification reports."""
    print("\n📊 Generating classification reports...")
    
    ct = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric_cols)], remainder="passthrough"
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=random_state),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    reports = []
    
    for name, model in models.items():
        pipe = Pipeline([("preproc", ct), ("clf", model)])
        y_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
        
        report = classification_report(y, y_pred, target_names=["Red Win", "Blue Win"], output_dict=True)
        reports.append(f"\n{'='*60}\n{name}\n{'='*60}\n")
        reports.append(classification_report(y, y_pred, target_names=["Red Win", "Blue Win"]))
    
    # Save to file
    output_path = output_dir / "classification_reports.txt"
    with open(output_path, "w") as f:
        f.writelines(reports)
    print(f"  ✓ Saved: {output_path}")


def main():
    ap = argparse.ArgumentParser(description="Generate model evaluation visualizations")
    ap.add_argument("--csv", default="TeamMatchTbl.csv", help="Path to TeamMatchTbl.csv")
    ap.add_argument("--summoner", default=None, help="Path to SummonerMatchTbl.csv (optional)")
    ap.add_argument("--stats", default=None, help="Path to MatchStatsTbl.csv (optional)")
    ap.add_argument("--nrows", type=int, default=None, help="Optional: read only first N rows")
    ap.add_argument("--output", default="visualizations", help="Output directory for plots")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    print("="*60)
    print("Model Evaluation & Visualization")
    print("="*60)

    # Load data
    print("\n📂 Loading data...")
    X, y, _, _, numeric_cols = load_teammatch_features(
        args.csv, args.summoner, args.stats, nrows=args.nrows
    )
    print(f"  ✓ Loaded {X.shape[0]} samples with {X.shape[1]} features")
    print(f"  ✓ {len(numeric_cols)} numeric/player stats, {X.shape[1] - len(numeric_cols)} champion one-hots")

    # Generate all visualizations
    plot_class_distribution(y, output_dir)
    plot_roc_curves(X, y, numeric_cols, output_dir, args.seed)
    plot_confusion_matrices(X, y, numeric_cols, output_dir, args.seed)
    plot_cv_scores_comparison(X, y, numeric_cols, output_dir, args.seed)
    plot_feature_importance(X, y, numeric_cols, output_dir, random_state=args.seed)
    generate_classification_report(X, y, numeric_cols, output_dir, args.seed)

    print("\n" + "="*60)
    print(f"✅ All visualizations saved to: {output_dir.absolute()}")
    print("="*60)


if __name__ == "__main__":
    main()
