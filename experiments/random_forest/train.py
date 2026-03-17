"""
Step 3: Train the PR scorer model.

Trains a Random Forest on labeled PRs and outputs:
  - A risk score between 0–100%
  - Feature importance (which signals matter most)
  - Evaluation metrics

Usage:
    python scripts/train.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    f1_score,
    confusion_matrix,
)

FEATURES_PATH = "data/features.csv"
MODEL_PATH    = "model/pr_scorer.pkl"
COLUMNS_PATH  = "model/feature_columns.pkl"


def load_labeled_features():
    print(f"Loading features from {FEATURES_PATH}...")
    df = pd.read_csv(FEATURES_PATH)

    # Only train on labeled rows
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)

    print(f"Labeled PRs available for training: {len(df):,}")
    print(f"Bug-inducing: {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
    print(f"Clean:        {(df['label'] == 0).sum()}\n")

    if len(df) < 20:
        print("WARNING: Very few labeled PRs. Consider labeling more before training.")
        print("The model needs at least ~50-100 labeled PRs for meaningful results.\n")

    return df


def split(df):
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].fillna(0)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(y.unique()) > 1 else None
    )

    print(f"Training set: {len(X_train)} PRs")
    print(f"Test set:     {len(X_test)} PRs\n")

    return X_train, X_test, y_train, y_test, feature_cols


def train(X_train, y_train):
    print("Training Random Forest...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",   # handles imbalanced bug/clean ratio
    )

    model.fit(X_train, y_train)
    print("Done.\n")
    return model


def evaluate(model, X_train, X_test, y_train, y_test, feature_cols):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("=== Evaluation ===")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.3f}  (0.5 = random, 1.0 = perfect)")

    # Cross-validation on training set for more reliable estimate
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"CV AUC:    {cv_scores.mean():.3f} ± {cv_scores.std():.3f}  (5-fold cross validation)\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Clean", "Bug-inducing"]))

    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))
    print()

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top = importances.sort_values(ascending=False).head(10)
    print("=== Top 10 Most Influential Features ===")
    for feat, score in top.items():
        bar = "█" * int(score * 200)
        print(f"  {feat:<30} {score:.4f}  {bar}")


def save(model, feature_cols):
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_cols, COLUMNS_PATH)
    print(f"\nModel saved to:   {MODEL_PATH}")
    print(f"Columns saved to: {COLUMNS_PATH}")


def main():
    df = load_labeled_features()
    X_train, X_test, y_train, y_test, feature_cols = split(df)
    model = train(X_train, y_train)
    evaluate(model, X_train, X_test, y_train, y_test, feature_cols)
    save(model, feature_cols)
    print("\nDone. Run predict.py or start the API to score a PR.")


if __name__ == "__main__":
    main()