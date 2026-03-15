"""
Step 3: Train a Random Forest model on the extracted features.

What this script does:
1. Loads the feature CSV from step 2
2. Splits data into train and test sets
3. Trains a Random Forest classifier
4. Evaluates it with standard metrics
5. Saves the trained model to disk
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score
)

FEATURES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/features.csv"))
MODEL_PATH    = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model/random_forest.pkl"))
COLUMNS_PATH  = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model/feature_columns.pkl"))


def load_features():
    print(f"Loading features from {FEATURES_PATH} ...")
    df = pd.read_csv(FEATURES_PATH)
    print(f"Loaded {len(df):,} rows.\n")
    return df


def split_data(df):
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].fillna(0)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,       # 80% train, 20% test
        random_state=42,     # reproducible results
        stratify=y           # keep same bug ratio in both splits
    )

    print(f"Training set:  {len(X_train):,} commits")
    print(f"Test set:      {len(X_test):,} commits\n")

    return X_train, X_test, y_train, y_test, feature_cols


def train(X_train, y_train):
    print("Training Random Forest...")
    print("(This may take 30–60 seconds on 100k commits)\n")

    model = RandomForestClassifier(
        n_estimators=100,    # 100 decision trees in the forest
        max_depth=10,        # limit tree depth to avoid overfitting
        random_state=42,
        n_jobs=-1,           # use all CPU cores
        class_weight="balanced"  # handle imbalanced bug/clean ratio
    )

    model.fit(X_train, y_train)
    print("Training complete.\n")
    return model


def evaluate(model, X_test, y_test, feature_cols):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("=== Model Evaluation ===")
    print(f"F1 Score:   {f1_score(y_test, y_pred):.3f}  (main metric used in research)")
    print(f"ROC-AUC:    {roc_auc_score(y_test, y_prob):.3f}  (1.0 = perfect, 0.5 = random)")
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Clean", "Bug-inducing"]))
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))
    print()

    # Show which features mattered most
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    print("=== Top Features (most influential) ===")
    print(importances.sort_values(ascending=False).round(4).to_string())


def save_model(model, feature_cols):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_cols, COLUMNS_PATH)
    print(f"\nModel saved to:    {MODEL_PATH}")
    print(f"Columns saved to:  {COLUMNS_PATH}")


def main():
    df = load_features()
    X_train, X_test, y_train, y_test, feature_cols = split_data(df)
    model = train(X_train, y_train)
    evaluate(model, X_test, y_test, feature_cols)
    save_model(model, feature_cols)
    print("\nDone. You can now run predict.py to score a real commit.")


if __name__ == "__main__":
    main()