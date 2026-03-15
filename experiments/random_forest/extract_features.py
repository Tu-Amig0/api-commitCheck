"""
Step 2: Extract features from the ApacheJIT dataset.

Based on Kamei et al. (2013) "Just-In-Time Defect Prediction" — 14 features
across 5 categories: Size, Diffusion, History, Experience, and Code metrics.

These are the same features validated across a decade of research.
"""

import os
import pandas as pd
import numpy as np

INPUT_PATH  = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/apachejit.csv"))
OUTPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/features.csv"))


def load_dataset():
    print(f"Loading dataset from {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df):,} commits.")
    print(f"Columns: {list(df.columns)}\n")
    return df


def extract_features(df):
    """
    Map ApacheJIT columns to Kamei's 14 features.

    The ApacheJIT dataset already contains pre-computed features — 
    this function selects and renames them clearly so you understand
    what each one means.
    """

    features = pd.DataFrame()

    # ----------------------------------------------------------------
    # CATEGORY 1: SIZE
    # How large is this commit? Larger commits = more risk.
    # ----------------------------------------------------------------
    features["lines_added"]         = df.get("la", np.nan)    # Lines of code added
    features["lines_deleted"]       = df.get("ld", np.nan)    # Lines of code deleted
    features["total_churn"]         = features["lines_added"] + features["lines_deleted"]

    # ----------------------------------------------------------------
    # CATEGORY 2: DIFFUSION
    # How spread out are the changes? Scattered = harder to review.
    # ----------------------------------------------------------------
    features["files_changed"]       = df.get("nf", np.nan)    # Number of modified files
    features["directories_changed"] = df.get("nd", np.nan)    # Number of modified directories
    features["subsystems_changed"]  = df.get("ns", np.nan)    # Number of modified subsystems

    # ----------------------------------------------------------------
    # CATEGORY 3: HISTORY
    # What is the track record of the changed files?
    # ----------------------------------------------------------------
    features["file_age"]            = df.get("age", np.nan)   # Average age of modified files (weeks)
    features["past_bugs"]           = df.get("ndb", np.nan)   # Unique bugs previously fixed in these files

    # ----------------------------------------------------------------
    # CATEGORY 4: EXPERIENCE
    # How experienced is the developer making this commit?
    # ----------------------------------------------------------------
    features["dev_experience"]      = df.get("ndev", np.nan)  # Number of developers who changed these files
    features["recent_experience"]   = df.get("rexp", np.nan)  # Recent experience (recent changes)
    features["subsystem_experience"]= df.get("sexp", np.nan)  # Experience in this subsystem

    # ----------------------------------------------------------------
    # CATEGORY 5: CODE CHURN (relative, stronger predictor)
    # How much of the file changed relative to its size?
    # ----------------------------------------------------------------
    features["churn_ratio"] = np.where(
        features["lines_added"] + features["lines_deleted"] > 0,
        features["lines_added"] / (features["lines_added"] + features["lines_deleted"] + 1),
        0
    )

    # ----------------------------------------------------------------
    # LABEL: What we are predicting
    # 0 = clean commit, 1 = bug-inducing commit
    # ----------------------------------------------------------------
    features["label"] = df.get("bug", df.get("label", np.nan))

    # Drop rows where label is missing
    features = features.dropna(subset=["label"])
    features["label"] = features["label"].astype(int)

    return features


def summarize(features):
    print("=== Feature Summary ===")
    print(f"Total commits:      {len(features):,}")
    print(f"Bug-inducing (1):   {features['label'].sum():,} ({features['label'].mean()*100:.1f}%)")
    print(f"Clean (0):          {(features['label'] == 0).sum():,} ({(1 - features['label'].mean())*100:.1f}%)")
    print(f"\nFeature stats:")
    print(features.drop(columns=["label"]).describe().round(2))


def main():
    df = load_dataset()
    features = extract_features(df)
    summarize(features)

    features.to_csv(OUTPUT_PATH, index=False)
    print(f"\nFeatures saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
    