"""
Step 2: Extract and combine features from:
  - data/raw/pull_requests.csv   (mined by mine_repos.py, includes AI columns)
  - data/raw/commits.csv         (commit-level data)

Produces:
  - data/features.csv            (final feature table for training)

Note: AI usage columns (ai_used, ai_tokens_in, ai_tokens_out, ai_agent_turns,
human_edit_ratio) are already in pull_requests.csv. Edit them there directly
before running this script if you want to log AI usage for specific PRs.

Usage:
    python scripts/extract_features.py
"""

import os
import numpy as np
import pandas as pd

PR_PATH     = "data/raw/pull_requests.csv"
COMMIT_PATH = "data/raw/commits.csv"
OUTPUT_PATH = "data/features.csv"


def load_data():
    print("Loading data...")
    prs     = pd.read_csv(PR_PATH)
    commits = pd.read_csv(COMMIT_PATH)

    print(f"  PRs:     {len(prs):,}")
    print(f"  Commits: {len(commits):,}")

    labeled = prs["label"].notna().sum()
    print(f"  Labeled PRs: {labeled} / {len(prs)}")

    return prs, commits


def extract_commit_aggregates(commits: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate commit-level features per PR.
    These capture the full picture of how the code was written.
    """
    agg = commits.groupby("pr_number").agg(
        commit_count          = ("commit_hash",  "count"),
        total_lines_added     = ("lines_added",  "sum"),
        total_lines_deleted   = ("lines_deleted","sum"),
        avg_lines_added       = ("lines_added",  "mean"),
        max_lines_in_commit   = ("lines_added",  "max"),
        has_late_night_commit = ("is_late_night", "max"),  # 1 if any commit was late
        earliest_commit_hour  = ("commit_hour",  "min"),
        latest_commit_hour    = ("commit_hour",  "max"),
        commit_hour_std       = ("commit_hour",  "std"),   # spread of commit times
        unique_authors        = ("author_email", "nunique"),
    ).reset_index()

    # Derived features
    agg["total_churn"]     = agg["total_lines_added"] + agg["total_lines_deleted"]
    agg["churn_ratio"]     = agg["total_lines_added"] / (agg["total_churn"] + 1)
    agg["commit_hour_std"] = agg["commit_hour_std"].fillna(0.0)

    return agg


def build_feature_table(prs: pd.DataFrame, commits: pd.DataFrame) -> pd.DataFrame:
    """
    Join PR and commit data into a single feature table.
    AI columns are already present in pull_requests.csv —
    derived AI features are computed here.
    """
    commit_agg = extract_commit_aggregates(commits)

    df = prs.copy()
    df = df.merge(commit_agg, on="pr_number", how="left")

    # Derived AI features — computed from columns already in pull_requests.csv
    if "ai_tokens_in" in df.columns and "ai_tokens_out" in df.columns:
        df["ai_total_tokens"] = df["ai_tokens_in"] + df["ai_tokens_out"]
        df["ai_token_ratio"]  = df["ai_tokens_out"] / (df["ai_total_tokens"] + 1)
    else:
        df["ai_total_tokens"] = 0
        df["ai_token_ratio"]  = 0.0

    # Zero out AI metrics for rows where ai_used = 0
    ai_cols = ["ai_tokens_in", "ai_tokens_out", "ai_agent_turns",
               "ai_total_tokens", "ai_token_ratio"]
    for col in ai_cols:
        if col in df.columns:
            df[col] = df[col] * df["ai_used"].fillna(0)

    return df


def select_final_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select only the columns the model will train on.
    Drops raw timestamps and string columns.
    """
    feature_cols = [
        # --- PR size ---
        "lines_added",
        "lines_deleted",
        "changed_files",
        "total_churn",
        "churn_ratio",

        # --- PR timing ---
        "time_to_merge_hours",
        "time_in_review_hours",
        "pr_open_hour",
        "pr_merged_hour",
        "pr_merged_weekday",
        "is_friday_merge",

        # --- Commit timing ---
        "commit_count",
        "has_late_night_commit",
        "earliest_commit_hour",
        "latest_commit_hour",
        "commit_hour_std",

        # --- Review quality ---
        "review_comment_count",
        "comments_count",
        "reviewer_count",
        "approvals_count",
        "pr_iteration_count",
        "first_response_hours",

        # --- Context ---
        "is_hotfix",
        "unique_authors",
        "subsystems_changed",

        # --- AI usage ---
        "ai_used",
        "ai_tokens_in",
        "ai_tokens_out",
        "ai_total_tokens",
        "ai_token_ratio",
        "ai_agent_turns",
        "human_edit_ratio",

        # --- Label ---
        "label",
    ]

    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]

    if missing:
        print(f"\nWARNING: These features are missing and will be skipped: {missing}")

    return df[available].copy()


def summarize(df: pd.DataFrame):
    labeled = df["label"].notna()
    print(f"\n=== Feature Table Summary ===")
    print(f"Total PRs:    {len(df):,}")
    print(f"Labeled PRs:  {labeled.sum():,}")
    print(f"Bug-inducing: {int(df.loc[labeled, 'label'].sum())} ({df.loc[labeled, 'label'].mean()*100:.1f}%)")
    print(f"Clean:        {int((df.loc[labeled, 'label'] == 0).sum())}")
    print(f"Features:     {len(df.columns) - 1}")
    print(f"\nFeature preview:")
    print(df.drop(columns=["label"]).describe().round(2).to_string())


def main():
    prs, commits = load_data()

    print("\nBuilding feature table...")
    df = build_feature_table(prs, commits)
    df = select_final_features(df)

    summarize(df)

    os.makedirs("data", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nFeatures saved to: {OUTPUT_PATH}")
    print("Next: python scripts/train.py")


if __name__ == "__main__":
    main()