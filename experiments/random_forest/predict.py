"""
Step 4: Score a real git commit using the trained model.

Usage:
    python scripts/predict.py <commit_hash> <path_to_repo>

Example:
    python scripts/predict.py a1b2c3d4 /home/user/my-project
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from git import Repo

MODEL_PATH   = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model/random_forest.pkl"))
COLUMNS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model/feature_columns.pkl"))


def load_model():
    if not os.path.exists(MODEL_PATH):
        print("ERROR: No trained model found.")
        print("Please run `python scripts/train.py` first.")
        sys.exit(1)
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(COLUMNS_PATH)
    return model, feature_cols


def extract_commit_features(repo_path, commit_hash):
    """
    Extract the same features from a real commit that we trained on.
    """
    repo = Repo(repo_path)

    try:
        commit = repo.commit(commit_hash)
    except Exception:
        print(f"ERROR: Commit '{commit_hash}' not found in repo at '{repo_path}'")
        sys.exit(1)

    # Get the diff against the parent commit
    if not commit.parents:
        print("WARNING: This is the initial commit, no parent to diff against.")
        diffs = []
    else:
        diffs = commit.diff(commit.parents[0], create_patch=True)

    # --- SIZE ---
    lines_added   = sum(d.diff.decode("utf-8", errors="ignore").count("\n+") for d in diffs)
    lines_deleted = sum(d.diff.decode("utf-8", errors="ignore").count("\n-") for d in diffs)
    total_churn   = lines_added + lines_deleted

    # --- DIFFUSION ---
    files_changed       = len(diffs)
    directories_changed = len(set(
        os.path.dirname(d.b_path or d.a_path) for d in diffs
    ))
    subsystems_changed  = len(set(
        (d.b_path or d.a_path).split("/")[0] for d in diffs if (d.b_path or d.a_path)
    ))

    # --- HISTORY (approximated from git log) ---
    changed_files = [d.b_path or d.a_path for d in diffs if (d.b_path or d.a_path)]
    file_ages     = []
    past_bugs     = 0

    for filepath in changed_files[:10]:  # limit to 10 files for speed
        try:
            file_commits = list(repo.iter_commits(paths=filepath, max_count=50))
            if file_commits:
                age_days = (commit.committed_date - file_commits[-1].committed_date) / 86400
                file_ages.append(max(age_days / 7, 0))  # convert to weeks
        except Exception:
            pass

    file_age = np.mean(file_ages) if file_ages else 0

    # --- EXPERIENCE ---
    author_email  = commit.author.email
    all_commits   = list(repo.iter_commits(max_count=500))
    dev_experience = sum(1 for c in all_commits if c.author.email == author_email)
    recent_experience = sum(
        1 for c in all_commits[:100] if c.author.email == author_email
    )
    subsystem_experience = sum(
        1 for c in all_commits[:200]
        if c.author.email == author_email and any(
            (d.b_path or "").startswith(sub)
            for d in c.diff(c.parents[0]) if c.parents
            for sub in [f.split("/")[0] for f in changed_files]
        )
    )

    # --- RELATIVE CHURN ---
    churn_ratio = lines_added / (total_churn + 1)

    features = {
        "lines_added":          lines_added,
        "lines_deleted":        lines_deleted,
        "total_churn":          total_churn,
        "files_changed":        files_changed,
        "directories_changed":  directories_changed,
        "subsystems_changed":   subsystems_changed,
        "file_age":             file_age,
        "past_bugs":            past_bugs,
        "dev_experience":       dev_experience,
        "recent_experience":    recent_experience,
        "subsystem_experience": subsystem_experience,
        "churn_ratio":          churn_ratio,
    }

    return features, commit


def predict(model, feature_cols, features):
    df = pd.DataFrame([features])

    # Align columns to what the model was trained on
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols].fillna(0)

    prediction  = model.predict(df)[0]
    probability = model.predict_proba(df)[0]

    return prediction, probability


def print_report(commit, features, prediction, probability):
    print("\n" + "="*50)
    print("COMMIT SCORE REPORT")
    print("="*50)
    print(f"Commit:   {commit.hexsha[:8]}")
    print(f"Author:   {commit.author.name}")
    print(f"Message:  {commit.message.strip()[:80]}")
    print()
    print("--- Extracted Features ---")
    for k, v in features.items():
        print(f"  {k:<25} {round(v, 3)}")
    print()
    print("--- Prediction ---")
    print(f"  Score:       {'⚠️  BUG-INDUCING (1)' if prediction == 1 else '✅  CLEAN (0)'}")
    print(f"  Confidence:  {max(probability)*100:.1f}%")
    print(f"  Bug risk:    {probability[1]*100:.1f}%")
    print(f"  Clean prob:  {probability[0]*100:.1f}%")
    print("="*50)


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/predict.py <commit_hash> <repo_path>")
        print("Example: python scripts/predict.py a1b2c3d /home/user/myproject")
        sys.exit(1)

    commit_hash = sys.argv[1]
    repo_path   = sys.argv[2]

    print(f"Loading model...")
    model, feature_cols = load_model()

    print(f"Extracting features from commit {commit_hash[:8]}...")
    features, commit = extract_commit_features(repo_path, commit_hash)

    prediction, probability = predict(model, feature_cols, features)
    print_report(commit, features, prediction, probability)


if __name__ == "__main__":
    main()