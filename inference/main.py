"""
Commit Scorer API

A single REST endpoint that receives a git commit hash + repo path
and returns a bug-risk score from the trained Random Forest model.

Usage:
    uvicorn main:app --reload

Docs available at:
    http://localhost:8000/docs
"""

import os
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from git import Repo, InvalidGitRepositoryError, NoSuchPathError

# ----------------------------------------------------------------
# Load model on startup — once, not on every request
# ----------------------------------------------------------------
MODEL_PATH   = os.environ.get("MODEL_PATH",   "model/random_forest.pkl")
COLUMNS_PATH = os.environ.get("COLUMNS_PATH", "model/feature_columns.pkl")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"Model not found at '{MODEL_PATH}'.\n"
        "Copy random_forest.pkl and feature_columns.pkl from your "
        "commit-scorer project into a 'model/' folder here."
    )

model        = joblib.load(MODEL_PATH)
feature_cols = joblib.load(COLUMNS_PATH)

# ----------------------------------------------------------------
# App
# ----------------------------------------------------------------
app = FastAPI(
    title="Commit Scorer API",
    description="Predicts whether a git commit is bug-inducing using a trained Random Forest model.",
    version="1.0.0",
)

# ----------------------------------------------------------------
# Request / Response schemas
# ----------------------------------------------------------------
class ScoreRequest(BaseModel):
    commit_hash: str
    repo_path: str

    class Config:
        json_schema_extra = {
            "example": {
                "commit_hash": "a1b2c3d4",
                "repo_path": "/home/user/my-project"
            }
        }


class ScoreResponse(BaseModel):
    commit_hash:   str
    author:        str
    message:       str
    prediction:    int           # 0 = clean, 1 = bug-inducing
    label:         str           # human readable
    bug_risk_pct:  float         # 0.0 – 100.0
    clean_pct:     float         # 0.0 – 100.0
    features:      dict


# ----------------------------------------------------------------
# Feature extraction (same logic as predict.py in commit-scorer)
# ----------------------------------------------------------------
def extract_features(repo_path: str, commit_hash: str) -> tuple[dict, object]:
    try:
        repo = Repo(repo_path)
    except NoSuchPathError:
        raise HTTPException(status_code=404, detail=f"Repo path not found: {repo_path}")
    except InvalidGitRepositoryError:
        raise HTTPException(status_code=400, detail=f"Not a valid git repo: {repo_path}")

    try:
        commit = repo.commit(commit_hash)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Commit '{commit_hash}' not found in repo")

    diffs = commit.diff(commit.parents[0], create_patch=True) if commit.parents else []

    # SIZE
    lines_added   = sum(d.diff.decode("utf-8", errors="ignore").count("\n+") for d in diffs)
    lines_deleted = sum(d.diff.decode("utf-8", errors="ignore").count("\n-") for d in diffs)
    total_churn   = lines_added + lines_deleted

    # DIFFUSION
    files_changed       = len(diffs)
    directories_changed = len(set(
        os.path.dirname(d.b_path or d.a_path) for d in diffs
    ))
    subsystems_changed  = len(set(
        (d.b_path or d.a_path).split("/")[0]
        for d in diffs if (d.b_path or d.a_path)
    ))

    # HISTORY
    changed_files = [d.b_path or d.a_path for d in diffs if (d.b_path or d.a_path)]
    file_ages = []
    for filepath in changed_files[:10]:
        try:
            file_commits = list(repo.iter_commits(paths=filepath, max_count=50))
            if file_commits:
                age_days = (commit.committed_date - file_commits[-1].committed_date) / 86400
                file_ages.append(max(age_days / 7, 0))
        except Exception:
            pass
    file_age = float(np.mean(file_ages)) if file_ages else 0.0

    # EXPERIENCE
    author_email       = commit.author.email
    all_commits        = list(repo.iter_commits(max_count=500))
    dev_experience     = sum(1 for c in all_commits if c.author.email == author_email)
    recent_experience  = sum(1 for c in all_commits[:100] if c.author.email == author_email)
    subsystem_experience = sum(
        1 for c in all_commits[:200]
        if c.author.email == author_email and any(
            (d.b_path or "").startswith(sub)
            for d in c.diff(c.parents[0]) if c.parents
            for sub in [f.split("/")[0] for f in changed_files]
        )
    )

    # RELATIVE CHURN
    churn_ratio = lines_added / (total_churn + 1)

    features = {
        "lines_added":           lines_added,
        "lines_deleted":         lines_deleted,
        "total_churn":           total_churn,
        "files_changed":         files_changed,
        "directories_changed":   directories_changed,
        "subsystems_changed":    subsystems_changed,
        "file_age":              round(file_age, 3),
        "past_bugs":             0,
        "dev_experience":        dev_experience,
        "recent_experience":     recent_experience,
        "subsystem_experience":  subsystem_experience,
        "churn_ratio":           round(churn_ratio, 3),
    }

    return features, commit


# ----------------------------------------------------------------
# The single endpoint
# ----------------------------------------------------------------
@app.post("/score", response_model=ScoreResponse)
def score_commit(request: ScoreRequest):
    """
    Score a git commit.

    - **commit_hash**: The commit SHA (full or short) to score
    - **repo_path**: Absolute path to the git repository on this machine

    Returns a prediction (0 = clean, 1 = bug-inducing) and a bug risk percentage.
    """
    features, commit = extract_features(request.repo_path, request.commit_hash)

    # Align features to training columns
    df = pd.DataFrame([features])
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols].fillna(0)

    prediction  = int(model.predict(df)[0])
    probability = model.predict_proba(df)[0]

    return ScoreResponse(
        commit_hash  = commit.hexsha,
        author       = commit.author.name,
        message      = commit.message.strip()[:120],
        prediction   = prediction,
        label        = "bug-inducing" if prediction == 1 else "clean",
        bug_risk_pct = round(float(probability[1]) * 100, 1),
        clean_pct    = round(float(probability[0]) * 100, 1),
        features     = features,
    )