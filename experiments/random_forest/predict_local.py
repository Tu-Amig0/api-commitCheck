"""
predict_local.py — Score any git commit locally. No GitHub API needed.

Works with regular commits, merge commits, squash commits — any SHA.

Usage:
    python scripts/predict_local.py <repo_path> <commit_sha>

Example:
    python scripts/predict_local.py /home/user/my-project a1b2c3d4

Optional AI usage flags:
    --ai-used 1
    --ai-tokens-in 3200
    --ai-tokens-out 5800
    --ai-agent-turns 12
    --human-edit-ratio 0.6
"""

import os
import re
import sys
import subprocess
import joblib
import pandas as pd
from datetime import datetime

MODEL_PATH   = os.environ.get("MODEL_PATH",   "model/pr_scorer.pkl")
COLUMNS_PATH = os.environ.get("COLUMNS_PATH", "model/feature_columns.pkl")

# ----------------------------------------------------------------
# Load model
# ----------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"Model not found at '{MODEL_PATH}'.\n"
        "Run python scripts/train.py first."
    )

model        = joblib.load(MODEL_PATH)
feature_cols = joblib.load(COLUMNS_PATH)


# ----------------------------------------------------------------
# Git helpers
# ----------------------------------------------------------------

def git(repo_path: str, *args) -> str:
    result = subprocess.run(
        ["git", "-C", repo_path, *args],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def parse_ts(ts: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(ts[:19], fmt)
        except Exception:
            continue
    return datetime.min


def hour_of(ts: str) -> int:
    dt = parse_ts(ts)
    return dt.hour if dt != datetime.min else -1


def weekday_of(ts: str) -> int:
    dt = parse_ts(ts)
    return dt.weekday() if dt != datetime.min else -1


def hours_between(start: str, end: str) -> float | None:
    a, b = parse_ts(start), parse_ts(end)
    if a == datetime.min or b == datetime.min:
        return None
    return round(abs((b - a).total_seconds() / 3600), 2)


# ----------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------

def extract_features(repo_path: str, commit_sha: str) -> tuple[dict, dict]:
    """
    Extract scoring features from any commit.

    For a regular commit:  diffs against its single parent
    For a merge commit:    diffs the PR branch tip against the base branch
    """

    # Resolve full SHA
    full_sha = git(repo_path, "rev-parse", commit_sha)
    if not full_sha:
        raise ValueError(f"Commit '{commit_sha}' not found in {repo_path}")

    # Get parents
    parents_raw = git(repo_path, "log", "-1", "--format=%P", full_sha)
    parents     = parents_raw.strip().split()

    if len(parents) == 0:
        raise ValueError("This is the initial commit — nothing to diff against.")

    # Diff base: for merge commits use both parents, for regular use parent^
    if len(parents) >= 2:
        diff_base = parents[0]   # base branch
        diff_tip  = parents[1]   # PR branch tip
        is_merge  = True
    else:
        diff_base = parents[0]   # previous commit
        diff_tip  = full_sha
        is_merge  = False

    # --- Commit metadata ---
    committed_at = git(repo_path, "log", "-1", "--format=%ai", full_sha)
    subject      = git(repo_path, "log", "-1", "--format=%s",  full_sha)
    author_email = git(repo_path, "log", "-1", "--format=%ae", full_sha)
    committed_dt = parse_ts(committed_at)

    pr_match  = re.search(r"#(\d+)", subject)
    pr_number = int(pr_match.group(1)) if pr_match else -1

    # --- Diff stats ---
    stat          = git(repo_path, "diff", "--shortstat", diff_base, diff_tip)
    lines_added   = int(m.group(1)) if (m := re.search(r"(\d+) insertion", stat)) else 0
    lines_deleted = int(m.group(1)) if (m := re.search(r"(\d+) deletion",  stat)) else 0
    files_changed = int(m.group(1)) if (m := re.search(r"(\d+) file",      stat)) else 0
    total_churn   = lines_added + lines_deleted
    churn_ratio   = round(lines_added / (total_churn + 1), 3)

    # --- Changed files ---
    files_raw     = git(repo_path, "diff", "--name-only", diff_base, diff_tip)
    changed_files = [f.strip() for f in files_raw.splitlines() if f.strip()]
    subsystems    = len(set(f.split("/")[0] for f in changed_files if "/" in f))

    # --- Commits in range ---
    # For merge commits: all commits in the PR branch
    # For regular commits: just this single commit
    if is_merge:
        log = git(repo_path, "log", f"{diff_base}..{diff_tip}", "--format=%H|||%ae|||%ai|||%s")
    else:
        log = git(repo_path, "log", "-1", "--format=%H|||%ae|||%ai|||%s", full_sha)

    commits = []
    for line in log.splitlines():
        parts = line.split("|||")
        if len(parts) < 4:
            continue
        sha, email, ts, msg = parts
        commits.append({
            "sha":   sha.strip(),
            "email": email.strip(),
            "ts":    ts.strip(),
            "hour":  hour_of(ts.strip()),
            "msg":   msg.strip(),
        })

    commit_hours   = [c["hour"] for c in commits]
    unique_authors = len(set(c["email"] for c in commits))

    # --- Timing ---
    first_commit_ts     = commits[-1]["ts"] if commits else committed_at
    time_to_merge_hours = hours_between(first_commit_ts, committed_at)

    # --- Context ---
    branch_ref = git(repo_path, "log", "-1", "--format=%D", diff_tip)
    is_hotfix  = int(bool(re.search(
        r"\b(hotfix|patch|fix)\b", branch_ref + subject, re.IGNORECASE
    )))

    features = {
        # --- Size ---
        "lines_added":           lines_added,
        "lines_deleted":         lines_deleted,
        "changed_files":         files_changed,
        "total_churn":           total_churn,
        "churn_ratio":           churn_ratio,

        # --- Timing ---
        "time_to_merge_hours":   time_to_merge_hours,
        "time_in_review_hours":  None,
        "first_response_hours":  None,
        "pr_open_hour":          hour_of(first_commit_ts),
        "pr_merged_hour":        hour_of(committed_at),
        "pr_merged_weekday":     weekday_of(committed_at),
        "is_friday_merge":       int(weekday_of(committed_at) == 4),

        # --- Commit signals ---
        "commit_count":          len(commits),
        "has_late_night_commit": int(any(h >= 22 or h <= 4 for h in commit_hours if h >= 0)),
        "earliest_commit_hour":  min(commit_hours) if commit_hours else -1,
        "latest_commit_hour":    max(commit_hours) if commit_hours else -1,
        "commit_hour_std":       round(float(pd.Series(commit_hours).std()), 2) if len(commit_hours) > 1 else 0.0,

        # --- Review signals (not available locally) ---
        "review_comment_count":  0,
        "comments_count":        0,
        "reviewer_count":        0,
        "approvals_count":       0,
        "pr_iteration_count":    0,

        # --- Context ---
        "is_hotfix":             is_hotfix,
        "unique_authors":        unique_authors,
        "subsystems_changed":    subsystems,

        # --- AI usage (injected after) ---
        "ai_used":               0,
        "ai_tokens_in":          0,
        "ai_tokens_out":         0,
        "ai_total_tokens":       0,
        "ai_token_ratio":        0.0,
        "ai_agent_turns":        0,
        "human_edit_ratio":      1.0,
    }

    meta = {
        "pr_number":    pr_number,
        "full_sha":     full_sha,
        "subject":      subject,
        "author":       author_email,
        "committed_at": committed_at,
        "is_merge":     is_merge,
        "commit_count": len(commits),
        "changed_files": changed_files[:5],
        "total_files":  len(changed_files),
    }

    return features, meta


# ----------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------

def score(features: dict) -> tuple[float, dict]:
    df = pd.DataFrame([features])

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols].fillna(0)

    probability = model.predict_proba(df)[0]
    risk_pct    = round(float(probability[1]) * 100, 1)

    contributions = {
        feat: {
            "value":      round(float(df[feat].iloc[0]), 4),
            "importance": round(float(imp), 4),
        }
        for feat, imp in zip(feature_cols, model.feature_importances_)
    }

    top_contributors = dict(
        sorted(contributions.items(), key=lambda x: x[1]["importance"], reverse=True)[:5]
    )

    return risk_pct, top_contributors


# ----------------------------------------------------------------
# CLI
# ----------------------------------------------------------------

def print_report(features, meta, risk_pct, top_contributors):
    risk_label = "🔴 HIGH" if risk_pct >= 60 else "🟡 MEDIUM" if risk_pct >= 30 else "🟢 LOW"
    commit_type = "merge commit" if meta["is_merge"] else "regular commit"

    print(f"\n{'='*55}")
    print(f"COMMIT SCORE REPORT (offline)")
    print(f"{'='*55}")
    print(f"SHA:         {meta['full_sha'][:8]}  ({commit_type})")
    if meta["pr_number"] > 0:
        print(f"PR:          #{meta['pr_number']}")
    print(f"Message:     {meta['subject'][:70]}")
    print(f"Author:      {meta['author']}")
    print(f"Committed:   {meta['committed_at']}")
    print(f"Commits:     {meta['commit_count']}")
    print(f"Files:       {meta['total_files']}  ({', '.join(meta['changed_files'][:3])}{'...' if meta['total_files'] > 3 else ''})")
    print(f"\nRisk Score:  {risk_pct}%  {risk_label}")
    print(f"\nTop contributing features:")
    for feat, data in top_contributors.items():
        print(f"  {feat:<30} value={data['value']}  importance={data['importance']}")
    print(f"\nNote: review features defaulted to 0 (not available locally).")
    print(f"{'='*55}")


def get_arg(args, name, default):
    try:
        idx = args.index(name)
        return type(default)(args[idx + 1])
    except (ValueError, IndexError):
        return default


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    repo_path = sys.argv[1]
    commit_sha = sys.argv[2]
    args = sys.argv[3:]

    ai_used          = get_arg(args, "--ai-used",          0)
    ai_tokens_in     = get_arg(args, "--ai-tokens-in",     0)
    ai_tokens_out    = get_arg(args, "--ai-tokens-out",    0)
    ai_agent_turns   = get_arg(args, "--ai-agent-turns",   0)
    human_edit_ratio = get_arg(args, "--human-edit-ratio", 1.0)

    print(f"Scoring {commit_sha[:8]} in {repo_path}...")

    try:
        features, meta = extract_features(repo_path, commit_sha)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    features["ai_used"]          = ai_used
    features["ai_tokens_in"]     = ai_tokens_in
    features["ai_tokens_out"]    = ai_tokens_out
    features["ai_total_tokens"]  = ai_tokens_in + ai_tokens_out
    features["ai_token_ratio"]   = ai_tokens_out / (ai_tokens_in + ai_tokens_out + 1)
    features["ai_agent_turns"]   = ai_agent_turns
    features["human_edit_ratio"] = human_edit_ratio

    risk_pct, top_contributors = score(features)
    print_report(features, meta, risk_pct, top_contributors)