# Commit Scorer API

A FastAPI REST endpoint that scores a git commit using the trained Random Forest model
from the `commit-scorer` project.

## Setup

### 1. Copy your trained model here

From your `commit-scorer` project, copy the two model files:

```bash
mkdir -p model
cp ../commit-scorer/model/random_forest.pkl   model/
cp ../commit-scorer/model/feature_columns.pkl model/
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the API

```bash
uvicorn main:app --reload
```

API is now live at: http://localhost:8000

Interactive docs at: http://localhost:8000/docs

---

## Usage

### Request

```
POST /score
Content-Type: application/json

{
  "commit_hash": "a1b2c3d4",
  "repo_path": "/home/user/my-project"
}
```

### Response

```json
{
  "commit_hash":  "a1b2c3d4e5f6...",
  "author":       "Jane Dev",
  "message":      "Fix login redirect bug",
  "prediction":   0,
  "label":        "clean",
  "bug_risk_pct": 12.3,
  "clean_pct":    87.7,
  "features": {
    "lines_added":          14,
    "lines_deleted":         3,
    "total_churn":          17,
    "files_changed":         2,
    "directories_changed":   1,
    "subsystems_changed":    1,
    "file_age":             42.1,
    "past_bugs":             0,
    "dev_experience":       87,
    "recent_experience":    12,
    "subsystem_experience":  9,
    "churn_ratio":         0.82
  }
}
```

---

## Project Structure

```
commit-scorer-api/
├── main.py           # FastAPI app — the entire API
├── model/
│   ├── random_forest.pkl      # trained model (copy from commit-scorer)
│   └── feature_columns.pkl    # feature column names
├── requirements.txt
└── README.md
```

## Environment Variables

| Variable       | Default                      | Description               |
|----------------|------------------------------|---------------------------|
| `MODEL_PATH`   | `model/random_forest.pkl`    | Path to trained model     |
| `COLUMNS_PATH` | `model/feature_columns.pkl`  | Path to feature columns   |