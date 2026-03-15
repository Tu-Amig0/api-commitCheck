# Commit Scorer

A machine learning model that scores git commits as clean (0) or bug-inducing (1).
Based on the Just-In-Time defect prediction research by Kamei et al. (2013).

## Project Structure

```
commit-scorer/
├── data/               # Dataset goes here
├── features/           # Feature extraction code
├── model/              # Trained model saved here
├── scripts/            # Runnable scripts
│   ├── download_data.py
│   ├── extract_features.py
│   ├── train.py
│   └── predict.py
├── requirements.txt
└── README.md
```

## Steps

1. Install dependencies:        pip install -r requirements.txt
2. Download dataset:            python scripts/download_data.py
3. Extract features:            python scripts/extract_features.py
4. Train the model:             python scripts/train.py
5. Score a new commit:          python scripts/predict.py <commit_hash> <repo_path>