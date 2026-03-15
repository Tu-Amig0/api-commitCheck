"""
Step 1: Download the ApacheJIT dataset.

ApacheJIT is a research dataset of 106,674 commits from Apache projects,
labeled as clean (0) or bug-inducing (1).

Source: https://github.com/shiroyasha11/ApacheJIT
"""

import os
import requests

# Direct CSV download from the ApacheJIT GitHub repo
DATASET_URL = "https://raw.githubusercontent.com/shiroyasha11/ApacheJIT/main/ApacheJIT.csv"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "../data/apachejit.csv")


def download():
    output_path = os.path.abspath(OUTPUT_PATH)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        print(f"Dataset already exists at {output_path}. Skipping download.")
        return

    print("Downloading ApacheJIT dataset...")
    print(f"Source: {DATASET_URL}\n")

    response = requests.get(DATASET_URL, stream=True)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Done. Saved to: {output_path}")


if __name__ == "__main__":
    download()