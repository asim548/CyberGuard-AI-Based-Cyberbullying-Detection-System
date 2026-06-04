"""
Download cyberbullying dataset from Kaggle via kagglehub.

Usage:
    python scripts/download_kaggle_dataset.py

Requires Kaggle credentials (kaggle.json in ~/.kaggle/ or KAGGLE_API_TOKEN).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DATA_DIR, DATASET_FILENAME  # noqa: E402

DATASET_SLUG = "soorajtomar/cyberbullying-tweets"
SOURCE_FILENAME = "CyberBullying Comments Dataset.csv"


def download_dataset() -> Path:
    import kagglehub

    cache_path = Path(
        kagglehub.dataset_download(DATASET_SLUG)
    )
    print("Path to dataset files:", cache_path)

    source = cache_path / SOURCE_FILENAME
    if not source.exists():
        csv_files = list(cache_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV found in {cache_path}")
        source = csv_files[0]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / DATASET_FILENAME
    shutil.copy2(source, dest)
    print(f"Copied to {dest}")
    return dest


if __name__ == "__main__":
    download_dataset()
