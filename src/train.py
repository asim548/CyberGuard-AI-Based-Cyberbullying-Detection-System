"""Train TF-IDF + Logistic Regression classifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import LABELS, MODEL_BUNDLE, MODELS_DIR  # noqa: E402
from src.preprocess import preprocess_text  # noqa: E402
from src.utils import load_dataset, save_metrics  # noqa: E402


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=25000,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def train_model(
    data_path: Path | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> dict:
    df = load_dataset(data_path)
    print(f"Loaded {len(df):,} samples")
    print(df["label"].value_counts())

    df["processed"] = df["text"].apply(preprocess_text)
    df = df[df["processed"].str.len() > 0].reset_index(drop=True)

    X = df["processed"]
    y = df["label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, random_state=42, stratify=y
    )
    relative_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - relative_val, random_state=42, stratify=y_temp
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    def evaluate(split_name: str, X_split, y_split) -> dict:
        preds = pipeline.predict(X_split)
        return {
            "accuracy": float(accuracy_score(y_split, preds)),
            "f1_macro": float(f1_score(y_split, preds, average="macro", labels=LABELS)),
            "report": classification_report(
                y_split, preds, labels=LABELS, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_split, preds, labels=LABELS).tolist(),
        }

    metrics = {
        "samples": len(df),
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "labels": LABELS,
        "train": evaluate("train", X_train, y_train),
        "validation": evaluate("validation", X_val, y_val),
        "test": evaluate("test", X_test, y_test),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": pipeline,
        "labels": LABELS,
        "version": "1.0",
    }
    joblib.dump(bundle, MODEL_BUNDLE)
    save_metrics(metrics)

    print("\n=== Test Results ===")
    print(f"Accuracy: {metrics['test']['accuracy']:.4f}")
    print(f"F1 (macro): {metrics['test']['f1_macro']:.4f}")
    print(json.dumps(metrics["test"]["report"], indent=2))

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CyberGuard model")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to CSV dataset (default: data/cyberbullying_tweets.csv)",
    )
    args = parser.parse_args()
    train_model(args.data)


if __name__ == "__main__":
    main()
