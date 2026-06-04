"""Inference helpers for CyberGuard."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from config import LABELS, MODEL_BUNDLE
from src.preprocess import preprocess_text


def load_model_bundle(path: Path | None = None) -> dict:
    path = path or MODEL_BUNDLE
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run: python -m src.train"
        )
    return joblib.load(path)


def predict_text(text: str, bundle: dict | None = None) -> dict:
    bundle = bundle or load_model_bundle()
    pipeline = bundle["pipeline"]
    labels = bundle.get("labels", LABELS)

    processed = preprocess_text(text)
    if not processed.strip():
        return {
            "label": "Safe",
            "confidence": 0.5,
            "probabilities": {lbl: 1.0 / len(labels) for lbl in labels},
            "processed_text": processed,
            "warning": "Empty or unrecognizable text after preprocessing.",
        }

    proba = pipeline.predict_proba([processed])[0]
    classes = list(pipeline.classes_)
    idx = int(np.argmax(proba))

    return {
        "label": classes[idx],
        "confidence": float(proba[idx]),
        "probabilities": {classes[i]: float(proba[i]) for i in range(len(classes))},
        "processed_text": processed,
    }


def predict_batch(texts: list[str], bundle: dict | None = None) -> list[dict]:
    bundle = bundle or load_model_bundle()
    return [predict_text(t, bundle) for t in texts]
