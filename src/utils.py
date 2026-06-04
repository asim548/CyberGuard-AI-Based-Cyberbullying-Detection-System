"""Dataset loading and label mapping utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import DATA_DIR, DATASET_FILENAME, KAGGLE_LABEL_MAP, LABELS, METRICS_FILE

# Heuristic keywords to split binary bullying (1) into Severe vs Toxic
SEVERE_KEYWORDS = (
    "kill",
    "die",
    "suicide",
    "murder",
    "hurt you",
    "beat you",
    "rape",
    "hope you die",
    "kill yourself",
    "kys",
    "worthless",
    "deserve to die",
    "end your life",
    "gonna hurt",
    "will hurt",
    "burn in hell",
)


def normalize_label(raw_label: str) -> str:
    key = str(raw_label).strip().lower().replace("-", "_").replace(" ", "_")
    if key in KAGGLE_LABEL_MAP:
        return KAGGLE_LABEL_MAP[key]
    for partial, mapped in KAGGLE_LABEL_MAP.items():
        if partial in key:
            return mapped
    if key in ("0", "1"):
        return "Safe" if key == "0" else "Toxic"
    return "Toxic"


def map_binary_label(text: str, raw_label) -> str:
    """Map CB_Label 0/1 (soorajtomar dataset) to 3-class labels."""
    try:
        is_bullying = int(float(raw_label)) == 1
    except (TypeError, ValueError):
        return normalize_label(raw_label)

    if not is_bullying:
        return "Safe"

    lowered = str(text).lower()
    if any(kw in lowered for kw in SEVERE_KEYWORDS):
        return "Severe Bullying"
    return "Toxic"


def detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    col_map = {c.lower(): c for c in df.columns}
    text_candidates = [
        "text",
        "tweet",
        "tweet_text",
        "comment",
        "message",
        "content",
    ]
    label_candidates = [
        "label",
        "class",
        "category",
        "cyberbullying_type",
        "target",
        "cb_label",
    ]

    text_col = next((col_map[c] for c in text_candidates if c in col_map), None)
    label_col = next((col_map[c] for c in label_candidates if c in col_map), None)

    if text_col is None:
        text_col = df.columns[0]
    if label_col is None:
        label_col = df.columns[-1] if len(df.columns) > 1 else df.columns[0]

    return text_col, label_col


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    path = path or (DATA_DIR / DATASET_FILENAME)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Place cyberbullying_tweets.csv in the data/ folder "
            "(download from Kaggle: Cyberbullying Tweet Dataset)."
        )

    df = pd.read_csv(path)
    text_col, label_col = detect_columns(df)

    label_series = df[label_col]
    is_binary = label_series.astype(str).str.strip().isin({"0", "1", "0.0", "1.0"}).all()

    if is_binary or label_col.lower() == "cb_label":
        labels = [
            map_binary_label(t, lbl)
            for t, lbl in zip(df[text_col].astype(str), label_series)
        ]
    else:
        labels = label_series.apply(normalize_label)

    out = pd.DataFrame({"text": df[text_col].astype(str), "label": labels})
    out = out[out["text"].str.len() > 0]
    out = out[out["label"].isin(LABELS)]
    return out.reset_index(drop=True)


def save_metrics(metrics: dict) -> None:
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def load_metrics() -> dict | None:
    if not METRICS_FILE.exists():
        return None
    with open(METRICS_FILE, encoding="utf-8") as f:
        return json.load(f)
