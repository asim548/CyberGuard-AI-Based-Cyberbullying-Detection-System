from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

DATASET_FILENAME = "cyberbullying_tweets.csv"
MODEL_BUNDLE = MODELS_DIR / "cyberguard_model.joblib"
METRICS_FILE = MODELS_DIR / "training_metrics.json"

LABELS = ["Safe", "Toxic", "Severe Bullying"]
LABEL_COLORS = {
    "Safe": "#22c55e",
    "Toxic": "#f59e0b",
    "Severe Bullying": "#ef4444",
}
LABEL_ICONS = {
    "Safe": "shield-check",
    "Toxic": "alert-triangle",
    "Severe Bullying": "shield-x",
}

# Map Kaggle multi-class labels to proposal's 3 categories
KAGGLE_LABEL_MAP = {
    "not_cyberbullying": "Safe",
    "not cyberbullying": "Safe",
    "no": "Safe",
    "0": "Safe",
    "other_cyberbullying": "Toxic",
    "other cyberbullying": "Toxic",
    "other": "Toxic",
    "gender": "Severe Bullying",
    "religion": "Severe Bullying",
    "ethnicity": "Severe Bullying",
    "age": "Severe Bullying",
    "severe": "Severe Bullying",
    "severe bullying": "Severe Bullying",
    "toxic": "Toxic",
    "safe": "Safe",
    "0": "Safe",
    "1": "Toxic",
}
