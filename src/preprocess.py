"""Text preprocessing utilities using NLTK."""

from __future__ import annotations

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_LEMMATIZER: WordNetLemmatizer | None = None
_STOP_WORDS: set[str] | None = None
_NLTK_READY = False


def ensure_nltk_data() -> None:
    global _NLTK_READY
    if _NLTK_READY:
        return
    packages = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for path, name in packages:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)
    _NLTK_READY = True


def _get_lemmatizer() -> WordNetLemmatizer:
    global _LEMMATIZER
    ensure_nltk_data()
    if _LEMMATIZER is None:
        _LEMMATIZER = WordNetLemmatizer()
    return _LEMMATIZER


def _get_stop_words() -> set[str]:
    global _STOP_WORDS
    ensure_nltk_data()
    if _STOP_WORDS is None:
        _STOP_WORDS = set(stopwords.words("english"))
    return _STOP_WORDS


def clean_text(text: str) -> str:
    """Lowercase, remove URLs, mentions, hashtags symbols, punctuation, extra spaces."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_lemmatize(text: str, remove_stopwords: bool = True) -> str:
    """Return space-joined lemmatized tokens."""
    text = clean_text(text)
    if not text:
        return ""
    lemmatizer = _get_lemmatizer()
    stops = _get_stop_words() if remove_stopwords else set()
    tokens = []
    for word in text.split():
        if len(word) < 2:
            continue
        if word in stops:
            continue
        tokens.append(lemmatizer.lemmatize(word))
    return " ".join(tokens)


def preprocess_text(text: str) -> str:
    """Full preprocessing pipeline for model input."""
    return tokenize_and_lemmatize(text)
