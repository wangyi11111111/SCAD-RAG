"""Lightweight text processing utilities."""

from __future__ import annotations

import re

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def normalize_text(text: str) -> str:
    """Normalize whitespace and case."""
    return re.sub(r"\s+", " ", text.strip().lower())


def split_sentences(text: str) -> list[str]:
    """Split English text into sentence-level claims."""
    clean = re.sub(r"\s+", " ", text.strip())
    if not clean:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", clean) if part.strip()]


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    """Tokenize text into lowercase words and numbers."""
    tokens = re.findall(r"[a-zA-Z]+|\d+(?:\.\d+)?", text.lower())
    return tokens if keep_stopwords else [token for token in tokens if token not in STOPWORDS]


def token_set(text: str, keep_stopwords: bool = False) -> set[str]:
    """Return a token set."""
    return set(tokenize(text, keep_stopwords=keep_stopwords))


def extract_numbers(text: str) -> set[str]:
    """Extract numeric strings."""
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", text.lower()))


def contains_phrase(text: str, phrase: str) -> bool:
    """Check whether a phrase appears with word boundaries."""
    return re.search(rf"\b{re.escape(phrase.lower())}\b", text.lower()) is not None
