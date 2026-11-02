from __future__ import annotations

import re

from .common import normalize_text


def sentences(text: str) -> list[str]:
    """A transparent sentence splitter that does not require a model download."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", normalize_text(text))
    return [part.strip() for part in parts if part.strip()]


def chunk_text(text: str, max_words: int = 180, overlap_sentences: int = 1) -> list[str]:
    """Sentence-aware chunks with overlapping trailing context."""
    sents = sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sents:
        length = len(sentence.split())
        if current and current_words + length > max_words:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_words = sum(len(s.split()) for s in current)
        current.append(sentence)
        current_words += length

    if current:
        chunks.append(" ".join(current))
    return chunks

