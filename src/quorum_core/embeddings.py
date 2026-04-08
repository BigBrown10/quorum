from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Protocol

from sklearn.feature_extraction.text import TfidfVectorizer


class EmbeddingBackend(Protocol):
    """Backend interface for turning text into numeric embeddings."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


@dataclass(slots=True)
class HashEmbeddingBackend:
    dimensions: int = 16

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vectors.append(self._embed_one(text))
        return vectors

    def _embed_one(self, text: str) -> list[float]:
        values = [0.0] * self.dimensions
        normalized = text.strip().lower()
        if not normalized:
            return values

        digest = hashlib.sha256(normalized.encode("utf-8")).digest()
        for index in range(self.dimensions):
            byte_value = digest[index % len(digest)]
            values[index] = (byte_value / 255.0) * 2.0 - 1.0

        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:
            return values
        return [value / norm for value in values]


@dataclass(slots=True)
class TfidfEmbeddingBackend:
    min_ngram: int = 3
    max_ngram: int = 5

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        prepared_texts = [text.strip().lower() or "<empty>" for text in texts]
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(self.min_ngram, self.max_ngram),
        )
        matrix = vectorizer.fit_transform(prepared_texts)
        return matrix.toarray().tolist()


def embed_texts(texts: list[str], backend: EmbeddingBackend | None = None) -> list[list[float]]:
    active_backend = backend or TfidfEmbeddingBackend()
    return active_backend.embed(texts)
