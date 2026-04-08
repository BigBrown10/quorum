from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from sklearn.feature_extraction.text import TfidfVectorizer


class EmbeddingBackend(Protocol):
    """Backend interface for turning text into numeric embeddings."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


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


@dataclass(slots=True)
class SentenceTransformerBackend:
    model_name: str = "all-MiniLM-L6-v2"
    _model: Any = field(default=None, init=False, repr=False, compare=False)

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - optional dependency path
                raise ImportError(
                    "sentence-transformers is required for SentenceTransformerBackend"
                ) from exc

            self._model = SentenceTransformer(self.model_name)

        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        model = self._get_model()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.tolist()


def embed_texts(texts: list[str], backend: EmbeddingBackend | None = None) -> list[list[float]]:
    active_backend = backend or TfidfEmbeddingBackend()
    return active_backend.embed(texts)
