import sys
import types
from typing import Any

import pytest

from quorum_core import AgentOutput, SentenceTransformerBackend, TfidfEmbeddingBackend, embed_texts
from quorum_core.graph import cluster_candidates


def test_hash_embedding_backend_is_removed() -> None:
    with pytest.raises(ImportError):
        exec("from quorum_core.embeddings import HashEmbeddingBackend", {})


def test_embed_texts_uses_default_backend() -> None:
    vectors = embed_texts(["alpha", "beta"])

    assert len(vectors) == 2
    assert len(vectors[0]) > 0
    assert vectors[0] != vectors[1]


def test_tfidf_embedding_backend_groups_similar_texts() -> None:
    backend = TfidfEmbeddingBackend()

    vectors = backend.embed(["the quick brown fox", "the quick brown fox jumps"])

    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1])
    assert vectors[0] != vectors[1]


def test_sentence_transformer_backend_lazy_loads_once(monkeypatch) -> None:
    calls: list[list[str]] = []

    class _EncodedVectors:
        def __init__(self, values: list[list[float]]) -> None:
            self._values = values

        def tolist(self) -> list[list[float]]:
            return self._values

    class FakeSentenceTransformer:
        init_count = 0

        def __init__(self, model_name: str) -> None:
            type(self).init_count += 1
            self.model_name = model_name

        def encode(
            self,
            texts: list[str],
            *,
            normalize_embeddings: bool,
            convert_to_numpy: bool,
        ) -> _EncodedVectors:
            calls.append(list(texts))
            assert normalize_embeddings is True
            assert convert_to_numpy is True
            if len(texts) == 1:
                return _EncodedVectors([[0.5, 0.5]])
            return _EncodedVectors([[0.1, 0.9], [0.2, 0.8]])

    module: Any = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    backend = SentenceTransformerBackend(model_name="demo-model")

    first = backend.embed(["alpha", "beta"])
    second = backend.embed(["gamma"])

    assert first == [[0.1, 0.9], [0.2, 0.8]]
    assert second == [[0.5, 0.5]]
    assert calls == [["alpha", "beta"], ["gamma"]]
    assert FakeSentenceTransformer.init_count == 1


def test_cluster_candidates_uses_text_fallback_when_embeddings_missing() -> None:
    candidates = [
        AgentOutput(id="a1", content="same answer"),
        AgentOutput(id="a2", content="same answer"),
        AgentOutput(id="a3", content="different answer"),
    ]

    clusters = cluster_candidates(candidates)

    assert any(set(cluster) == {0, 1} for cluster in clusters)
