from quorum_core import AgentOutput, HashEmbeddingBackend, TfidfEmbeddingBackend, embed_texts
from quorum_core.graph import cluster_candidates


def test_hash_embedding_backend_is_deterministic() -> None:
    backend = HashEmbeddingBackend(dimensions=8)

    first = backend.embed(["Hello World"])[0]
    second = backend.embed(["hello world"])[0]

    assert first == second
    assert len(first) == 8
    assert abs(sum(value * value for value in first) - 1.0) < 1e-6


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


def test_cluster_candidates_uses_text_fallback_when_embeddings_missing() -> None:
    candidates = [
        AgentOutput(id="a1", content="same answer"),
        AgentOutput(id="a2", content="same answer"),
        AgentOutput(id="a3", content="different answer"),
    ]

    clusters = cluster_candidates(candidates)

    assert any(set(cluster) == {0, 1} for cluster in clusters)
