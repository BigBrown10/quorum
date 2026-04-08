from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from .embeddings import embed_texts
from .models import AgentOutput, DisagreementEdge
from .optimizer import OptimizationProblem


@dataclass(slots=True)
class ConsensusCluster:
    cluster_id: str
    agent_ids: list[str]
    consensus_content: Any
    agreement_score: float
    centroid_embedding: list[float] | None = None


def _normalize_embedding(embedding: list[float] | None) -> np.ndarray | None:
    if embedding is None:
        return None

    vector = np.asarray(embedding, dtype=float)
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return None
    return vector / norm


def cosine_similarity(left: list[float] | None, right: list[float] | None) -> float:
    left_vector = _normalize_embedding(left)
    right_vector = _normalize_embedding(right)
    if left_vector is None or right_vector is None:
        return 0.0
    return float(np.clip(np.dot(left_vector, right_vector), -1.0, 1.0))


def cluster_candidates(
    candidates: list[AgentOutput],
    similarity_threshold: float = 0.75,
) -> list[list[int]]:
    if len(candidates) <= 1:
        return [[0]] if candidates else []

    embeddings = _candidate_embeddings(candidates)

    matrix = np.vstack([vector for vector in embeddings if vector is not None])
    distances = 1.0 - np.clip(matrix @ matrix.T, -1.0, 1.0)

    clustering = AgglomerativeClustering(
        metric="precomputed",
        linkage="average",
        distance_threshold=max(0.0, 1.0 - similarity_threshold),
        n_clusters=None,
    )
    labels = clustering.fit_predict(distances)

    clusters: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(index)

    return [clusters[label] for label in sorted(clusters)]


def build_disagreement_graph(candidates: list[AgentOutput]) -> list[DisagreementEdge]:
    edges: list[DisagreementEdge] = []

    for left_index in range(len(candidates)):
        for right_index in range(left_index + 1, len(candidates)):
            left = candidates[left_index]
            right = candidates[right_index]
            similarity = _candidate_similarity(left, right)
            confidence_factor = ((_candidate_confidence(left) + _candidate_confidence(right)) / 2.0)
            weight = max(0.0, (1.0 - similarity) * (0.5 + 0.5 * confidence_factor))
            if weight > 0:
                edges.append(
                    DisagreementEdge(
                        source_id=left.id,
                        target_id=right.id,
                        weight=weight,
                    )
                )

    return edges


def build_qubo_problem(candidates: list[AgentOutput]) -> OptimizationProblem:
    labels = [candidate.id for candidate in candidates]
    unary_costs: list[float] = []
    pairwise_costs: dict[tuple[int, int], float] = {}
    content_counts = _content_counts(candidates)

    for candidate in candidates:
        confidence = _candidate_confidence(candidate)
        source_bonus = min(len(candidate.sources) * 0.05, 0.15)
        support_count = content_counts[_normalize_content(candidate.content)]
        support_bonus = min(max(support_count - 1, 0) * 0.35, 0.7)
        uniqueness_penalty = 0.9 if support_count == 1 else 0.0
        unary_costs.append(-(confidence + source_bonus + support_bonus) + uniqueness_penalty)

    for left_index in range(len(candidates)):
        for right_index in range(left_index + 1, len(candidates)):
            similarity = _candidate_similarity(candidates[left_index], candidates[right_index])
            disagreement = max(0.0, 1.0 - similarity)
            if disagreement > 0:
                pairwise_costs[(left_index, right_index)] = disagreement

    return OptimizationProblem(
        labels=labels,
        unary_costs=unary_costs,
        pairwise_costs=pairwise_costs,
        metadata={
            "candidate_count": len(candidates),
            "disagreement_edges": [
                {
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "weight": edge.weight,
                }
                for edge in build_disagreement_graph(candidates)
            ],
        },
    )


def minimum_disagreement_cut(candidates: list[AgentOutput]) -> tuple[list[int], float]:
    if not candidates:
        return [], 0.0
    if len(candidates) == 1:
        return [0], 0.0

    edges = build_disagreement_graph(candidates)
    index_by_id = {candidate.id: index for index, candidate in enumerate(candidates)}
    adjacency = [[0.0 for _ in candidates] for _ in candidates]

    for edge in edges:
        left_index = index_by_id[edge.source_id]
        right_index = index_by_id[edge.target_id]
        adjacency[left_index][right_index] += edge.weight
        adjacency[right_index][left_index] += edge.weight

    supernodes = [set([index]) for index in range(len(candidates))]
    active = list(range(len(candidates)))
    best_cut_weight = float("inf")
    best_partition = [0]

    while len(active) > 1:
        used: set[int] = set()
        weights = {index: 0.0 for index in active}
        previous = None

        for _ in range(len(active)):
            current = max(
                (index for index in active if index not in used),
                key=lambda index: (weights[index], -min(supernodes[index]), -index),
            )
            used.add(current)

            if len(used) == len(active):
                cut_weight = weights[current]
                if cut_weight < best_cut_weight:
                    best_cut_weight = cut_weight
                    best_partition = sorted(supernodes[current])

                if previous is not None:
                    for neighbor in active:
                        if neighbor in {previous, current}:
                            continue
                        adjacency[previous][neighbor] += adjacency[current][neighbor]
                        adjacency[neighbor][previous] = adjacency[previous][neighbor]
                    supernodes[previous].update(supernodes[current])
                    active.remove(current)
                break

            previous = current
            for neighbor in active:
                if neighbor not in used:
                    weights[neighbor] += adjacency[current][neighbor]

    return best_partition, best_cut_weight


def _candidate_confidence(candidate: AgentOutput) -> float:
    if candidate.confidence is None:
        return 1.0
    return max(0.0, min(1.0, float(candidate.confidence)))


def _content_counts(candidates: list[AgentOutput]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        key = _normalize_content(candidate.content)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _candidate_similarity(left: AgentOutput, right: AgentOutput) -> float:
    similarity = cosine_similarity(left.embedding, right.embedding)
    if left.embedding is None and right.embedding is None:
        left_vector, right_vector = embed_texts([
            _normalize_content(left.content),
            _normalize_content(right.content),
        ])
        return cosine_similarity(left_vector, right_vector)
    return similarity


def _candidate_embeddings(candidates: list[AgentOutput]) -> list[np.ndarray | None]:
    raw_embeddings = [candidate.embedding for candidate in candidates]
    if all(embedding is not None for embedding in raw_embeddings):
        return [_normalize_embedding(embedding) for embedding in raw_embeddings]

    texts = [_normalize_content(candidate.content) for candidate in candidates]
    derived_embeddings = embed_texts(texts)
    return [_normalize_embedding(vector) for vector in derived_embeddings]


def score_selected_indices(
    selected_indices: list[int],
    problem: OptimizationProblem,
) -> float:
    selected = set(selected_indices)
    total = 0.0

    for index in selected:
        total += problem.unary_costs[index]

    for (left_index, right_index), cost in problem.pairwise_costs.items():
        if left_index in selected and right_index in selected:
            total += cost

    return total


def consensus_cluster_from_indices(
    candidates: list[AgentOutput],
    selected_indices: list[int],
) -> ConsensusCluster:
    selected_candidates = [candidates[index] for index in selected_indices]
    if not selected_candidates:
        return ConsensusCluster(
            cluster_id="unstable",
            agent_ids=[],
            consensus_content="NO CONSENSUS",
            agreement_score=0.0,
            centroid_embedding=None,
        )

    centroid_embedding: list[float] | None = None
    embeddings = [
        candidate.embedding
        for candidate in selected_candidates
        if candidate.embedding is not None
    ]
    if embeddings:
        matrix = np.asarray(embeddings, dtype=float)
        centroid = matrix.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid_embedding = (centroid / norm).tolist()

    first_candidate = selected_candidates[0]
    if all(
        _normalize_content(candidate.content)
        == _normalize_content(first_candidate.content)
        for candidate in selected_candidates
    ):
        consensus_content: Any = first_candidate.content
    else:
        consensus_content = first_candidate.content

    average_confidence = sum(
        _candidate_confidence(candidate) for candidate in selected_candidates
    ) / len(selected_candidates)
    return ConsensusCluster(
        cluster_id=f"cluster_{selected_indices[0]}",
        agent_ids=[candidate.id for candidate in selected_candidates],
        consensus_content=consensus_content,
        agreement_score=average_confidence,
        centroid_embedding=centroid_embedding,
    )


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()
