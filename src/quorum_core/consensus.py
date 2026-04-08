from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from .graph import (
    build_disagreement_graph,
    build_qubo_problem,
    consensus_cluster_from_indices,
)
from .models import AgentOutput, ConsensusResult
from .optimizer import GreedyOptimizer
from .quantum import QuantumBackendUnavailableError, get_optimizer

ConsensusMode = Literal[
    "simple_majority",
    "weighted_majority",
    "graph_min_cut",
    "quantum_ready",
]


@dataclass(slots=True)
class _CandidateGroup:
    normalized_content: str
    canonical_content: Any
    agent_ids: list[str]
    confidences: list[float]
    first_index: int

    @property
    def count(self) -> int:
        return len(self.agent_ids)

    @property
    def confidence_sum(self) -> float:
        return sum(self.confidences)


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    try:
        return json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except TypeError:
        return str(content).strip()


def _candidate_confidence(candidate: AgentOutput) -> float:
    if candidate.confidence is None:
        return 1.0
    return max(0.0, min(1.0, float(candidate.confidence)))


def _group_candidates(candidates: list[AgentOutput]) -> list[_CandidateGroup]:
    groups: dict[str, _CandidateGroup] = {}

    for index, candidate in enumerate(candidates):
        normalized = _normalize_content(candidate.content)
        confidence = _candidate_confidence(candidate)

        if normalized not in groups:
            groups[normalized] = _CandidateGroup(
                normalized_content=normalized,
                canonical_content=candidate.content,
                agent_ids=[candidate.id],
                confidences=[confidence],
                first_index=index,
            )
            continue

        group = groups[normalized]
        group.agent_ids.append(candidate.id)
        group.confidences.append(confidence)

    return sorted(groups.values(), key=lambda group: group.first_index)


def _build_result(
    *,
    mode: ConsensusMode,
    winner: _CandidateGroup,
    total_candidates: int,
    score: float,
    unstable: bool,
    rationale: str,
) -> ConsensusResult:
    if unstable:
        return ConsensusResult(
            consensus_answer="NO CONSENSUS",
            consensus_cluster_id="unstable",
            selected_agent_ids=[],
            agreement_score=score,
            supporting_candidate_count=winner.count,
            total_candidates=total_candidates,
            unstable=True,
            mode=mode,
            disagreement_edges=[],
            rationale=rationale,
        )

    return ConsensusResult(
        consensus_answer=winner.canonical_content,
        consensus_cluster_id=f"cluster_{winner.first_index}",
        selected_agent_ids=winner.agent_ids,
        agreement_score=score,
        supporting_candidate_count=winner.count,
        total_candidates=total_candidates,
        unstable=False,
        mode=mode,
        disagreement_edges=[],
        rationale=rationale,
    )


def _simple_majority(candidates: list[AgentOutput], *, mode: ConsensusMode) -> ConsensusResult:
    groups = _group_candidates(candidates)
    if not groups:
        raise ValueError("resolve_consensus() requires at least one candidate")

    winner = max(
        groups,
        key=lambda group: (group.count, group.confidence_sum, -group.first_index),
    )
    score = winner.count / len(candidates)
    unstable = winner.count == 1 and len(candidates) > 1
    rationale = "majority vote"
    if unstable:
        rationale = "no repeated answer reached the majority threshold"
    return _build_result(
        mode=mode,
        winner=winner,
        total_candidates=len(candidates),
        score=score,
        unstable=unstable,
        rationale=rationale,
    )


def _weighted_majority(candidates: list[AgentOutput], *, mode: ConsensusMode) -> ConsensusResult:
    groups = _group_candidates(candidates)
    if not groups:
        raise ValueError("resolve_consensus() requires at least one candidate")

    winner = max(
        groups,
        key=lambda group: (group.confidence_sum, group.count, -group.first_index),
    )
    total_weight = sum(group.confidence_sum for group in groups) or 1.0
    score = winner.confidence_sum / total_weight
    unstable = winner.count == 1 and len(candidates) > 1
    rationale = "confidence-weighted majority"
    if unstable:
        rationale = "no repeated answer carried enough support to be considered stable"
    return _build_result(
        mode=mode,
        winner=winner,
        total_candidates=len(candidates),
        score=score,
        unstable=unstable,
        rationale=rationale,
    )


def _graph_or_quantum_ready(
    candidates: list[AgentOutput],
    *,
    mode: ConsensusMode,
) -> ConsensusResult:
    if not candidates:
        raise ValueError("resolve_consensus() requires at least one candidate")

    disagreement_edges = build_disagreement_graph(candidates)
    problem = build_qubo_problem(candidates)
    try:
        optimizer = get_optimizer()
        solution = optimizer.solve(problem)
    except QuantumBackendUnavailableError:
        optimizer = GreedyOptimizer()
        solution = optimizer.solve(problem)

    if not solution.selected_indices:
        return ConsensusResult(
            consensus_answer="NO CONSENSUS",
            consensus_cluster_id="unstable",
            selected_agent_ids=[],
            agreement_score=0.0,
            supporting_candidate_count=0,
            total_candidates=len(candidates),
            unstable=True,
            mode=mode,
            disagreement_edges=disagreement_edges,
            rationale="optimizer returned no stable subset",
        )

    cluster = consensus_cluster_from_indices(candidates, solution.selected_indices)
    total_possible = max(1.0, len(candidates) + len(disagreement_edges))
    agreement_score = max(0.0, min(1.0, 1.0 - (solution.energy / total_possible)))
    unstable = agreement_score < 0.45 or len(cluster.agent_ids) == 1 and len(candidates) > 1

    if unstable:
        return ConsensusResult(
            consensus_answer="NO CONSENSUS",
            consensus_cluster_id="unstable",
            selected_agent_ids=[],
            agreement_score=agreement_score,
            supporting_candidate_count=len(cluster.agent_ids),
            total_candidates=len(candidates),
            unstable=True,
            mode=mode,
            disagreement_edges=disagreement_edges,
            rationale="graph-based optimizer could not find a stable consensus",
        )

    return ConsensusResult(
        consensus_answer=cluster.consensus_content,
        consensus_cluster_id=cluster.cluster_id,
        selected_agent_ids=cluster.agent_ids,
        agreement_score=agreement_score,
        supporting_candidate_count=len(cluster.agent_ids),
        total_candidates=len(candidates),
        unstable=False,
        mode=mode,
        disagreement_edges=disagreement_edges,
        rationale="graph-based consensus",
    )


def resolve_consensus(
    candidates: list[AgentOutput],
    mode: ConsensusMode = "quantum_ready",
) -> ConsensusResult:
    """Resolve a consensus answer from candidate agent outputs.

    The initial implementation covers majority-based baselines and keeps the
    result format ready for graph/QUBO-based optimizers later.
    """

    if mode == "simple_majority":
        return _simple_majority(candidates, mode=mode)
    if mode == "weighted_majority":
        return _weighted_majority(candidates, mode=mode)
    if mode == "graph_min_cut":
        return _graph_or_quantum_ready(candidates, mode=mode)
    if mode == "quantum_ready":
        return _graph_or_quantum_ready(candidates, mode=mode)

    raise ValueError(f"Unknown consensus mode: {mode}")
