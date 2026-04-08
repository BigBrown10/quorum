from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AgentOutput:
    """A single agent response and its metadata."""

    id: str
    content: Any
    confidence: float | None = None
    sources: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DisagreementEdge:
    """A weighted disagreement relationship between two candidates."""

    source_id: str
    target_id: str
    weight: float


@dataclass(slots=True)
class ConsensusResult:
    """Consensus decision and diagnostics returned by quorum-core."""

    consensus_answer: Any
    consensus_cluster_id: str
    selected_agent_ids: list[str]
    agreement_score: float
    supporting_candidate_count: int
    total_candidates: int
    unstable: bool
    mode: str
    disagreement_edges: list[DisagreementEdge] = field(default_factory=list)
    rationale: str = ""
