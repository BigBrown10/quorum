from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from quorum_core.consensus import ConsensusMode
from quorum_core.models import DisagreementEdge

VALID_CONSENSUS_MODES: set[str] = {
    "simple_majority",
    "weighted_majority",
    "graph_min_cut",
    "quantum_ready",
}


@dataclass(slots=True)
class QuorumXConfig:
    n_agents: int = 3
    max_rounds: int = 2
    stability_threshold: float = 0.7
    token_cap_per_agent_round: int = 150
    consensus_mode: ConsensusMode = "quantum_ready"
    model: str = "gpt-4o-mini"
    quorum_model: str | None = None
    temperature: float = 0.2
    backend: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    request_timeout_seconds: float = 30.0
    mock_mode: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.n_agents <= 5:
            raise ValueError("n_agents must be between 1 and 5")
        if not 1 <= self.max_rounds <= 3:
            raise ValueError("max_rounds must be between 1 and 3")
        if not 0.0 <= self.stability_threshold <= 1.0:
            raise ValueError("stability_threshold must be between 0 and 1")
        if self.token_cap_per_agent_round <= 0:
            raise ValueError("token_cap_per_agent_round must be positive")
        if self.consensus_mode not in VALID_CONSENSUS_MODES:
            raise ValueError(f"Unsupported consensus_mode: {self.consensus_mode}")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.quorum_model is not None and not self.quorum_model.strip():
            raise ValueError("quorum_model must not be empty when provided")
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")

        if self.backend is None:
            self.backend = "mock" if self.mock_mode else "openai"
        else:
            self.backend = self.backend.strip().lower()
            if not self.backend:
                raise ValueError("backend must not be empty when provided")
            self.mock_mode = self.backend == "mock"


@dataclass(slots=True)
class AgentBenchmark:
    agent_id: str
    stance: str
    rounds_used: int
    token_count: int
    confidence: float
    answer_preview: str


@dataclass(slots=True)
class QuorumXResult:
    answer: Any
    agreement_score: float
    unstable: bool
    rounds_used: int
    total_tokens: int
    tokens_per_round: list[int] = field(default_factory=list)
    benchmark: list[AgentBenchmark] = field(default_factory=list)
    disagreement_edges_final: list[DisagreementEdge] = field(default_factory=list)
    selected_agent_ids: list[str] = field(default_factory=list)
    consensus_mode: str = "quantum_ready"
    rationale: str = ""