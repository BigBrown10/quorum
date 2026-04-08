"""Public API for quorum-core."""

from .api import resolve_consensus_payload
from .consensus import resolve_consensus
from .embeddings import HashEmbeddingBackend, embed_texts
from .graph import ConsensusCluster, build_disagreement_graph, build_qubo_problem
from .models import AgentOutput, ConsensusResult, DisagreementEdge
from .quantum import DWaveOptimizer, QiskitOptimizer, QuantumBackendUnavailableError, get_optimizer

__all__ = [
    "AgentOutput",
    "ConsensusCluster",
    "ConsensusResult",
    "DisagreementEdge",
    "HashEmbeddingBackend",
    "DWaveOptimizer",
    "build_disagreement_graph",
    "build_qubo_problem",
    "embed_texts",
    "QiskitOptimizer",
    "QuantumBackendUnavailableError",
    "get_optimizer",
    "resolve_consensus_payload",
    "resolve_consensus",
]
