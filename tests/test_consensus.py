import pytest

from quorum_core import AgentOutput, resolve_consensus


def test_simple_majority_picks_most_common_answer() -> None:
    candidates = [
        AgentOutput(id="a1", content="42", confidence=0.4),
        AgentOutput(id="a2", content="42", confidence=0.8),
        AgentOutput(id="a3", content="7", confidence=0.9),
    ]

    result = resolve_consensus(candidates, mode="simple_majority")

    assert result.consensus_answer == "42"
    assert result.selected_agent_ids == ["a1", "a2"]
    assert result.supporting_candidate_count == 2
    assert result.total_candidates == 3
    assert result.unstable is False
    assert result.agreement_score == 2 / 3
    assert result.disagreement_edges


def test_weighted_majority_uses_confidence_sum() -> None:
    candidates = [
        AgentOutput(id="a1", content="alpha", confidence=0.2),
        AgentOutput(id="a2", content="beta", confidence=0.9),
        AgentOutput(id="a3", content="beta", confidence=0.3),
    ]

    result = resolve_consensus(candidates, mode="weighted_majority")

    assert result.consensus_answer == "beta"
    assert result.selected_agent_ids == ["a2", "a3"]
    assert result.supporting_candidate_count == 2
    assert result.unstable is False
    assert result.agreement_score == pytest.approx((0.9 + 0.3) / (0.2 + 0.9 + 0.3))
    assert result.disagreement_edges


def test_unstable_result_is_explicit_when_every_answer_is_unique() -> None:
    candidates = [
        AgentOutput(id="a1", content="one", confidence=0.9),
        AgentOutput(id="a2", content="two", confidence=0.8),
        AgentOutput(id="a3", content="three", confidence=0.7),
    ]

    result = resolve_consensus(candidates, mode="weighted_majority")

    assert result.unstable is True
    assert result.consensus_answer == "NO CONSENSUS"
    assert result.selected_agent_ids == []
    assert result.consensus_cluster_id == "unstable"


def test_quantum_ready_falls_back_to_weighted_baseline_for_now() -> None:
    candidates = [
        AgentOutput(id="a1", content={"answer": 12}, confidence=0.1),
        AgentOutput(id="a2", content={"answer": 12}, confidence=0.7),
    ]

    result = resolve_consensus(candidates, mode="quantum_ready")

    assert result.consensus_answer == {"answer": 12}
    assert result.mode == "quantum_ready"
    assert result.unstable is False


def test_quantum_ready_prefers_stable_structured_answers_without_embeddings() -> None:
    candidates = [
        AgentOutput(id="a1", content={"answer": 12}, confidence=0.1),
        AgentOutput(id="a2", content={"answer": 12}, confidence=0.7),
        AgentOutput(id="a3", content={"answer": 99}, confidence=0.9),
    ]

    result = resolve_consensus(candidates, mode="quantum_ready")

    assert result.consensus_answer == {"answer": 12}
    assert result.unstable is False
    assert set(result.selected_agent_ids) == {"a1", "a2"}
    assert result.supporting_candidate_count == 2


def test_graph_min_cut_uses_distinct_mode_and_configurable_threshold() -> None:
    candidates = [
        AgentOutput(id="a1", content="shared answer", confidence=0.4),
        AgentOutput(id="a2", content="shared answer", confidence=0.6),
        AgentOutput(id="a3", content="different answer", confidence=0.9),
    ]

    stable_result = resolve_consensus(candidates, mode="graph_min_cut")
    strict_result = resolve_consensus(candidates, mode="graph_min_cut", unstable_threshold=0.99)

    assert stable_result.mode == "graph_min_cut"
    assert stable_result.disagreement_edges
    assert strict_result.unstable is True
