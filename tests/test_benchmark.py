from quorum_core import AgentOutput, resolve_consensus


def test_graph_mode_beats_majority_on_synthetic_cluster() -> None:
    candidates = [
        AgentOutput(id="a1", content={"answer": "red"}, confidence=0.2, embedding=[1.0, 0.0]),
        AgentOutput(id="a2", content={"answer": "red"}, confidence=0.4, embedding=[0.98, 0.05]),
        AgentOutput(id="a3", content={"answer": "red"}, confidence=0.6, embedding=[0.97, 0.04]),
        AgentOutput(id="b1", content={"answer": "blue"}, confidence=0.95, embedding=[0.0, 1.0]),
    ]

    majority = resolve_consensus(candidates, mode="simple_majority")
    graph = resolve_consensus(candidates, mode="quantum_ready")

    assert majority.consensus_answer in ({"answer": "red"}, "NO CONSENSUS")
    assert graph.consensus_answer == {"answer": "red"}
    assert graph.unstable is False
    assert graph.supporting_candidate_count == 3
    assert graph.agreement_score >= majority.agreement_score
