from quorum_core import AgentOutput


def test_agent_output_defaults_are_independent() -> None:
    first = AgentOutput(id="a1", content="hello")
    second = AgentOutput(id="a2", content="world")

    first.sources.append("doc-1")
    first.stats["latency_ms"] = 12

    assert second.sources == []
    assert second.stats == {}
