from collections.abc import Callable
from types import SimpleNamespace

from quorum_core import ConsensusResult
from quorumx.adapters import (
    from_autogen_message,
    from_crewai_result,
    from_langchain_output,
    from_langgraph_node_output,
    from_openclaw_artifact,
    normalize_to_agent_output,
    run_autogen_consensus,
    run_consensus_round,
    run_crewai_consensus,
    run_langchain_consensus,
    run_langgraph_consensus,
    run_openclaw_consensus,
)


def test_normalize_to_agent_output_from_mapping() -> None:
    result = normalize_to_agent_output(
        {
            "id": "answer-1",
            "content": "Use retries",
            "confidence": 0.8,
            "sources": ["langchain"],
            "stats": {"round": 2},
        },
        default_id="candidate-1",
        source_tag="adapter",
    )

    assert result.id == "answer-1"
    assert result.content == "Use retries"
    assert result.confidence == 0.8
    assert result.sources == ["langchain", "adapter"]
    assert result.stats == {"round": 2}


def test_from_langchain_output_normalizes_message_object() -> None:
    result = from_langchain_output(
        SimpleNamespace(content=["Draft A", {"text": "with guardrails"}]),
        default_id="langchain-7",
    )

    assert result.id == "langchain-7"
    assert result.content == "Draft A with guardrails"
    assert result.sources == ["langchain"]


def test_from_crewai_result_uses_framework_specific_fields() -> None:
    result = from_crewai_result(
        {"final_answer": "Ship the patch", "confidence": "0.6"},
        default_id="crewai-2",
    )

    assert result.id == "crewai-2"
    assert result.content == "Ship the patch"
    assert result.confidence == 0.6
    assert result.sources == ["crewai"]


def test_from_autogen_message_and_langgraph_helpers() -> None:
    autogen_result = from_autogen_message(
        SimpleNamespace(text="Use a smaller model", score=0.9),
        default_id="autogen-4",
    )
    langgraph_result = from_langgraph_node_output(
        {"output": "Keep the adapter thin"},
        default_id="langgraph-3",
    )

    assert autogen_result.id == "autogen-4"
    assert autogen_result.content == "Use a smaller model"
    assert autogen_result.sources == ["autogen"]
    assert langgraph_result.id == "langgraph-3"
    assert langgraph_result.content == "Keep the adapter thin"
    assert langgraph_result.sources == ["langgraph"]


def test_from_openclaw_artifact_and_run_consensus_round() -> None:
    artifact_result = from_openclaw_artifact(
        {"artifact": "Generate the follow-up email"},
        default_id="openclaw-1",
    )
    consensus = run_consensus_round(
        [
            {"id": "a1", "content": "Use retries"},
            {"id": "a2", "content": "Use retries"},
            {"id": "a3", "content": "Use retries carefully"},
        ],
        mode="simple_majority",
    )

    assert artifact_result.id == "openclaw-1"
    assert artifact_result.content == "Generate the follow-up email"
    assert artifact_result.sources == ["openclaw"]
    assert consensus.consensus_answer == "Use retries"
    assert consensus.unstable is False
    assert len(consensus.selected_agent_ids) == 2


def test_framework_specific_consensus_wrappers() -> None:
    wrapper_cases: list[tuple[Callable[..., ConsensusResult], list[object]]] = [
        (
            run_langchain_consensus,
            [
                SimpleNamespace(content="Use retries"),
                SimpleNamespace(content="Use retries"),
                SimpleNamespace(content="Add guardrails"),
            ],
        ),
        (
            run_langgraph_consensus,
            [
                {"output": "Use retries"},
                {"output": "Use retries"},
                {"output": "Add guardrails"},
            ],
        ),
        (
            run_crewai_consensus,
            [
                {"final_answer": "Use retries"},
                {"final_answer": "Use retries"},
                {"final_answer": "Add guardrails"},
            ],
        ),
        (
            run_autogen_consensus,
            [
                SimpleNamespace(text="Use retries"),
                SimpleNamespace(text="Use retries"),
                SimpleNamespace(text="Add guardrails"),
            ],
        ),
        (
            run_openclaw_consensus,
            [
                {"artifact": "Use retries"},
                {"artifact": "Use retries"},
                {"artifact": "Add guardrails"},
            ],
        ),
    ]

    for wrapper, outputs in wrapper_cases:
        result = wrapper(outputs, mode="simple_majority")

        assert result.consensus_answer == "Use retries"
        assert result.unstable is False
        assert len(result.selected_agent_ids) == 2