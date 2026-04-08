from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from quorum_core import AgentOutput, ConsensusResult, resolve_consensus
from quorum_core.consensus import ConsensusMode

__all__ = [
    "from_autogen_message",
    "from_crewai_result",
    "from_langchain_output",
    "from_langgraph_node_output",
    "from_openclaw_artifact",
    "normalize_candidates",
    "normalize_to_agent_output",
    "run_autogen_consensus",
    "run_crewai_consensus",
    "run_langchain_consensus",
    "run_langgraph_consensus",
    "run_openclaw_consensus",
    "run_consensus_round",
]


def normalize_to_agent_output(
    item: Any,
    *,
    default_id: str,
    id_field: str = "id",
    content_fields: Sequence[str] = (
        "content",
        "text",
        "output",
        "response",
        "answer",
    ),
    confidence_field: str | None = "confidence",
    sources_field: str = "sources",
    stats_field: str = "stats",
    source_tag: str | None = None,
) -> AgentOutput:
    identifier = _coerce_identifier(
        _first_value(item, (id_field, "name", "agent_id", "role"))
    )
    if not identifier:
        identifier = default_id

    content = _first_value(item, content_fields)
    if content is None:
        content = item

    sources = _coerce_sources(_first_value(item, (sources_field, "source", "framework")))
    if source_tag and source_tag not in sources:
        sources.append(source_tag)

    confidence = None
    if confidence_field is not None:
        confidence = _coerce_confidence(
            _first_value(item, (confidence_field, "score", "certainty"))
        )

    stats = _coerce_stats(_first_value(item, (stats_field, "metadata", "details", "extra")))

    return AgentOutput(
        id=identifier,
        content=_coerce_content(content),
        confidence=confidence,
        sources=sources,
        stats=stats,
    )


def normalize_candidates(
    outputs: Iterable[Any],
    *,
    prefix: str = "candidate",
    **normalize_kwargs: Any,
) -> list[AgentOutput]:
    return [
        normalize_to_agent_output(
            item,
            default_id=f"{prefix}-{index}",
            **normalize_kwargs,
        )
        for index, item in enumerate(outputs, start=1)
    ]


def run_consensus_round(
    outputs: Iterable[Any],
    *,
    mode: ConsensusMode = "quantum_ready",
    unstable_threshold: float = 0.45,
    prefix: str = "candidate",
    **normalize_kwargs: Any,
) -> ConsensusResult:
    normalized = normalize_candidates(outputs, prefix=prefix, **normalize_kwargs)
    return resolve_consensus(normalized, mode=mode, unstable_threshold=unstable_threshold)


def run_langchain_consensus(
    outputs: Iterable[Any],
    *,
    mode: ConsensusMode = "quantum_ready",
    unstable_threshold: float = 0.45,
    prefix: str = "langchain",
) -> ConsensusResult:
    return _run_framework_consensus(
        outputs,
        normalizer=from_langchain_output,
        default_prefix=prefix,
        mode=mode,
        unstable_threshold=unstable_threshold,
    )


def run_langgraph_consensus(
    outputs: Iterable[Any],
    *,
    mode: ConsensusMode = "quantum_ready",
    unstable_threshold: float = 0.45,
    prefix: str = "langgraph",
) -> ConsensusResult:
    return _run_framework_consensus(
        outputs,
        normalizer=from_langgraph_node_output,
        default_prefix=prefix,
        mode=mode,
        unstable_threshold=unstable_threshold,
    )


def run_crewai_consensus(
    outputs: Iterable[Any],
    *,
    mode: ConsensusMode = "quantum_ready",
    unstable_threshold: float = 0.45,
    prefix: str = "crewai",
) -> ConsensusResult:
    return _run_framework_consensus(
        outputs,
        normalizer=from_crewai_result,
        default_prefix=prefix,
        mode=mode,
        unstable_threshold=unstable_threshold,
    )


def run_autogen_consensus(
    outputs: Iterable[Any],
    *,
    mode: ConsensusMode = "quantum_ready",
    unstable_threshold: float = 0.45,
    prefix: str = "autogen",
) -> ConsensusResult:
    return _run_framework_consensus(
        outputs,
        normalizer=from_autogen_message,
        default_prefix=prefix,
        mode=mode,
        unstable_threshold=unstable_threshold,
    )


def run_openclaw_consensus(
    outputs: Iterable[Any],
    *,
    mode: ConsensusMode = "quantum_ready",
    unstable_threshold: float = 0.45,
    prefix: str = "openclaw",
) -> ConsensusResult:
    return _run_framework_consensus(
        outputs,
        normalizer=from_openclaw_artifact,
        default_prefix=prefix,
        mode=mode,
        unstable_threshold=unstable_threshold,
    )


def from_langchain_output(item: Any, *, default_id: str = "langchain-1") -> AgentOutput:
    return normalize_to_agent_output(
        item,
        default_id=default_id,
        content_fields=(
            "content",
            "text",
            "output",
            "response",
            "answer",
            "generation",
        ),
        source_tag="langchain",
    )


def from_langgraph_node_output(
    item: Any,
    *,
    default_id: str = "langgraph-1",
) -> AgentOutput:
    return normalize_to_agent_output(
        item,
        default_id=default_id,
        content_fields=("content", "text", "output", "response", "answer", "result"),
        source_tag="langgraph",
    )


def from_crewai_result(item: Any, *, default_id: str = "crewai-1") -> AgentOutput:
    return normalize_to_agent_output(
        item,
        default_id=default_id,
        content_fields=(
            "content",
            "text",
            "output",
            "response",
            "answer",
            "final_answer",
            "result",
            "raw",
        ),
        source_tag="crewai",
    )


def from_autogen_message(item: Any, *, default_id: str = "autogen-1") -> AgentOutput:
    return normalize_to_agent_output(
        item,
        default_id=default_id,
        content_fields=("content", "text", "message", "response", "answer", "summary"),
        source_tag="autogen",
    )


def from_openclaw_artifact(item: Any, *, default_id: str = "openclaw-1") -> AgentOutput:
    return normalize_to_agent_output(
        item,
        default_id=default_id,
        content_fields=("content", "artifact", "payload", "output", "response", "answer"),
        source_tag="openclaw",
    )


def _run_framework_consensus(
    outputs: Iterable[Any],
    *,
    normalizer: Callable[..., AgentOutput],
    default_prefix: str,
    mode: ConsensusMode,
    unstable_threshold: float,
) -> ConsensusResult:
    normalized = [
        normalizer(item, default_id=f"{default_prefix}-{index}")
        for index, item in enumerate(outputs, start=1)
    ]
    return resolve_consensus(normalized, mode=mode, unstable_threshold=unstable_threshold)


def _first_value(item: Any, field_names: Sequence[str]) -> Any | None:
    for field_name in field_names:
        value = _get_value(item, field_name)
        if value is None:
            continue

        if _coerce_content(value):
            return value

    return None


def _get_value(item: Any, field_name: str) -> Any | None:
    if isinstance(item, Mapping):
        return item.get(field_name)

    return getattr(item, field_name, None)


def _coerce_identifier(value: Any) -> str:
    if value is None:
        return ""

    return _coerce_content(value).strip()


def _coerce_content(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, Mapping):
        try:
            return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        except TypeError:
            return str(value).strip()

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        fragments: list[str] = []
        for item in value:
            if isinstance(item, str):
                fragments.append(item)
                continue

            if isinstance(item, Mapping) and "text" in item:
                fragments.append(str(item["text"]))
                continue

            if hasattr(item, "content"):
                fragments.append(_coerce_content(item.content))
                continue

            fragments.append(str(item))

        return " ".join(fragment for fragment in fragments if fragment).strip()

    return str(value).strip()


def _coerce_confidence(value: Any) -> float | None:
    if value is None:
        return None

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    return max(0.0, min(1.0, numeric))


def _coerce_sources(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []

    if isinstance(value, Mapping):
        try:
            text = json.dumps(value, sort_keys=True, ensure_ascii=False)
        except TypeError:
            text = str(value).strip()
        return [text] if text else []

    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        sources = [str(item).strip() for item in value]
        return [source for source in sources if source]

    text = str(value).strip()
    return [text] if text else []


def _coerce_stats(value: Any) -> dict[str, Any]:
    if value is None:
        return {}

    if isinstance(value, Mapping):
        return dict(value)

    return {"value": value}