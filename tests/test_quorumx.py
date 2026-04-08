from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

from quorumx import (
    OpenAIBackend,
    QuorumX,
    QuorumXBackend,
    QuorumXBackendResult,
    QuorumXConfig,
    QuorumXResult,
    QuorumXUsage,
    quorum_x,
)


class BarrierBackend(QuorumXBackend):
    def __init__(self, parties: int) -> None:
        self.barrier = threading.Barrier(parties)
        self.thread_ids: list[int] = []
        self.messages: list[list[dict[str, Any]]] = []

    def call_llm(self, messages: list[dict[str, Any]], config: QuorumXConfig) -> str:
        self.thread_ids.append(threading.get_ident())
        self.messages.append([dict(message) for message in messages])
        self.barrier.wait(timeout=5)
        return "Parallel answer"


class RecordingBackend(QuorumXBackend):
    def __init__(self) -> None:
        self.messages: list[list[dict[str, Any]]] = []

    def call_llm(self, messages: list[dict[str, Any]], config: QuorumXConfig) -> str:
        self.messages.append([dict(message) for message in messages])
        return "Recorded answer"


class PartialFailureBackend(QuorumXBackend):
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.failures: list[str] = []
        self.lock = threading.Lock()

    def call_llm(
        self,
        messages: list[dict[str, Any]],
        config: QuorumXConfig,
    ) -> QuorumXBackendResult:
        persona_prompt = str(messages[0]["content"])
        with self.lock:
            self.calls.append(persona_prompt)

        if "Probe assumptions" in persona_prompt:
            with self.lock:
                self.failures.append("skeptic")
            raise RuntimeError("rate limit")

        return QuorumXBackendResult(
            text="Stable answer",
            usage=QuorumXUsage(prompt_tokens=17, completion_tokens=23),
        )


def test_quorumx_mock_run_returns_structured_result() -> None:
    engine = QuorumX(
        QuorumXConfig(
            n_agents=3,
            max_rounds=2,
            stability_threshold=0.65,
            consensus_mode="quantum_ready",
            mock_mode=True,
        )
    )

    result = engine.run(
        "Should we prioritize retention or acquisition for a SaaS with $8k MRR and 12% churn?"
    )

    assert isinstance(result, QuorumXResult)
    assert result.answer
    assert result.rounds_used >= 1
    assert result.total_tokens > 0
    assert result.prompt_tokens > 0
    assert result.completion_tokens > 0
    assert result.total_tokens == result.prompt_tokens + result.completion_tokens
    assert len(result.tokens_per_round) == result.rounds_used
    assert len(result.benchmark) == 3
    assert result.consensus_mode == "quantum_ready"


def test_quorumx_runs_personas_concurrently_per_round() -> None:
    backend = BarrierBackend(parties=3)
    engine = QuorumX(
        QuorumXConfig(
            n_agents=3,
            max_rounds=1,
            stability_threshold=0.7,
            consensus_mode="quantum_ready",
            mock_mode=True,
        ),
        backend=backend,
    )

    result = engine.run(
        "Draft a short launch email for a product update.",
        messages=[
            {"role": "system", "content": "You are careful and direct."},
            {"role": "user", "content": "Keep it short."},
            {"role": "assistant", "content": "Here is a draft."},
            {"role": "user", "content": "Now revise it."},
        ],
        system_instructions="Follow the company style guide.",
    )

    assert result.answer == "Parallel answer"
    assert result.unstable is False
    assert len(set(backend.thread_ids)) == 3
    assert len(backend.messages) == 3
    for message_batch in backend.messages:
        assert message_batch[0]["role"] == "system"
        assert message_batch[0]["content"] == "Follow the company style guide."
        assert message_batch[1]["role"] == "system"
        assert message_batch[2]["role"] == "system"
        assert message_batch[3]["role"] == "system"
        assert message_batch[4]["role"] == "user"
        assert message_batch[5]["role"] == "assistant"
        assert message_batch[6]["role"] == "user"


def test_quorumx_preserves_full_message_history_in_backend() -> None:
    backend = RecordingBackend()
    engine = QuorumX(
        QuorumXConfig(
            n_agents=2,
            max_rounds=1,
            stability_threshold=0.6,
            consensus_mode="quantum_ready",
            mock_mode=True,
        ),
        backend=backend,
    )

    original_messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Summarize this RFC in one sentence."},
        {"role": "assistant", "content": "Here is a draft."},
        {"role": "user", "content": "Revise it for concision."},
    ]

    engine.run(
        "Summarize this RFC in one sentence.",
        messages=original_messages,
        system_instructions="Follow the company style guide.",
    )

    assert len(backend.messages) == 2
    for message_batch in backend.messages:
        assert message_batch[0] == {
            "role": "system",
            "content": "Follow the company style guide.",
        }
        assert message_batch[1]["role"] == "system"
        assert message_batch[2]["role"] == "system"
        assert message_batch[3:] == original_messages


def test_quorumx_skips_failed_agents_but_keeps_successful_candidates() -> None:
    backend = PartialFailureBackend()
    engine = QuorumX(
        QuorumXConfig(
            n_agents=3,
            max_rounds=1,
            stability_threshold=0.7,
            consensus_mode="quantum_ready",
            mock_mode=True,
        ),
        backend=backend,
    )

    result = engine.run("Draft a short launch email for a product update.")

    assert result.answer == "Stable answer"
    assert result.unstable is False
    assert result.prompt_tokens == 34
    assert result.completion_tokens == 46
    assert result.total_tokens == 80
    assert result.tokens_per_round == [80]
    assert len(backend.calls) == 3
    assert backend.failures == ["skeptic"]
    assert any(benchmark.token_count == 0 for benchmark in result.benchmark)


def test_quorumx_decorator_replaces_original_return_value() -> None:
    called = False

    @quorum_x(
        QuorumXConfig(
            n_agents=2,
            max_rounds=1,
            stability_threshold=0.6,
            consensus_mode="quantum_ready",
            mock_mode=True,
        )
    )
    def legacy_agent(task: str) -> str:
        nonlocal called
        called = True
        return "ignored"

    result = legacy_agent("Draft a short launch email for a product update.")

    assert isinstance(result, QuorumXResult)
    assert called is False
    assert result.unstable in {True, False}


def test_quorumx_simple_majority_marks_unstable_when_candidates_disagree() -> None:
    engine = QuorumX(
        QuorumXConfig(
            n_agents=3,
            max_rounds=1,
            stability_threshold=0.95,
            consensus_mode="simple_majority",
            mock_mode=True,
        )
    )

    result = engine.run("Draft a short launch email for a product update.")

    assert result.unstable is True
    assert result.answer == "NO CONSENSUS"


def test_quorumx_real_mode_selects_default_openai_backend() -> None:
    engine = QuorumX(
        QuorumXConfig(
            n_agents=2,
            max_rounds=1,
            stability_threshold=0.6,
            consensus_mode="quantum_ready",
            mock_mode=False,
        )
    )

    assert isinstance(engine.backend, OpenAIBackend)


def test_openai_backend_uses_response_usage(monkeypatch) -> None:
    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Live answer"))],
        usage=SimpleNamespace(prompt_tokens=31, completion_tokens=19, total_tokens=50),
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: fake_response)
        )
    )

    monkeypatch.setattr(OpenAIBackend, "_get_client", lambda self: fake_client)

    backend = OpenAIBackend(model_name="gpt-4o-mini", temperature=0.2)
    result = backend.generate(
        messages=[{"role": "user", "content": "Hello"}],
        config=QuorumXConfig(),
    )

    assert result.text == "Live answer"
    assert result.usage.prompt_tokens == 31
    assert result.usage.completion_tokens == 19
    assert result.usage.total_tokens == 50


def test_quorumx_langchain_backend_is_optional() -> None:
    engine = QuorumX(
        QuorumXConfig(
            n_agents=2,
            max_rounds=1,
            stability_threshold=0.6,
            consensus_mode="quantum_ready",
            backend="langchain",
        )
    )

    from quorumx import LangChainOpenAIBackend

    assert isinstance(engine.backend, LangChainOpenAIBackend)