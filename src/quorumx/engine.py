from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Sequence

from quorum_core import AgentOutput, resolve_consensus

from .backends import (
    LangChainOpenAIBackend,
    MockQuorumXBackend,
    OpenAIBackend,
    QuorumXBackend,
)
from .models import AgentBenchmark, QuorumXConfig, QuorumXResult
from .personas import PersonaSpec, select_personas

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _BenchmarkAccumulator:
    stance: str
    rounds_used: int = 0
    token_count: int = 0
    confidence_total: float = 0.0
    last_answer: str = ""

    def record(self, answer: str, tokens: int, confidence: float) -> None:
        self.rounds_used += 1
        self.token_count += tokens
        self.confidence_total += confidence
        self.last_answer = answer

    def snapshot(self, agent_id: str) -> AgentBenchmark:
        if self.rounds_used:
            confidence = self.confidence_total / self.rounds_used
        else:
            confidence = 0.0

        return AgentBenchmark(
            agent_id=agent_id,
            stance=self.stance,
            rounds_used=self.rounds_used,
            token_count=self.token_count,
            confidence=confidence,
            answer_preview=_truncate_text(self.last_answer, 120),
        )


class QuorumX:
    def __init__(
        self,
        config: QuorumXConfig,
        backend: QuorumXBackend | None = None,
        personas: Sequence[PersonaSpec] | None = None,
    ) -> None:
        self.config = config
        self.backend = backend or self._build_default_backend(config)
        self.personas = self._build_personas(config, personas)

    def run(
        self,
        task: str,
        *,
        messages: Sequence[dict[str, Any]] | None = None,
        system_instructions: str | None = None,
    ) -> QuorumXResult:
        cleaned_task = " ".join(task.strip().split())
        if not cleaned_task:
            raise ValueError("task must not be empty")

        effective_system_instructions = system_instructions or self.config.system_instructions
        benchmark_state = {
            persona.name: _BenchmarkAccumulator(stance=persona.name)
            for persona in self.personas
        }
        tokens_per_round: list[int] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        last_result = None
        disagreement_summary = ""
        conversation_messages = _build_conversation_messages(cleaned_task, messages=messages)

        for round_index in range(self.config.max_rounds):
            candidates: list[AgentOutput] = []
            round_tokens = 0

            with ThreadPoolExecutor(max_workers=max(1, len(self.personas))) as executor:
                round_inputs: list[tuple[PersonaSpec, list[dict[str, Any]]]] = []
                futures = []

                for persona in self.personas:
                    prompt_messages = _build_round_messages(
                        conversation_messages,
                        task=cleaned_task,
                        persona=persona,
                        round_index=round_index,
                        disagreement_summary=disagreement_summary,
                        config=self.config,
                        system_instructions=effective_system_instructions,
                    )
                    round_inputs.append((persona, prompt_messages))
                    futures.append(
                        executor.submit(
                            self.backend.generate,
                            messages=prompt_messages,
                            config=self.config,
                        )
                    )

                for (persona, prompt_messages), future in zip(
                    round_inputs,
                    futures,
                    strict=True,
                ):
                    try:
                        backend_result = future.result()
                    except Exception as exc:
                        LOGGER.warning(
                            "QuorumX agent failure persona=%s round=%s error_type=%s error=%s",
                            persona.name,
                            round_index + 1,
                            type(exc).__name__,
                            exc,
                            exc_info=True,
                        )
                        continue

                    answer = _truncate_to_token_cap(
                        backend_result.text,
                        self.config.token_cap_per_agent_round,
                    )
                    prompt_tokens = backend_result.usage.prompt_tokens
                    completion_tokens = backend_result.usage.completion_tokens
                    total_tokens = backend_result.usage.total_tokens
                    round_tokens += total_tokens
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens

                    confidence = _estimate_confidence(
                        persona,
                        round_index,
                        last_result.agreement_score if last_result is not None else None,
                    )
                    candidate = AgentOutput(
                        id=f"{persona.name}_{round_index + 1}",
                        content=answer,
                        confidence=confidence,
                        sources=[f"quorumx:{persona.name}"],
                        stats={
                            "persona": persona.name,
                            "round": round_index + 1,
                            "message_count": len(prompt_messages),
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "response_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                        },
                    )
                    candidates.append(candidate)
                    benchmark_state[persona.name].record(answer, total_tokens, confidence)

            if not candidates:
                return self._backend_error_result(
                    task=cleaned_task,
                    error=RuntimeError(f"all agents failed in round {round_index + 1}"),
                    tokens_per_round=tokens_per_round,
                    benchmark_state=benchmark_state,
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                )

            last_result = resolve_consensus(
                candidates,
                mode=self.config.consensus_mode,
                unstable_threshold=self.config.stability_threshold,
            )
            tokens_per_round.append(round_tokens)

            if not last_result.unstable or round_index == self.config.max_rounds - 1:
                break

            disagreement_summary = _build_disagreement_summary(last_result, candidates)

        assert last_result is not None

        benchmark = [
            benchmark_state[persona.name].snapshot(f"{persona.name}_agent")
            for persona in self.personas
        ]

        return QuorumXResult(
            answer=last_result.consensus_answer,
            agreement_score=last_result.agreement_score,
            unstable=last_result.unstable,
            rounds_used=len(tokens_per_round),
            total_tokens=total_prompt_tokens + total_completion_tokens,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            tokens_per_round=tokens_per_round,
            benchmark=benchmark,
            disagreement_edges_final=last_result.disagreement_edges,
            selected_agent_ids=last_result.selected_agent_ids,
            consensus_mode=self.config.consensus_mode,
            rationale=last_result.rationale,
        )

    @staticmethod
    def _build_default_backend(config: QuorumXConfig) -> QuorumXBackend:
        if config.backend == "mock" or config.mock_mode:
            return MockQuorumXBackend()
        if config.backend == "langchain":
            return LangChainOpenAIBackend(
                model_name=config.quorum_model or config.model,
                temperature=config.temperature,
                api_key=config.api_key,
                base_url=config.base_url,
                timeout_seconds=config.request_timeout_seconds,
            )
        return OpenAIBackend(
            model_name=config.quorum_model or config.model,
            temperature=config.temperature,
            api_key=config.api_key,
            base_url=config.base_url,
            timeout_seconds=config.request_timeout_seconds,
        )

    @staticmethod
    def _build_personas(
        config: QuorumXConfig,
        personas: Sequence[PersonaSpec] | None,
    ) -> list[PersonaSpec]:
        if personas is not None:
            resolved_personas = list(personas)
            if not resolved_personas:
                raise ValueError("personas must not be empty when provided")
            if len(resolved_personas) > 5:
                raise ValueError("personas must contain at most 5 entries")
            if config.n_agents != len(resolved_personas):
                config.n_agents = len(resolved_personas)
            return resolved_personas

        if config.roles is not None:
            return select_personas(config.n_agents, role_names=config.roles)

        return select_personas(config.n_agents)

    def _backend_error_result(
        self,
        *,
        task: str,
        error: Exception,
        tokens_per_round: list[int],
        benchmark_state: dict[str, _BenchmarkAccumulator],
        prompt_tokens: int,
        completion_tokens: int,
    ) -> QuorumXResult:
        snapshots = [
            accumulator.snapshot(f"{persona.name}_agent")
            for persona, accumulator in zip(self.personas, benchmark_state.values(), strict=False)
        ]
        return QuorumXResult(
            answer="NO CONSENSUS",
            agreement_score=0.0,
            unstable=True,
            rounds_used=max(1, len(tokens_per_round) + 1),
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tokens_per_round=list(tokens_per_round),
            benchmark=snapshots,
            disagreement_edges_final=[],
            selected_agent_ids=[],
            consensus_mode=self.config.consensus_mode,
            rationale=(
                f"NO CONSENSUS: backend error while processing task "
                f"{_truncate_text(task, 80)!r}: {error}"
            ),
        )


def quorum_x(
    config: QuorumXConfig,
    backend: QuorumXBackend | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., QuorumXResult]]:
    engine = QuorumX(config, backend=backend)

    def decorator(function: Callable[..., Any]) -> Callable[..., QuorumXResult]:
        @wraps(function)
        def wrapper(task: str, *args: Any, **kwargs: Any) -> QuorumXResult:
            return engine.run(task)

        return wrapper

    return decorator


def _build_conversation_messages(
    task: str,
    *,
    messages: Sequence[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if messages is None:
        return [{"role": "user", "content": task}]

    normalized_messages = [dict(message) for message in messages if isinstance(message, dict)]
    if normalized_messages:
        return normalized_messages

    return [{"role": "user", "content": task}]


def _build_round_messages(
    conversation_messages: Sequence[dict[str, Any]],
    *,
    task: str,
    persona: PersonaSpec,
    round_index: int,
    disagreement_summary: str,
    config: QuorumXConfig,
    system_instructions: str | None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_instructions:
        messages.append({"role": "system", "content": system_instructions})
    messages.append({"role": "system", "content": persona.system_prompt})
    messages.append(
        {
            "role": "system",
            "content": _build_round_instruction(
                task=task,
                persona=persona,
                round_index=round_index,
                disagreement_summary=disagreement_summary,
                config=config,
            ),
        }
    )
    messages.extend(dict(message) for message in conversation_messages)
    return messages


def _build_round_instruction(
    *,
    task: str,
    persona: PersonaSpec,
    round_index: int,
    disagreement_summary: str,
    config: QuorumXConfig,
) -> str:
    prompt_lines = [
        f"Persona: {persona.name}",
        f"Round: {round_index + 1}/{config.max_rounds}",
        f"Task: {task}",
    ]
    if disagreement_summary:
        prompt_lines.append(f"Disagreement summary: {disagreement_summary}")
    prompt_lines.append(
        f"Return a concise answer within roughly {config.token_cap_per_agent_round} tokens."
    )
    return "\n".join(prompt_lines)


def _build_disagreement_summary(
    consensus_result: Any,
    candidates: list[AgentOutput],
) -> str:
    if not consensus_result.disagreement_edges:
        return "No material disagreements were detected in the prior round."

    candidate_lookup = {candidate.id: candidate for candidate in candidates}
    ranked_edges = sorted(
        consensus_result.disagreement_edges,
        key=lambda edge: edge.weight,
        reverse=True,
    )[:3]

    fragments: list[str] = []
    for edge in ranked_edges:
        left = candidate_lookup.get(edge.source_id)
        right = candidate_lookup.get(edge.target_id)
        if left is None or right is None:
            continue
        fragments.append(f"{left.id} vs {right.id} (weight {edge.weight:.2f})")

    if not fragments:
        return "The previous round showed disagreement, but no edges were retained."

    return "Strongest disagreements: " + "; ".join(fragments)


def _estimate_confidence(
    persona: PersonaSpec,
    round_index: int,
    previous_agreement_score: float | None,
) -> float:
    confidence = persona.confidence_bias + (0.03 * round_index)
    if previous_agreement_score is not None:
        confidence += (previous_agreement_score - 0.5) * 0.1
    return max(0.1, min(0.95, confidence))


def _truncate_to_token_cap(text: str, token_cap: int) -> str:
    words = text.split()
    if len(words) <= token_cap:
        return text.strip()
    return " ".join(words[:token_cap]).strip()


def _truncate_text(text: str, max_length: int) -> str:
    cleaned = text.strip()
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[: max_length - 3].rstrip() + "..."