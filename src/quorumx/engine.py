from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable

from quorum_core import AgentOutput, resolve_consensus

from .backends import LangChainOpenAIBackend, MockQuorumXBackend, QuorumXBackend
from .models import AgentBenchmark, QuorumXConfig, QuorumXResult
from .personas import PersonaSpec, select_personas


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
    ) -> None:
        self.config = config
        self.backend = backend or self._build_default_backend(config)
        self.personas = select_personas(config.n_agents)

    def run(self, task: str) -> QuorumXResult:
        cleaned_task = " ".join(task.strip().split())
        if not cleaned_task:
            raise ValueError("task must not be empty")

        benchmark_state = {
            persona.name: _BenchmarkAccumulator(stance=persona.name)
            for persona in self.personas
        }
        tokens_per_round: list[int] = []
        last_result = None
        disagreement_summary = ""

        for round_index in range(self.config.max_rounds):
            candidates: list[AgentOutput] = []
            round_tokens = 0

            for _position, persona in enumerate(self.personas):
                prompt_text = _build_prompt(
                    cleaned_task,
                    persona,
                    disagreement_summary,
                    round_index,
                    self.config,
                )
                raw_response = self.backend.generate(
                    task=cleaned_task,
                    persona=persona,
                    round_index=round_index,
                    disagreement_summary=disagreement_summary,
                    config=self.config,
                )
                answer = _truncate_to_token_cap(raw_response, self.config.token_cap_per_agent_round)
                prompt_tokens = _token_count(prompt_text)
                response_tokens = _token_count(answer)
                total_tokens = prompt_tokens + response_tokens
                round_tokens += total_tokens

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
                        "prompt_tokens": prompt_tokens,
                        "response_tokens": response_tokens,
                    },
                )
                candidates.append(candidate)
                benchmark_state[persona.name].record(answer, total_tokens, confidence)

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
            total_tokens=sum(tokens_per_round),
            tokens_per_round=tokens_per_round,
            benchmark=benchmark,
            disagreement_edges_final=last_result.disagreement_edges,
            selected_agent_ids=last_result.selected_agent_ids,
            consensus_mode=self.config.consensus_mode,
            rationale=last_result.rationale,
        )

    @staticmethod
    def _build_default_backend(config: QuorumXConfig) -> QuorumXBackend:
        if config.mock_mode:
            return MockQuorumXBackend()
        return LangChainOpenAIBackend(
            model_name=config.model,
            temperature=config.temperature,
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


def _build_prompt(
    task: str,
    persona: PersonaSpec,
    disagreement_summary: str,
    round_index: int,
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
        fragments.append(
            f"{left.id} vs {right.id} (weight {edge.weight:.2f})"
        )

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


def _token_count(text: str) -> int:
    return max(1, len(text.split()))


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