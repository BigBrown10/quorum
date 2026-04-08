from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .models import QuorumXConfig
from .personas import PersonaSpec


class QuorumXBackend(ABC):
    @abstractmethod
    def call_llm(self, prompt: str, config: QuorumXConfig) -> str:
        raise NotImplementedError

    def generate(
        self,
        *,
        task: str,
        persona: PersonaSpec,
        round_index: int,
        disagreement_summary: str,
        config: QuorumXConfig,
    ) -> str:
        prompt = _build_prompt(task, persona, round_index, disagreement_summary, config)
        return self.call_llm(prompt, config)


@dataclass(slots=True)
class MockQuorumXBackend(QuorumXBackend):
    def call_llm(self, prompt: str, config: QuorumXConfig) -> str:
        persona_name = _extract_prompt_value(prompt, "Persona") or "mock"
        task = _extract_prompt_value(prompt, "Task") or prompt
        round_value = _extract_prompt_value(prompt, "Round") or "1/1"
        disagreement_summary = _extract_prompt_value(prompt, "Disagreement summary") or ""
        round_index = _parse_round_index(round_value)

        persona = PersonaSpec(
            name=persona_name,
            system_prompt="",
            confidence_bias=0.7,
        )
        return _compose_mock_response(
            task=task,
            persona=persona,
            round_index=round_index,
            disagreement_summary=disagreement_summary,
            config=config,
        )

    def generate(
        self,
        *,
        task: str,
        persona: PersonaSpec,
        round_index: int,
        disagreement_summary: str,
        config: QuorumXConfig,
    ) -> str:
        prompt = _build_prompt(task, persona, round_index, disagreement_summary, config)
        return self.call_llm(prompt, config)


@dataclass(slots=True)
class LangChainOpenAIBackend(QuorumXBackend):
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None
    timeout_seconds: float = 30.0
    _client: Any = field(default=None, init=False, repr=False, compare=False)

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as exc:  # pragma: no cover - optional dependency path
                raise ImportError(
                    "langchain-openai is required for real QuorumX mode"
                ) from exc

            client_kwargs: dict[str, Any] = {
                "model": self.model_name,
                "temperature": self.temperature,
                "timeout": self.timeout_seconds,
            }
            if self.api_key is not None:
                client_kwargs["api_key"] = self.api_key
            if self.base_url is not None:
                client_kwargs["base_url"] = self.base_url

            self._client = ChatOpenAI(**client_kwargs)

        return self._client

    def call_llm(self, prompt: str, config: QuorumXConfig) -> str:
        client = self._get_client()
        try:
            from langchain_core.messages import HumanMessage
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise ImportError(
                "langchain-core is required for real QuorumX mode"
            ) from exc

        response = client.invoke([HumanMessage(content=prompt)])
        return _message_content_to_text(response.content)

    def generate(
        self,
        *,
        task: str,
        persona: PersonaSpec,
        round_index: int,
        disagreement_summary: str,
        config: QuorumXConfig,
    ) -> str:
        prompt = _build_prompt(task, persona, round_index, disagreement_summary, config)
        prompt = f"System: {persona.system_prompt}\n\n{prompt}"
        return self.call_llm(prompt, config)


def _build_prompt(
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


def _compose_mock_response(
    *,
    task: str,
    persona: PersonaSpec,
    round_index: int,
    disagreement_summary: str,
    config: QuorumXConfig,
) -> str:
    topic = _task_topic(task)
    shared_answer = (
        f"Recommended answer: for {topic}, choose the simplest approach that "
        "satisfies the constraints."
    )
    round_prefix = "Initial view" if round_index == 0 else "Refined view"

    persona_suffixes = {
        "asserter": "Proceed directly and keep the implementation direct.",
        "skeptic": "Verify the assumptions and add guardrails before you proceed.",
        "synthesiser": "Balance speed with reliability and document the trade-offs.",
        "contrarian": "Consider the opposite path, but this remains the stronger option.",
        "verifier": "Check edge cases, failure modes, and consistency before shipping.",
    }

    response = f"{round_prefix}: {shared_answer} {persona_suffixes.get(persona.name, '')}"
    if disagreement_summary:
        response += " After reviewing the disagreement summary, I still favor this answer."

    return response.strip()


def _task_topic(task: str) -> str:
    cleaned = " ".join(task.strip().split())
    if not cleaned:
        return "the task"

    words = cleaned.split()
    return " ".join(words[:18]).rstrip(".,;:")


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        fragments: list[str] = []
        for item in content:
            if isinstance(item, str):
                fragments.append(item)
            elif isinstance(item, dict) and "text" in item:
                fragments.append(str(item["text"]))
            else:
                fragments.append(str(item))
        return " ".join(fragment for fragment in fragments if fragment).strip()

    return str(content).strip()


def _extract_prompt_value(prompt: str, label: str) -> str | None:
    prefix = f"{label}:"
    for line in prompt.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return None


def _parse_round_index(round_value: str) -> int:
    if "/" not in round_value:
        return 0

    candidate = round_value.split("/", 1)[0].strip()
    try:
        return max(0, int(candidate) - 1)
    except ValueError:
        return 0