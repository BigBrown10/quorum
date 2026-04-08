from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .models import QuorumXConfig
from .personas import PersonaSpec


class QuorumXBackend(Protocol):
    def generate(
        self,
        *,
        task: str,
        persona: PersonaSpec,
        round_index: int,
        disagreement_summary: str,
        config: QuorumXConfig,
    ) -> str:
        raise NotImplementedError


@dataclass(slots=True)
class MockQuorumXBackend:
    def generate(
        self,
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
            response += (
                " After reviewing the disagreement summary, I still favor this answer."
            )

        return response.strip()


@dataclass(slots=True)
class LangChainOpenAIBackend:
    model_name: str
    temperature: float
    _client: Any = field(default=None, init=False, repr=False, compare=False)

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as exc:  # pragma: no cover - optional dependency path
                raise ImportError(
                    "langchain-openai is required for real QuorumX mode"
                ) from exc

            self._client = ChatOpenAI(model=self.model_name, temperature=self.temperature)

        return self._client

    def generate(
        self,
        *,
        task: str,
        persona: PersonaSpec,
        round_index: int,
        disagreement_summary: str,
        config: QuorumXConfig,
    ) -> str:
        client = self._get_client()
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise ImportError(
                "langchain-core is required for real QuorumX mode"
            ) from exc

        prompt_parts = [
            f"Task: {task}",
            f"Round: {round_index + 1}/{config.max_rounds}",
        ]
        if disagreement_summary:
            prompt_parts.append(f"Disagreement summary: {disagreement_summary}")

        response = client.invoke(
            [
                SystemMessage(content=persona.system_prompt),
                HumanMessage(content="\n\n".join(prompt_parts)),
            ]
        )
        return _message_content_to_text(response.content)


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