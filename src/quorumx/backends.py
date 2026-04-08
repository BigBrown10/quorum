from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

from .models import QuorumXConfig
from .personas import PersonaSpec


class QuorumXBackend(ABC):
    @abstractmethod
    def call_llm(self, messages: list[dict[str, Any]], config: QuorumXConfig) -> str:
        raise NotImplementedError

    def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        config: QuorumXConfig,
    ) -> str:
        return self.call_llm(messages, config)


@dataclass(slots=True)
class OpenAIBackend(QuorumXBackend):
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None
    timeout_seconds: float = 30.0
    _client: Any = field(default=None, init=False, repr=False, compare=False)

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - optional dependency path
                raise ImportError("openai is required for the default QuorumX backend") from exc

            client_kwargs: dict[str, Any] = {"timeout": self.timeout_seconds}
            if self.api_key is not None:
                client_kwargs["api_key"] = self.api_key
            if self.base_url is not None:
                client_kwargs["base_url"] = self.base_url

            self._client = OpenAI(**client_kwargs)

        return self._client

    def call_llm(self, messages: list[dict[str, Any]], config: QuorumXConfig) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=list(messages),
            temperature=self.temperature,
        )
        message = response.choices[0].message if response.choices else None
        content = getattr(message, "content", None)
        return _message_content_to_text(content)


@dataclass(slots=True)
class MockQuorumXBackend(QuorumXBackend):
    def call_llm(self, messages: list[dict[str, Any]], config: QuorumXConfig) -> str:
        persona_name = _extract_message_marker(messages, "Persona") or "mock"
        task = _extract_message_marker(messages, "Task") or _last_user_message(messages) or "the task"
        round_value = _extract_message_marker(messages, "Round") or "1/1"
        disagreement_summary = _extract_message_marker(messages, "Disagreement summary") or ""
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
                    "langchain-openai is required when backend='langchain'"
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

    def call_llm(self, messages: list[dict[str, Any]], config: QuorumXConfig) -> str:
        client = self._get_client()
        try:
            from langchain_core.messages import (
                AIMessage,
                ChatMessage,
                HumanMessage,
                SystemMessage,
                ToolMessage,
            )
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise ImportError(
                "langchain-core is required when backend='langchain'"
            ) from exc

        def to_message(message: dict[str, Any]) -> Any:
            role = str(message.get("role", "user")).lower()
            content = _message_content_to_text(message.get("content"))
            if role == "system":
                return SystemMessage(content=content)
            if role == "assistant":
                return AIMessage(content=content)
            if role == "tool":
                tool_call_id = str(message.get("tool_call_id") or message.get("id") or "tool")
                return ToolMessage(content=content, tool_call_id=tool_call_id)
            if role == "developer":
                return ChatMessage(role="developer", content=content)
            return HumanMessage(content=content)

        response = client.invoke([to_message(message) for message in messages])
        return _message_content_to_text(response.content)


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


def _extract_message_marker(messages: Sequence[dict[str, Any]], label: str) -> str | None:
    prefix = f"{label}:"
    for message in messages:
        content = _message_content_to_text(message.get("content"))
        for line in content.splitlines():
            if line.startswith(prefix):
                return line[len(prefix):].strip()
    return None


def _last_user_message(messages: Sequence[dict[str, Any]]) -> str | None:
    for message in reversed(messages):
        if str(message.get("role", "")).lower() != "user":
            continue
        content = _message_content_to_text(message.get("content"))
        if content:
            return content
    return None


def _parse_round_index(round_value: str) -> int:
    if "/" not in round_value:
        return 0

    candidate = round_value.split("/", 1)[0].strip()
    try:
        return max(0, int(candidate) - 1)
    except ValueError:
        return 0