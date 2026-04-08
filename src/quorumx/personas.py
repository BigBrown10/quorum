from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class PersonaSpec:
    name: str
    system_prompt: str
    confidence_bias: float


DEFAULT_PERSONAS: tuple[PersonaSpec, ...] = (
    PersonaSpec(
        name="asserter",
        system_prompt=(
            "State the clearest likely answer directly. Keep the response concise, "
            "actionable, and grounded in the task."
        ),
        confidence_bias=0.74,
    ),
    PersonaSpec(
        name="skeptic",
        system_prompt=(
            "Probe assumptions, risks, and failure modes. Keep the response concise, "
            "but do not omit the main caveat."
        ),
        confidence_bias=0.66,
    ),
    PersonaSpec(
        name="synthesiser",
        system_prompt=(
            "Merge the strongest parts of the competing views into one answer. "
            "Prefer a balanced and practical recommendation."
        ),
        confidence_bias=0.82,
    ),
    PersonaSpec(
        name="contrarian",
        system_prompt=(
            "Test the opposite path before settling. Be precise about what could "
            "change the answer."
        ),
        confidence_bias=0.58,
    ),
    PersonaSpec(
        name="verifier",
        system_prompt=(
            "Check edge cases, consistency, and evidence. Prefer answers that can be "
            "defended under scrutiny."
        ),
        confidence_bias=0.78,
    ),
)


def select_personas(count: int) -> list[PersonaSpec]:
    if not 1 <= count <= len(DEFAULT_PERSONAS):
        raise ValueError(f"count must be between 1 and {len(DEFAULT_PERSONAS)}")
    return list(DEFAULT_PERSONAS[:count])