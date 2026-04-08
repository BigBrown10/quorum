from __future__ import annotations

from collections.abc import Callable
from typing import Any

TelemetryHook = Callable[[str, dict[str, Any]], None]


def emit_telemetry(
    hook: TelemetryHook | None,
    event: str,
    payload: dict[str, Any],
) -> None:
    if hook is None:
        return

    hook(event, payload)