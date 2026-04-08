# QuorumX Configuration

QuorumX is designed to stay small at the surface and explicit at the edges. The config object controls debate shape, backend selection, and prompt scaffolding, while personas and telemetry stay opt-in and easy to override.

## QuorumXConfig

`QuorumXConfig` lives in `src/quorumx/models.py` and is the main runtime configuration object.

- `n_agents`: Number of debate personas to use. Valid range is 1 to 5.
- `max_rounds`: Maximum debate rounds to run. Valid range is 1 to 3.
- `stability_threshold`: Consensus instability threshold passed to Quorum Core.
- `token_cap_per_agent_round`: Soft answer length cap applied to each agent response.
- `consensus_mode`: Consensus strategy. Supported values are `simple_majority`, `weighted_majority`, `graph_min_cut`, and `quantum_ready`.
- `system_instructions`: Optional outermost system instruction string applied to every round.
- `roles`: Optional ordered list of default persona role names. When provided, `n_agents` is normalized to the list length.
- `model`: Primary debate model name.
- `quorum_model`: Optional cheaper or smaller synthesis model name. If omitted, QuorumX falls back to `model`.
- `temperature`: Sampling temperature for the real backend.
- `backend`: Explicit backend selector. Use `mock`, `openai`, or `langchain`.
- `api_key`: API key for the real backend.
- `base_url`: Optional API base URL for OpenAI-compatible providers.
- `request_timeout_seconds`: Timeout for the LLM client path.
- `mock_mode`: Shortcut for offline and CI runs. When true, QuorumX uses the mock backend unless `backend` is explicitly set.

Example:

```python
from quorumx import QuorumX, QuorumXConfig

config = QuorumXConfig(
    n_agents=3,
    max_rounds=2,
    stability_threshold=0.7,
    consensus_mode="quantum_ready",
    system_instructions="Be concise and concrete.",
    roles=["asserter", "skeptic", "verifier"],
    model="gpt-4o-mini",
    quorum_model="gpt-4o-mini",
)

result = QuorumX(config).run("Review this patch for correctness.")
```

## Personas

Default personas live in `src/quorumx/personas.py` as `DEFAULT_PERSONAS`. The built-in roles are:

- `asserter`: Direct, decisive, and implementation-first.
- `skeptic`: Focused on assumptions, risk, and failure modes.
- `synthesiser`: Balances the strongest parts of the competing views.
- `contrarian`: Tests the opposite path before settling.
- `verifier`: Checks edge cases, consistency, and evidence.

Use `select_personas(count)` when you want the first `count` default personas.
Use `select_personas(count, role_names=[...])` when you want a specific subset of the defaults in a fixed order.

If you need custom prompts instead of the built-in personas, pass a `personas=[PersonaSpec(...)]` list directly to `QuorumX(...)`:

```python
from quorumx import PersonaSpec, QuorumX, QuorumXConfig

config = QuorumXConfig(n_agents=2, mock_mode=True)
custom_personas = [
    PersonaSpec(name="planner", system_prompt="Optimize for scope and sequencing.", confidence_bias=0.74),
    PersonaSpec(name="reviewer", system_prompt="Focus on risks and regressions.", confidence_bias=0.68),
]

result = QuorumX(config, personas=custom_personas).run("Plan a safe API rollout.")
```

## Telemetry

The HTTP gateway and MCP server accept a telemetry hook shaped like `Callable[[str, dict[str, Any]], None]`.

Implement a hook that forwards events to logs, metrics, or tracing:

```python
from typing import Any

from quorumx.http import chat_completions_payload, resolve_quorumx_payload
from quorumx.mcp import QuorumXMCPServer, create_server


def telemetry_hook(event: str, payload: dict[str, Any]) -> None:
    print(event, payload)


resolve_quorumx_payload({
    "task": "Review this patch.",
    "config": {"mock_mode": True},
}, telemetry=telemetry_hook)

chat_completions_payload({
    "messages": [{"role": "user", "content": "Draft a short follow-up email."}],
    "config": {"mock_mode": True},
}, telemetry=telemetry_hook)

create_server(telemetry=telemetry_hook)
QuorumXMCPServer(telemetry=telemetry_hook)
```

Use `emit_telemetry(...)` inside your own code when you want to forward a QuorumX event without importing the gateway module directly.

Common event names include:

- `quorumx.resolve`
- `quorumx.chat_completions`
- `quorumx.mcp.tool_call`
- `quorumx.http.invalid_json`
- `quorumx.http.invalid_request`
- `quorumx.http.error`

A typical telemetry payload includes the task or model name, agreement score, instability flag, and token accounting fields.
