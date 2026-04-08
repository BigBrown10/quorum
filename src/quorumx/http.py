from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .engine import QuorumX
from .models import QuorumXConfig, QuorumXResult
from .telemetry import TelemetryHook, emit_telemetry

LOGGER = logging.getLogger(__name__)

CONFIG_FIELD_NAMES = {
    "n_agents",
    "max_rounds",
    "stability_threshold",
    "token_cap_per_agent_round",
    "consensus_mode",
    "system_instructions",
    "roles",
    "model",
    "quorum_model",
    "temperature",
    "backend",
    "api_key",
    "base_url",
    "request_timeout_seconds",
    "mock_mode",
}


def resolve_quorumx_payload(
    payload: dict[str, Any],
    *,
    telemetry: TelemetryHook | None = None,
) -> dict[str, Any]:
    messages = _messages_from_payload(payload)
    task = _extract_task(payload, messages)
    config = _config_from_payload(payload)
    result = QuorumX(config).run(
        task,
        messages=messages,
        system_instructions=payload.get("system_instructions"),
    )
    response = quorumx_result_to_payload(result)
    emit_telemetry(
        telemetry,
        "quorumx.resolve",
        {
            "task": task,
            "consensus_mode": response["consensus_mode"],
            "agreement_score": response["agreement_score"],
            "unstable": response["unstable"],
            "rounds_used": response["rounds_used"],
            "prompt_tokens": response["prompt_tokens"],
            "completion_tokens": response["completion_tokens"],
            "total_tokens": response["total_tokens"],
        },
    )
    return response


def chat_completions_payload(
    payload: dict[str, Any],
    *,
    telemetry: TelemetryHook | None = None,
) -> dict[str, Any]:
    messages = _messages_from_payload(payload)
    if not messages:
        raise ValueError("'messages' must be a non-empty list")

    task = _primary_task_from_messages(messages)
    config = _config_from_payload(payload)
    result = QuorumX(config).run(
        task,
        messages=messages,
        system_instructions=payload.get("system_instructions"),
    )
    created = int(time.time())
    model_name = config.quorum_model or config.model
    prompt_tokens = result.prompt_tokens
    completion_tokens = result.completion_tokens
    total_tokens = result.total_tokens

    response = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": str(result.answer)},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        "quorumx": quorumx_result_to_payload(result),
    }
    emit_telemetry(
        telemetry,
        "quorumx.chat_completions",
        {
            "model": model_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "agreement_score": result.agreement_score,
            "unstable": result.unstable,
        },
    )

    return response


def chat_completions_stream_response(
    payload: dict[str, Any],
    *,
    telemetry: TelemetryHook | None = None,
) -> tuple[dict[str, Any], list[str]]:
    response = chat_completions_payload(payload, telemetry=telemetry)
    return response, _chat_completions_stream_events(response)


def quorumx_result_to_payload(result: QuorumXResult) -> dict[str, Any]:
    return {
        "answer": result.answer,
        "agreement_score": result.agreement_score,
        "unstable": result.unstable,
        "rounds_used": result.rounds_used,
        "total_tokens": result.total_tokens,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "tokens_per_round": result.tokens_per_round,
        "benchmark": [asdict(item) for item in result.benchmark],
        "disagreement_edges_final": [
            asdict(edge) for edge in result.disagreement_edges_final
        ],
        "selected_agent_ids": result.selected_agent_ids,
        "consensus_mode": result.consensus_mode,
        "rationale": result.rationale,
    }


class QuorumXHTTPRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, status: HTTPStatus, events: list[str]) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        for event in events:
            self.wfile.write(event.encode("utf-8"))
            self.wfile.flush()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/v1/quorumx", "/v1/chat/completions"}:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return

        telemetry = getattr(self.server, "quorumx_telemetry", None)
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)

        try:
            payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
            stream_requested = bool(payload.get("stream"))
            if self.path == "/v1/quorumx":
                response = resolve_quorumx_payload(payload, telemetry=telemetry)
                LOGGER.info(
                    "quorumx request path=%s agreement_score=%.3f "
                    "unstable=%s rounds=%s tokens=%s",
                    self.path,
                    response.get("agreement_score", 0.0),
                    response.get("unstable", False),
                    response.get("rounds_used", 0),
                    response.get("total_tokens", 0),
                )
                self._send_json(HTTPStatus.OK, response)
                return
            else:
                if stream_requested:
                    response, events = chat_completions_stream_response(
                        payload,
                        telemetry=telemetry,
                    )
                    LOGGER.info(
                        "quorumx request path=%s agreement_score=%.3f "
                        "unstable=%s rounds=%s tokens=%s",
                        self.path,
                        response.get("quorumx", {}).get("agreement_score", 0.0),
                        response.get("quorumx", {}).get("unstable", False),
                        response.get("quorumx", {}).get("rounds_used", 0),
                        response.get("quorumx", {}).get("total_tokens", 0),
                    )
                    self._send_sse(HTTPStatus.OK, events)
                    return

                response = chat_completions_payload(payload, telemetry=telemetry)
                LOGGER.info(
                    "quorumx request path=%s agreement_score=%.3f "
                    "unstable=%s rounds=%s tokens=%s",
                    self.path,
                    response.get("quorumx", {}).get("agreement_score", 0.0),
                    response.get("quorumx", {}).get("unstable", False),
                    response.get("quorumx", {}).get("rounds_used", 0),
                    response.get("quorumx", {}).get("total_tokens", 0),
                )
                self._send_json(HTTPStatus.OK, response)
                return
        except (json.JSONDecodeError, UnicodeDecodeError):
            emit_telemetry(
                telemetry,
                "quorumx.http.invalid_json",
                {"path": self.path},
            )
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_json"})
            return
        except (KeyError, TypeError, ValueError) as exc:
            emit_telemetry(
                telemetry,
                "quorumx.http.invalid_request",
                {"path": self.path, "detail": str(exc)},
            )
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "invalid_request", "detail": str(exc)},
            )
            return
        except Exception as exc:  # pragma: no cover - defensive server boundary
            LOGGER.exception("Unexpected QuorumX gateway error")
            emit_telemetry(
                telemetry,
                "quorumx.http.error",
                {"path": self.path, "detail": str(exc)},
            )
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "internal_error", "detail": str(exc)},
            )
            return

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def create_server(
    host: str = "127.0.0.1",
    port: int = 8010,
    *,
    telemetry: TelemetryHook | None = None,
) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), QuorumXHTTPRequestHandler)
    server.quorumx_telemetry = telemetry  # type: ignore[attr-defined]
    return server


def main() -> None:
    server = create_server()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _config_from_payload(payload: dict[str, Any]) -> QuorumXConfig:
    config_payload: dict[str, Any] = {}

    nested_config = payload.get("config")
    if nested_config is not None:
        if not isinstance(nested_config, dict):
            raise ValueError("'config' must be an object when provided")
        config_payload.update(nested_config)

    for field_name in CONFIG_FIELD_NAMES:
        if field_name in payload and field_name not in config_payload:
            config_payload[field_name] = payload[field_name]

    return QuorumXConfig(**config_payload)


def _extract_task(payload: dict[str, Any], messages: list[dict[str, Any]] | None = None) -> str:
    task = payload.get("task")
    if isinstance(task, str) and task.strip():
        return task.strip()

    if messages:
        inferred_task = _primary_task_from_messages(messages)
        if inferred_task:
            return inferred_task

    raise ValueError("'task' must be a non-empty string")


def _messages_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages = payload.get("messages")
    if messages is None:
        return []

    if not isinstance(messages, list):
        raise ValueError("'messages' must be a non-empty list")

    normalized_messages: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        normalized_messages.append(dict(message))

    return normalized_messages


def _primary_task_from_messages(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if str(message.get("role", "")).lower() != "user":
            continue
        content = _message_to_text(message.get("content"))
        if content:
            return content

    for message in reversed(messages):
        content = _message_to_text(message.get("content"))
        if content:
            return content

    return ""


def _message_to_text(content: Any) -> str:
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


def _chat_completions_stream_events(response: dict[str, Any]) -> list[str]:
    message = response.get("choices", [{}])[0].get("message", {})
    content = str(message.get("content", ""))
    base_chunk = {
        "id": response.get("id"),
        "object": "chat.completion.chunk",
        "created": response.get("created"),
        "model": response.get("model"),
    }
    first_chunk = {
        **base_chunk,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": content},
                "finish_reason": None,
            }
        ],
    }
    final_chunk = {
        **base_chunk,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    return [
        f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n",
        f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n",
        "data: [DONE]\n\n",
    ]