from __future__ import annotations

import json
import logging
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .consensus import resolve_consensus
from .models import AgentOutput

LOGGER = logging.getLogger(__name__)


def _candidate_from_payload(raw_candidate: dict[str, Any]) -> AgentOutput:
    return AgentOutput(
        id=str(raw_candidate["id"]),
        content=raw_candidate["content"],
        confidence=raw_candidate.get("confidence"),
        sources=list(raw_candidate.get("sources", [])),
        embedding=raw_candidate.get("embedding"),
        stats=dict(raw_candidate.get("stats", {})),
    )


def consensus_result_to_payload(result) -> dict[str, Any]:
    return {
        "consensus_answer": result.consensus_answer,
        "consensus_cluster_id": result.consensus_cluster_id,
        "selected_agent_ids": result.selected_agent_ids,
        "agreement_score": result.agreement_score,
        "supporting_candidate_count": result.supporting_candidate_count,
        "total_candidates": result.total_candidates,
        "unstable": result.unstable,
        "mode": result.mode,
        "disagreement_edges": [asdict(edge) for edge in result.disagreement_edges],
        "rationale": result.rationale,
    }


def resolve_consensus_payload(payload: dict[str, Any]) -> dict[str, Any]:
    candidates_payload = payload.get("candidates")
    if not isinstance(candidates_payload, list) or not candidates_payload:
        raise ValueError("'candidates' must be a non-empty list")

    mode = payload.get("mode", "quantum_ready")
    if mode not in {"simple_majority", "weighted_majority", "graph_min_cut", "quantum_ready"}:
        raise ValueError(f"Unsupported mode: {mode}")

    candidates = [_candidate_from_payload(candidate) for candidate in candidates_payload]
    result = resolve_consensus(candidates, mode=mode)
    return consensus_result_to_payload(result)


class QuorumHTTPRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/resolve":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)

        try:
            payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
            response = resolve_consensus_payload(payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_json"})
            return
        except (KeyError, TypeError, ValueError) as exc:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "invalid_request", "detail": str(exc)},
            )
            return
        except Exception as exc:  # pragma: no cover - defensive server boundary
            LOGGER.exception("Unexpected error while resolving consensus")
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "internal_error", "detail": str(exc)},
            )
            return

        self._send_json(HTTPStatus.OK, response)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def create_server(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), QuorumHTTPRequestHandler)


def main() -> None:
    server = create_server()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
