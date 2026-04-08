from quorum_core.api import resolve_consensus_payload


def test_resolve_consensus_payload_serializes_response() -> None:
    payload = {
        "candidates": [
            {"id": "a1", "content": "yes", "confidence": 0.3},
            {"id": "a2", "content": "yes", "confidence": 0.9},
            {"id": "a3", "content": "no", "confidence": 0.4},
        ],
        "mode": "weighted_majority",
    }

    response = resolve_consensus_payload(payload)

    assert response["consensus_answer"] == "yes"
    assert response["selected_agent_ids"] == ["a1", "a2"]
    assert response["mode"] == "weighted_majority"
    assert response["unstable"] is False
    assert response["disagreement_edges"]
    assert "rationale" in response


def test_resolve_consensus_payload_accepts_unstable_threshold() -> None:
    payload = {
        "candidates": [
            {"id": "a1", "content": "yes", "confidence": 0.3},
            {"id": "a2", "content": "yes", "confidence": 0.9},
            {"id": "a3", "content": "no", "confidence": 0.4},
        ],
        "mode": "graph_min_cut",
        "unstable_threshold": 0.95,
    }

    response = resolve_consensus_payload(payload)

    assert response["unstable"] is True


def test_resolve_consensus_payload_rejects_empty_candidates() -> None:
    try:
        resolve_consensus_payload({"candidates": []})
    except ValueError as exc:
        assert "non-empty list" in str(exc)
    else:
        raise AssertionError("expected ValueError")
