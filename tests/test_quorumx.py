from quorumx import LangChainOpenAIBackend, QuorumX, QuorumXConfig, QuorumXResult, quorum_x


def test_quorumx_mock_run_returns_structured_result() -> None:
    engine = QuorumX(
        QuorumXConfig(
            n_agents=3,
            max_rounds=2,
            stability_threshold=0.65,
            consensus_mode="quantum_ready",
            mock_mode=True,
        )
    )

    result = engine.run(
        "Should we prioritize retention or acquisition for a SaaS with $8k MRR and 12% churn?"
    )

    assert isinstance(result, QuorumXResult)
    assert result.answer
    assert result.rounds_used >= 1
    assert result.total_tokens > 0
    assert len(result.tokens_per_round) == result.rounds_used
    assert len(result.benchmark) == 3
    assert result.consensus_mode == "quantum_ready"


def test_quorumx_decorator_replaces_original_return_value() -> None:
    called = False

    @quorum_x(
        QuorumXConfig(
            n_agents=2,
            max_rounds=1,
            stability_threshold=0.6,
            consensus_mode="quantum_ready",
            mock_mode=True,
        )
    )
    def legacy_agent(task: str) -> str:
        nonlocal called
        called = True
        return "ignored"

    result = legacy_agent("Draft a short launch email for a product update.")

    assert isinstance(result, QuorumXResult)
    assert called is False
    assert result.unstable in {True, False}


def test_quorumx_simple_majority_marks_unstable_when_candidates_disagree() -> None:
    engine = QuorumX(
        QuorumXConfig(
            n_agents=3,
            max_rounds=1,
            stability_threshold=0.95,
            consensus_mode="simple_majority",
            mock_mode=True,
        )
    )

    result = engine.run("Draft a short launch email for a product update.")

    assert result.unstable is True
    assert result.answer == "NO CONSENSUS"


def test_quorumx_real_mode_selects_lazy_openai_backend() -> None:
    engine = QuorumX(
        QuorumXConfig(
            n_agents=2,
            max_rounds=1,
            stability_threshold=0.6,
            consensus_mode="quantum_ready",
            mock_mode=False,
        )
    )

    assert isinstance(engine.backend, LangChainOpenAIBackend)