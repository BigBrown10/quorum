import pytest

from quorum_core.optimizer import GreedyOptimizer, OptimizationProblem
from quorum_core.quantum import ALLOW_CLASSICAL_FALLBACK_ENV_VAR, OPTIMIZER_ENV_VAR, get_optimizer


def _energy(selected_indices: list[int], problem: OptimizationProblem) -> float:
    selected = set(selected_indices)
    total = 0.0
    for index in selected:
        total += problem.unary_costs[index]
    for (left_index, right_index), cost in problem.pairwise_costs.items():
        if left_index in selected and right_index in selected:
            total += cost
    return total


def test_optimizer_finds_global_minimum_for_small_problem() -> None:
    problem = OptimizationProblem(
        labels=["a", "b", "c", "d"],
        unary_costs=[-0.9, -0.9, -0.1, -0.1],
        pairwise_costs={
            (0, 1): 0.1,
            (0, 2): 0.8,
            (0, 3): 0.8,
            (1, 2): 0.8,
            (1, 3): 0.8,
            (2, 3): 0.2,
        },
    )

    optimizer = GreedyOptimizer()
    solution = optimizer.solve(problem)

    best_indices: list[int] = []
    best_energy: float | None = None
    for mask in range(1 << len(problem.unary_costs)):
        selected = [index for index in range(len(problem.unary_costs)) if mask & (1 << index)]
        energy = _energy(selected, problem)
        if best_energy is None or energy < best_energy:
            best_indices = selected
            best_energy = energy

    assert solution.selected_indices == best_indices
    assert solution.energy == best_energy
    assert solution.metadata["strategy"] == "exact"


def test_get_optimizer_defaults_to_greedy(monkeypatch) -> None:
    monkeypatch.delenv(OPTIMIZER_ENV_VAR, raising=False)
    monkeypatch.delenv(ALLOW_CLASSICAL_FALLBACK_ENV_VAR, raising=False)

    optimizer = get_optimizer()

    assert isinstance(optimizer, GreedyOptimizer)


def test_get_optimizer_can_fall_back_when_quantum_backend_is_unavailable(monkeypatch) -> None:
    monkeypatch.setenv(OPTIMIZER_ENV_VAR, "qiskit")
    monkeypatch.setenv(ALLOW_CLASSICAL_FALLBACK_ENV_VAR, "true")

    optimizer = get_optimizer()

    assert isinstance(optimizer, GreedyOptimizer)


def test_get_optimizer_raises_when_quantum_backend_is_unavailable_and_fallback_disabled(
    monkeypatch,
) -> None:
    monkeypatch.setenv(OPTIMIZER_ENV_VAR, "dwave")
    monkeypatch.delenv(ALLOW_CLASSICAL_FALLBACK_ENV_VAR, raising=False)

    with pytest.raises(Exception) as exc_info:
        get_optimizer()

    assert "dwave" in str(exc_info.value).lower()
