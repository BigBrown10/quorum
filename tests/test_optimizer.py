import sys
import types

import pytest

from quorum_core.optimizer import GreedyOptimizer, OptimizationProblem
from quorum_core.quantum import (
    ALLOW_CLASSICAL_FALLBACK_ENV_VAR,
    OPTIMIZER_ENV_VAR,
    DWaveOptimizer,
    get_optimizer,
)


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


def test_local_search_seeds_with_lowest_cost_candidate() -> None:
    problem = OptimizationProblem(
        labels=["a", "b"],
        unary_costs=[1.0, 2.0],
        pairwise_costs={(0, 1): -3.0},
    )

    optimizer = GreedyOptimizer()
    optimizer.exact_limit = 0

    solution = optimizer.solve(problem)

    assert solution.selected_indices == [0, 1]
    assert solution.energy == pytest.approx(0.0)
    assert solution.metadata["strategy"] == "local_search"


def test_dwave_optimizer_falls_back_to_simulated_annealing(monkeypatch) -> None:
    dimod_module = types.ModuleType("dimod")

    class _BinaryQuadraticModel:
        def __init__(self, linear, quadratic, offset, vartype):
            self.linear = linear
            self.quadratic = quadratic
            self.offset = offset
            self.vartype = vartype

    dimod_module.BinaryQuadraticModel = _BinaryQuadraticModel
    dimod_module.BINARY = object()

    reference_module = types.ModuleType("dimod.reference")
    samplers_module = types.ModuleType("dimod.reference.samplers")

    class _SampleSet:
        def __init__(self):
            self.first = types.SimpleNamespace(sample={"x_0": 1, "x_1": 0}, energy=-1.0)

    class _SimulatedAnnealingSampler:
        def sample(self, bqm, num_reads):
            return _SampleSet()

    samplers_module.SimulatedAnnealingSampler = _SimulatedAnnealingSampler
    reference_module.samplers = samplers_module
    dimod_module.reference = reference_module

    dwave_system_module = types.ModuleType("dwave.system")
    dwave_module = types.ModuleType("dwave")

    class _DWaveSampler:
        def __init__(self):
            raise RuntimeError("missing token")

    class _EmbeddingComposite:
        def __init__(self, sampler):
            self.sampler = sampler

        def sample(self, bqm, num_reads):
            raise AssertionError("should not reach live sampler path")

    dwave_system_module.DWaveSampler = _DWaveSampler
    dwave_system_module.EmbeddingComposite = _EmbeddingComposite
    dwave_module.system = dwave_system_module

    monkeypatch.setitem(sys.modules, "dimod", dimod_module)
    monkeypatch.setitem(sys.modules, "dimod.reference", reference_module)
    monkeypatch.setitem(sys.modules, "dimod.reference.samplers", samplers_module)
    monkeypatch.setitem(sys.modules, "dwave", dwave_module)
    monkeypatch.setitem(sys.modules, "dwave.system", dwave_system_module)

    problem = OptimizationProblem(labels=["a", "b"], unary_costs=[-0.5, 0.1])

    result = DWaveOptimizer().solve(problem)

    assert result.metadata["strategy"] == "simulated-annealing"
    assert result.selected_indices == [0]
