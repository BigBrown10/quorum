from __future__ import annotations

import os
from dataclasses import dataclass
from importlib.util import find_spec

from .optimizer import OptimizationProblem, OptimizationSolution, Optimizer

OPTIMIZER_ENV_VAR = "QUORUM_OPTIMIZER"
ALLOW_CLASSICAL_FALLBACK_ENV_VAR = "QUORUM_ALLOW_CLASSICAL_FALLBACK"


@dataclass(slots=True)
class QuantumBackendUnavailableError(RuntimeError):
    backend_name: str
    detail: str

    def __post_init__(self) -> None:
        super().__init__(f"{self.backend_name} backend is unavailable: {self.detail}")


class QiskitOptimizer(Optimizer):
    def solve(self, problem: OptimizationProblem) -> OptimizationSolution:
        try:
            from qiskit_aer.primitives import Sampler
            from qiskit_algorithms import QAOA
            from qiskit_optimization.algorithms import MinimumEigenOptimizer
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise QuantumBackendUnavailableError("qiskit", str(exc)) from exc

        quadratic_program = _problem_to_quadratic_program(problem)
        optimizer = MinimumEigenOptimizer(QAOA(sampler=Sampler()))
        result = optimizer.solve(quadratic_program)

        selected_indices = [index for index, value in enumerate(result.x) if value >= 0.5]
        return OptimizationSolution(
            selected_indices=selected_indices,
            energy=float(result.fval),
            metadata={
                "strategy": "qiskit-qaoa",
                "samples": getattr(result, "min_eigen_solver_result", None) is not None,
            },
        )


class DWaveOptimizer(Optimizer):
    def solve(self, problem: OptimizationProblem) -> OptimizationSolution:
        try:
            import dimod
            from dwave.system import DWaveSampler, EmbeddingComposite
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise QuantumBackendUnavailableError("dwave", str(exc)) from exc

        linear, quadratic, offset = _problem_to_bqm(problem)
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, dimod.BINARY)

        sampler = EmbeddingComposite(DWaveSampler())
        sample_set = sampler.sample(bqm, num_reads=100)
        best_sample = sample_set.first.sample
        selected_indices = [
            index
            for index, _label in enumerate(problem.labels)
            if int(best_sample.get(f"x_{index}", 0)) == 1
        ]

        return OptimizationSolution(
            selected_indices=selected_indices,
            energy=float(sample_set.first.energy),
            metadata={
                "strategy": "dwave-ocean",
                "num_reads": 100,
            },
        )


def get_optimizer(name: str | None = None) -> Optimizer:
    if name is None:
        requested_name = os.getenv(OPTIMIZER_ENV_VAR) or "greedy"
    else:
        requested_name = name
    normalized = requested_name.strip().lower()
    if normalized in {"greedy", "classical", "default"}:
        from .optimizer import GreedyOptimizer

        return GreedyOptimizer()
    if normalized in {"qiskit", "qaoa", "quantum"}:
        if not _qiskit_available():
            if _allow_classical_fallback():
                return get_optimizer("greedy")
            raise QuantumBackendUnavailableError("qiskit", "required packages are not installed")
        return QiskitOptimizer()
    if normalized in {"dwave", "ocean", "annealing"}:
        if not _dwave_available():
            if _allow_classical_fallback():
                return get_optimizer("greedy")
            raise QuantumBackendUnavailableError("dwave", "required packages are not installed")
        return DWaveOptimizer()

    raise ValueError(f"Unknown optimizer: {name}")


def _allow_classical_fallback() -> bool:
    value = os.getenv(ALLOW_CLASSICAL_FALLBACK_ENV_VAR, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _qiskit_available() -> bool:
    return find_spec("qiskit_algorithms") is not None and find_spec("qiskit_aer") is not None


def _dwave_available() -> bool:
    return find_spec("dwave.system") is not None


def _problem_to_quadratic_program(problem: OptimizationProblem):
    from qiskit_optimization import QuadraticProgram

    quadratic_program = QuadraticProgram()
    for index, _label in enumerate(problem.labels):
        quadratic_program.binary_var(name=f"x_{index}")

    linear = {
        f"x_{index}": problem.unary_costs[index]
        for index in range(len(problem.unary_costs))
    }
    quadratic = {}
    for (left_index, right_index), cost in problem.pairwise_costs.items():
        quadratic[(f"x_{left_index}", f"x_{right_index}")] = cost

    quadratic_program.minimize(linear=linear, quadratic=quadratic)
    return quadratic_program


def _problem_to_bqm(problem: OptimizationProblem):
    linear = {
        f"x_{index}": problem.unary_costs[index]
        for index in range(len(problem.unary_costs))
    }
    quadratic = {
        (f"x_{left_index}", f"x_{right_index}"): cost
        for (left_index, right_index), cost in problem.pairwise_costs.items()
    }
    return linear, quadratic, 0.0
