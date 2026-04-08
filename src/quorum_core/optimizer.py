from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any


@dataclass(slots=True)
class OptimizationProblem:
    labels: list[str]
    unary_costs: list[float]
    pairwise_costs: dict[tuple[int, int], float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OptimizationSolution:
    selected_indices: list[int]
    energy: float
    metadata: dict[str, Any] = field(default_factory=dict)


class Optimizer(ABC):
    @abstractmethod
    def solve(self, problem: OptimizationProblem) -> OptimizationSolution:
        raise NotImplementedError


class GreedyOptimizer(Optimizer):
    exact_limit: int = 18

    def solve(self, problem: OptimizationProblem) -> OptimizationSolution:
        if len(problem.unary_costs) <= self.exact_limit:
            selected_indices, energy = self._solve_exact(problem)
        else:
            selected_indices, energy = self._solve_local_search(problem)

        return OptimizationSolution(
            selected_indices=sorted(selected_indices),
            energy=energy,
            metadata={
                "strategy": (
                    "exact"
                    if len(problem.unary_costs) <= self.exact_limit
                    else "local_search"
                ),
                "exact_limit": self.exact_limit,
            },
        )

    def _solve_exact(self, problem: OptimizationProblem) -> tuple[list[int], float]:
        best_indices: list[int] = []
        best_energy = 0.0
        best_found = False

        candidate_count = len(problem.unary_costs)
        for mask in range(1 << candidate_count):
            selected_indices = [index for index in range(candidate_count) if mask & (1 << index)]
            energy = self._energy(selected_indices, problem)
            if not best_found or energy < best_energy:
                best_indices = selected_indices
                best_energy = energy
                best_found = True

        return best_indices, best_energy

    def _solve_local_search(self, problem: OptimizationProblem) -> tuple[list[int], float]:
        selected_indices = [index for index, cost in enumerate(problem.unary_costs) if cost <= 0]
        if not selected_indices and problem.unary_costs:
            seed_index = min(
                range(len(problem.unary_costs)),
                key=problem.unary_costs.__getitem__,
            )
            selected_indices = [seed_index]
        energy = self._energy(selected_indices, problem)

        improved = True
        while improved:
            improved = False
            candidate_indices = list(range(len(problem.unary_costs)))

            for index in candidate_indices:
                if index in selected_indices:
                    continue
                trial = selected_indices + [index]
                trial_energy = self._energy(trial, problem)
                if trial_energy < energy:
                    selected_indices = trial
                    energy = trial_energy
                    improved = True

            for index in candidate_indices:
                trial = [candidate for candidate in selected_indices if candidate != index]
                trial_energy = self._energy(trial, problem)
                if trial_energy < energy:
                    selected_indices = trial
                    energy = trial_energy
                    improved = True

        return selected_indices, energy

    def _energy(self, selected_indices: list[int], problem: OptimizationProblem) -> float:
        selected = set(selected_indices)
        energy = 0.0

        for index in selected:
            energy += problem.unary_costs[index]

        for left_index, right_index in combinations(sorted(selected), 2):
            energy += problem.pairwise_costs.get((left_index, right_index), 0.0)

        return energy
