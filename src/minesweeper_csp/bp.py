"""Belief propagation for the planted Minesweeper factor graph."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from .exact import ExactResult
from .factor_graph import FactorGraph

BPInitialization = Literal["uniform", "prior", "random"]
BPStatus = Literal["converged", "max_iterations", "infeasible"]
FloatMatrix = NDArray[np.float64]
FloatVector = NDArray[np.float64]


def _frozen_copy(values: NDArray) -> NDArray:
    result = np.array(values, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class BPConfig:
    """Numerical and initialization settings for one BP run.

    ``damping`` is the fraction of the old message retained at each update:
    ``damped = damping * old + (1 - damping) * new``.
    """

    max_iterations: int = 1_000
    tolerance: float = 1e-10
    damping: float = 0.0
    initialization: BPInitialization = "prior"
    seed: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.max_iterations, bool) or not isinstance(
            self.max_iterations, (int, np.integer)
        ):
            raise TypeError("max_iterations must be an integer")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if not isinstance(self.tolerance, (int, float, np.number)) or isinstance(
            self.tolerance, bool
        ):
            raise TypeError("tolerance must be a real number")
        if not math.isfinite(float(self.tolerance)) or self.tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")
        if not isinstance(self.damping, (int, float, np.number)) or isinstance(
            self.damping, bool
        ):
            raise TypeError("damping must be a real number")
        if not math.isfinite(float(self.damping)) or not 0.0 <= self.damping < 1.0:
            raise ValueError("damping must lie in [0, 1)")
        if self.initialization not in ("uniform", "prior", "random"):
            raise ValueError(f"unknown BP initialization: {self.initialization!r}")


@dataclass(frozen=True)
class BPResult:
    """Messages, marginals, and convergence diagnostics from one BP run."""

    config: BPConfig
    rho: float
    status: BPStatus
    iterations: int
    final_max_message_delta: float
    residual_history: tuple[float, ...]
    marginals: FloatVector
    variable_to_factor_messages: FloatMatrix
    factor_to_variable_messages: FloatMatrix
    edge_factors: tuple[int, ...]
    edge_variables: tuple[int, ...]
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        marginals = np.asarray(self.marginals, dtype=np.float64)
        variable_to_factor = np.asarray(
            self.variable_to_factor_messages, dtype=np.float64
        )
        factor_to_variable = np.asarray(
            self.factor_to_variable_messages, dtype=np.float64
        )
        if variable_to_factor.ndim != 2 or variable_to_factor.shape[1:] != (2,):
            raise ValueError("variable-to-factor messages must have shape (E, 2)")
        if factor_to_variable.shape != variable_to_factor.shape:
            raise ValueError("both message arrays must have the same shape")
        if len(self.edge_factors) != variable_to_factor.shape[0]:
            raise ValueError("edge_factors must contain one entry per message")
        if len(self.edge_variables) != variable_to_factor.shape[0]:
            raise ValueError("edge_variables must contain one entry per message")
        object.__setattr__(self, "marginals", _frozen_copy(marginals))
        object.__setattr__(
            self, "variable_to_factor_messages", _frozen_copy(variable_to_factor)
        )
        object.__setattr__(
            self, "factor_to_variable_messages", _frozen_copy(factor_to_variable)
        )

    @property
    def converged(self) -> bool:
        return self.status == "converged"


@dataclass(frozen=True)
class BPMultipleResult:
    """Results used to diagnose initialization-dependent BP fixed points."""

    runs: tuple[BPResult, ...]

    @property
    def all_converged(self) -> bool:
        return bool(self.runs) and all(run.converged for run in self.runs)

    @property
    def max_pairwise_marginal_difference(self) -> float | None:
        converged = [run for run in self.runs if run.converged]
        if len(converged) < 2:
            return None
        maximum = 0.0
        for left_index, left in enumerate(converged[:-1]):
            for right in converged[left_index + 1 :]:
                if left.marginals.size:
                    difference = float(np.max(np.abs(left.marginals - right.marginals)))
                    maximum = max(maximum, difference)
        return maximum

    def fixed_point_groups(self, tolerance: float = 1e-6) -> tuple[tuple[int, ...], ...]:
        """Cluster converged runs by maximum marginal difference."""

        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")
        groups: list[list[int]] = []
        representatives: list[FloatVector] = []
        for index, run in enumerate(self.runs):
            if not run.converged:
                continue
            for group, representative in zip(groups, representatives):
                difference = (
                    float(np.max(np.abs(run.marginals - representative)))
                    if run.marginals.size
                    else 0.0
                )
                if difference <= tolerance:
                    group.append(index)
                    break
            else:
                groups.append([index])
                representatives.append(run.marginals)
        return tuple(tuple(group) for group in groups)


@dataclass(frozen=True)
class BPMarginalComparison:
    """Errors between BP and a normalized exact posterior."""

    mae: float
    rmse: float
    max_absolute_error: float


@dataclass(frozen=True)
class _EdgeLayout:
    edge_factors: tuple[int, ...]
    edge_variables: tuple[int, ...]
    factor_edges: tuple[tuple[int, ...], ...]
    variable_edges: tuple[tuple[int, ...], ...]


def _build_edge_layout(graph: FactorGraph) -> _EdgeLayout:
    edge_factors: list[int] = []
    edge_variables: list[int] = []
    factor_edges: list[list[int]] = [
        [] for _ in range(graph.number_of_constraints)
    ]
    variable_edges: list[list[int]] = [
        [] for _ in range(graph.number_of_variables)
    ]
    for factor, variables in enumerate(graph.factor_to_variables):
        for variable in variables:
            edge = len(edge_factors)
            edge_factors.append(factor)
            edge_variables.append(variable)
            factor_edges[factor].append(edge)
            variable_edges[variable].append(edge)
    return _EdgeLayout(
        edge_factors=tuple(edge_factors),
        edge_variables=tuple(edge_variables),
        factor_edges=tuple(tuple(edges) for edges in factor_edges),
        variable_edges=tuple(tuple(edges) for edges in variable_edges),
    )


def _log_probabilities(probabilities: FloatVector) -> FloatVector:
    result = np.full(2, -math.inf, dtype=np.float64)
    positive = probabilities > 0.0
    result[positive] = np.log(probabilities[positive])
    return result


def _normalize_logs(log_values: FloatVector) -> FloatVector | None:
    maximum = float(np.max(log_values))
    if not math.isfinite(maximum):
        return None
    probabilities = np.exp(log_values - maximum)
    total = float(probabilities.sum())
    if not math.isfinite(total) or total <= 0.0:
        return None
    return probabilities / total


def _damp(new: FloatVector, old: FloatVector, damping: float) -> FloatVector:
    result = (1.0 - damping) * new + damping * old
    return result / result.sum()


def _initialize_variable_messages(
    number_of_edges: int,
    prior: FloatVector,
    config: BPConfig,
) -> FloatMatrix:
    if config.initialization == "uniform":
        return np.full((number_of_edges, 2), 0.5, dtype=np.float64)
    if config.initialization == "prior":
        return np.tile(prior, (number_of_edges, 1))
    rng = np.random.default_rng(config.seed)
    messages = rng.random((number_of_edges, 2)) + np.finfo(np.float64).eps
    messages /= messages.sum(axis=1, keepdims=True)
    return messages


def _factor_message(
    target_edge: int,
    factor: int,
    clue: int,
    layout: _EdgeLayout,
    variable_to_factor: FloatMatrix,
) -> FloatVector | None:
    log_distribution = np.array([0.0], dtype=np.float64)
    for edge in layout.factor_edges[factor]:
        if edge == target_edge:
            continue
        log_message = _log_probabilities(variable_to_factor[edge])
        updated = np.full(log_distribution.size + 1, -math.inf, dtype=np.float64)
        updated[:-1] = np.logaddexp(
            updated[:-1], log_distribution + log_message[0]
        )
        updated[1:] = np.logaddexp(
            updated[1:], log_distribution + log_message[1]
        )
        log_distribution = updated

    log_values = np.full(2, -math.inf, dtype=np.float64)
    if 0 <= clue < log_distribution.size:
        log_values[0] = log_distribution[clue]
    if 0 <= clue - 1 < log_distribution.size:
        log_values[1] = log_distribution[clue - 1]
    return _normalize_logs(log_values)


def _variable_message(
    target_edge: int,
    variable: int,
    prior: FloatVector,
    layout: _EdgeLayout,
    factor_to_variable: FloatMatrix,
) -> FloatVector | None:
    log_values = _log_probabilities(prior)
    for edge in layout.variable_edges[variable]:
        if edge != target_edge:
            log_values += _log_probabilities(factor_to_variable[edge])
    return _normalize_logs(log_values)


def _compute_marginals(
    graph: FactorGraph,
    prior: FloatVector,
    layout: _EdgeLayout,
    factor_to_variable: FloatMatrix,
) -> FloatVector | None:
    marginals = np.empty(graph.number_of_variables, dtype=np.float64)
    for variable, edges in enumerate(layout.variable_edges):
        log_values = _log_probabilities(prior)
        for edge in edges:
            log_values += _log_probabilities(factor_to_variable[edge])
        probabilities = _normalize_logs(log_values)
        if probabilities is None:
            return None
        marginals[variable] = probabilities[1]
    return marginals


def _make_result(
    graph: FactorGraph,
    config: BPConfig,
    rho: float,
    status: BPStatus,
    iterations: int,
    residual_history: list[float],
    variable_to_factor: FloatMatrix,
    factor_to_variable: FloatMatrix,
    layout: _EdgeLayout,
    marginals: FloatVector | None,
    failure_reason: str | None = None,
) -> BPResult:
    if marginals is None:
        marginals = np.full(graph.number_of_variables, np.nan, dtype=np.float64)
    final_delta = residual_history[-1] if residual_history else 0.0
    return BPResult(
        config=config,
        rho=rho,
        status=status,
        iterations=iterations,
        final_max_message_delta=final_delta,
        residual_history=tuple(residual_history),
        marginals=marginals,
        variable_to_factor_messages=variable_to_factor,
        factor_to_variable_messages=factor_to_variable,
        edge_factors=layout.edge_factors,
        edge_variables=layout.edge_variables,
        failure_reason=failure_reason,
    )


def run_bp(
    graph: FactorGraph,
    rho: float,
    *,
    config: BPConfig | None = None,
) -> BPResult:
    """Run normalized sum-product BP for the Bernoulli-prior posterior."""

    if isinstance(rho, bool) or not isinstance(rho, (int, float, np.number)):
        raise TypeError("rho must be a real number")
    if not np.isfinite(rho) or not 0.0 <= float(rho) <= 1.0:
        raise ValueError("rho must lie in [0, 1]")
    rho = float(rho)
    config = config if config is not None else BPConfig()
    if not isinstance(config, BPConfig):
        raise TypeError("config must be a BPConfig")

    layout = _build_edge_layout(graph)
    prior = np.array([1.0 - rho, rho], dtype=np.float64)
    variable_to_factor = _initialize_variable_messages(
        len(layout.edge_factors), prior, config
    )
    factor_to_variable = np.full_like(variable_to_factor, 0.5)
    residual_history: list[float] = []

    for factor, edges in enumerate(layout.factor_edges):
        clue = graph.clue_values[factor]
        if not 0 <= clue <= len(edges):
            return _make_result(
                graph,
                config,
                rho,
                "infeasible",
                0,
                residual_history,
                variable_to_factor,
                factor_to_variable,
                layout,
                None,
                f"clue {clue} at {graph.clue_cells[factor]} exceeds factor degree {len(edges)}",
            )

    if not layout.edge_factors:
        marginals = np.full(graph.number_of_variables, rho, dtype=np.float64)
        return _make_result(
            graph,
            config,
            rho,
            "converged",
            0,
            residual_history,
            variable_to_factor,
            factor_to_variable,
            layout,
            marginals,
        )

    for iteration in range(1, config.max_iterations + 1):
        updated_factor_to_variable = np.empty_like(factor_to_variable)
        for edge, factor in enumerate(layout.edge_factors):
            message = _factor_message(
                edge,
                factor,
                graph.clue_values[factor],
                layout,
                variable_to_factor,
            )
            if message is None:
                return _make_result(
                    graph,
                    config,
                    rho,
                    "infeasible",
                    iteration,
                    residual_history,
                    variable_to_factor,
                    factor_to_variable,
                    layout,
                    None,
                    f"factor {factor} produced zero mass for both states",
                )
            updated_factor_to_variable[edge] = _damp(
                message, factor_to_variable[edge], config.damping
            )

        updated_variable_to_factor = np.empty_like(variable_to_factor)
        for edge, variable in enumerate(layout.edge_variables):
            message = _variable_message(
                edge,
                variable,
                prior,
                layout,
                updated_factor_to_variable,
            )
            if message is None:
                return _make_result(
                    graph,
                    config,
                    rho,
                    "infeasible",
                    iteration,
                    residual_history,
                    variable_to_factor,
                    updated_factor_to_variable,
                    layout,
                    None,
                    f"variable {variable} produced zero mass for both states",
                )
            updated_variable_to_factor[edge] = _damp(
                message, variable_to_factor[edge], config.damping
            )

        delta = max(
            float(np.max(np.abs(updated_factor_to_variable - factor_to_variable))),
            float(np.max(np.abs(updated_variable_to_factor - variable_to_factor))),
        )
        residual_history.append(delta)
        factor_to_variable = updated_factor_to_variable
        variable_to_factor = updated_variable_to_factor

        if delta < config.tolerance:
            marginals = _compute_marginals(
                graph, prior, layout, factor_to_variable
            )
            if marginals is None:
                return _make_result(
                    graph,
                    config,
                    rho,
                    "infeasible",
                    iteration,
                    residual_history,
                    variable_to_factor,
                    factor_to_variable,
                    layout,
                    None,
                    "belief normalization has zero posterior mass",
                )
            return _make_result(
                graph,
                config,
                rho,
                "converged",
                iteration,
                residual_history,
                variable_to_factor,
                factor_to_variable,
                layout,
                marginals,
            )

    marginals = _compute_marginals(graph, prior, layout, factor_to_variable)
    status: BPStatus = "max_iterations" if marginals is not None else "infeasible"
    return _make_result(
        graph,
        config,
        rho,
        status,
        config.max_iterations,
        residual_history,
        variable_to_factor,
        factor_to_variable,
        layout,
        marginals,
        None if marginals is not None else "belief normalization has zero posterior mass",
    )


def run_bp_multiple(
    graph: FactorGraph,
    rho: float,
    configs: Sequence[BPConfig],
) -> BPMultipleResult:
    """Run BP from several explicitly recorded initialization configurations."""

    configs = tuple(configs)
    if not configs:
        raise ValueError("at least one BPConfig is required")
    return BPMultipleResult(tuple(run_bp(graph, rho, config=config) for config in configs))


def compare_bp_to_exact(bp: BPResult, exact: ExactResult) -> BPMarginalComparison:
    """Compare BP marginals with an exact result for the same variable ordering."""

    if not exact.has_posterior_mass:
        raise ValueError("exact posterior has zero mass")
    if bp.rho != exact.rho:
        raise ValueError("BP and exact results use different rho values")
    if bp.marginals.shape != exact.marginals.shape:
        raise ValueError("BP and exact results have different numbers of variables")
    if not np.all(np.isfinite(bp.marginals)):
        raise ValueError("BP marginals are unavailable")
    differences = bp.marginals - exact.marginals
    if differences.size == 0:
        return BPMarginalComparison(0.0, 0.0, 0.0)
    return BPMarginalComparison(
        mae=float(np.mean(np.abs(differences))),
        rmse=float(np.sqrt(np.mean(differences * differences))),
        max_absolute_error=float(np.max(np.abs(differences))),
    )
