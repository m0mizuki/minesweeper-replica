"""Exact enumeration for small Minesweeper CSP instances."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from .factor_graph import FactorGraph, assert_csp_consistent
from .observation import _as_bool_grid

BoolMatrix = NDArray[np.bool_]
FloatVector = NDArray[np.float64]


class ExactEnumerationLimitError(ValueError):
    """Raised when an exact solve would exceed its configured size limit."""


def _frozen_copy(values: NDArray) -> NDArray:
    result = np.array(values, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class OverlapDistribution:
    """A normalized discrete overlap distribution."""

    values: FloatVector
    probabilities: FloatVector

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        probabilities = np.asarray(self.probabilities, dtype=np.float64)
        if values.ndim != 1 or probabilities.ndim != 1:
            raise ValueError("overlap values and probabilities must be one-dimensional")
        if values.shape != probabilities.shape or values.size == 0:
            raise ValueError("overlap values and probabilities must have equal nonzero length")
        if not np.all(np.isfinite(values)) or np.any((values < -1.0) | (values > 1.0)):
            raise ValueError("overlap values must be finite and lie in [-1, 1]")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            raise ValueError("overlap probabilities must be finite and nonnegative")
        if not np.isclose(float(probabilities.sum()), 1.0, rtol=1e-12, atol=1e-14):
            raise ValueError("overlap probabilities must sum to one")
        object.__setattr__(self, "values", _frozen_copy(values))
        object.__setattr__(self, "probabilities", _frozen_copy(probabilities))

    @property
    def mean(self) -> float:
        return float(self.values @ self.probabilities)

    @property
    def variance(self) -> float:
        centered = self.values - self.mean
        return float((centered * centered) @ self.probabilities)


@dataclass(frozen=True)
class ExactResult:
    """Complete exact result for the uniform measure on feasible assignments.

    ``feasible_states`` has one row per constraint-satisfying assignment and
    follows ``graph.variable_cells`` ordering.  Its corresponding normalized
    masses are in ``posterior_probabilities``.  ``rho`` records the planted
    instance's generation density; it does not weight the inference measure.
    """

    rho: float
    feasible_states: BoolMatrix
    posterior_probabilities: FloatVector
    marginals: FloatVector
    partition_function: float
    log_partition_function: float
    mine_count_histogram: tuple[int, ...]
    planted_overlap: OverlapDistribution | None
    replica_overlap: OverlapDistribution | None

    def __post_init__(self) -> None:
        states = np.asarray(self.feasible_states, dtype=np.bool_)
        probabilities = np.asarray(self.posterior_probabilities, dtype=np.float64)
        marginals = np.asarray(self.marginals, dtype=np.float64)
        if states.ndim != 2:
            raise ValueError("feasible_states must be a matrix")
        if probabilities.shape != (states.shape[0],):
            raise ValueError("one posterior probability is required per feasible state")
        if marginals.shape != (states.shape[1],):
            raise ValueError("one marginal is required per variable")
        if len(self.mine_count_histogram) != states.shape[1] + 1:
            raise ValueError("mine_count_histogram must cover counts from 0 through N")
        object.__setattr__(self, "feasible_states", _frozen_copy(states))
        object.__setattr__(self, "posterior_probabilities", _frozen_copy(probabilities))
        object.__setattr__(self, "marginals", _frozen_copy(marginals))

    @property
    def number_of_variables(self) -> int:
        return int(self.feasible_states.shape[1])

    @property
    def number_of_feasible_solutions(self) -> int:
        return int(self.feasible_states.shape[0])

    @property
    def has_posterior_mass(self) -> bool:
        return math.isfinite(self.log_partition_function)


def _validate_limit(number_of_variables: int, max_variables: int) -> None:
    if isinstance(max_variables, bool) or not isinstance(max_variables, (int, np.integer)):
        raise TypeError("max_variables must be an integer")
    if not 0 <= int(max_variables) <= 62:
        raise ValueError("max_variables must lie in [0, 62]")
    if number_of_variables > int(max_variables):
        raise ExactEnumerationLimitError(
            "exact enumeration disabled for this graph: "
            f"{number_of_variables} variables exceeds max_variables={max_variables}"
        )


def _assignment_chunk(start: int, stop: int, number_of_variables: int) -> BoolMatrix:
    codes = np.arange(start, stop, dtype=np.uint64)
    if number_of_variables == 0:
        return np.empty((stop - start, 0), dtype=np.bool_)
    shifts = np.arange(number_of_variables, dtype=np.uint64)
    return ((codes[:, None] >> shifts[None, :]) & np.uint64(1)).astype(np.bool_)


def enumerate_feasible_states(
    graph: FactorGraph,
    *,
    max_variables: int = 20,
    chunk_size: int = 65_536,
) -> BoolMatrix:
    """Enumerate every binary assignment satisfying all graph constraints.

    The explicit ``max_variables`` guard prevents accidentally starting an
    exponential computation on a large experimental instance.
    """

    number_of_variables = graph.number_of_variables
    _validate_limit(number_of_variables, max_variables)
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)):
        raise TypeError("chunk_size must be an integer")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    feasible_chunks: list[BoolMatrix] = []
    number_of_states = 1 << number_of_variables
    for start in range(0, number_of_states, int(chunk_size)):
        states = _assignment_chunk(
            start, min(start + int(chunk_size), number_of_states), number_of_variables
        )
        feasible = np.ones(states.shape[0], dtype=np.bool_)
        for factor, variables in enumerate(graph.factor_to_variables):
            if variables:
                totals = states[:, variables].sum(axis=1)
            else:
                totals = np.zeros(states.shape[0], dtype=np.int_)
            feasible &= totals == graph.clue_values[factor]
            if not feasible.any():
                break
        if feasible.any():
            feasible_chunks.append(states[feasible])

    if feasible_chunks:
        result = np.concatenate(feasible_chunks, axis=0)
    else:
        result = np.empty((0, number_of_variables), dtype=np.bool_)
    result.flags.writeable = False
    return result


def _normalized_uniform_weights(
    states: BoolMatrix,
) -> tuple[FloatVector, float, float]:
    number_of_states = states.shape[0]
    if number_of_states == 0:
        return np.empty(0, dtype=np.float64), 0.0, -math.inf
    partition = float(number_of_states)
    probabilities = np.full(number_of_states, 1.0 / partition, dtype=np.float64)
    return probabilities, partition, math.log(partition)


def _overlap_grid(number_of_variables: int) -> FloatVector:
    if number_of_variables == 0:
        return np.array([1.0], dtype=np.float64)
    distances = np.arange(number_of_variables + 1, dtype=np.float64)
    return 1.0 - 2.0 * distances / number_of_variables


def _planted_overlap_distribution(
    states: BoolMatrix,
    probabilities: FloatVector,
    planted_assignment: NDArray[np.bool_],
) -> OverlapDistribution:
    number_of_variables = states.shape[1]
    if number_of_variables == 0:
        masses = np.array([1.0], dtype=np.float64)
    else:
        distances = np.count_nonzero(states != planted_assignment[None, :], axis=1)
        masses = np.bincount(
            distances, weights=probabilities, minlength=number_of_variables + 1
        ).astype(np.float64)
        masses /= masses.sum()
    return OverlapDistribution(_overlap_grid(number_of_variables), masses)


def _fwht(values: FloatVector) -> FloatVector:
    """Unnormalized Walsh-Hadamard transform."""

    transformed = np.array(values, dtype=np.float64, copy=True)
    width = 1
    while width < transformed.size:
        blocks = transformed.reshape(-1, 2 * width)
        left = blocks[:, :width].copy()
        right = blocks[:, width:].copy()
        blocks[:, :width] = left + right
        blocks[:, width:] = left - right
        width *= 2
    return transformed


def _replica_overlap_distribution(
    states: BoolMatrix, probabilities: FloatVector
) -> OverlapDistribution:
    """Compute exact P(q12) by XOR autocorrelation via a Walsh transform."""

    number_of_variables = states.shape[1]
    if number_of_variables == 0:
        return OverlapDistribution(
            np.array([1.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        )

    number_of_states = 1 << number_of_variables
    shifts = np.arange(number_of_variables, dtype=np.uint64)
    codes = (states.astype(np.uint64) << shifts[None, :]).sum(
        axis=1, dtype=np.uint64
    )
    dense_probabilities = np.zeros(number_of_states, dtype=np.float64)
    dense_probabilities[codes] = probabilities

    spectrum = _fwht(dense_probabilities)
    xor_probabilities = _fwht(spectrum * spectrum) / number_of_states
    # Roundoff from the transform can create tiny negative masses.
    xor_probabilities = np.maximum(xor_probabilities, 0.0)
    xor_probabilities /= xor_probabilities.sum()

    xor_codes = np.arange(number_of_states, dtype=np.uint64)
    distances = np.zeros(number_of_states, dtype=np.uint8)
    for shift in shifts:
        distances += ((xor_codes >> shift) & np.uint64(1)).astype(np.uint8)
    masses = np.bincount(
        distances, weights=xor_probabilities, minlength=number_of_variables + 1
    ).astype(np.float64)
    masses /= masses.sum()
    return OverlapDistribution(_overlap_grid(number_of_variables), masses)


def solve_exact(
    graph: FactorGraph,
    rho: float,
    *,
    planted_ground_truth: NDArray[np.bool_] | None = None,
    max_variables: int = 20,
    chunk_size: int = 65_536,
) -> ExactResult:
    """Enumerate the exact uniform distribution over feasible assignments.

    ``rho`` is retained as planted-instance metadata and is not used as an
    inference prior. If the CSP is unsatisfiable, the result reports ``Z=0``;
    marginals are NaN and overlap distributions are unavailable.
    """

    if isinstance(rho, bool) or not isinstance(rho, (int, float, np.number)):
        raise TypeError("rho must be a real number")
    if not np.isfinite(rho) or not 0.0 <= float(rho) <= 1.0:
        raise ValueError("rho must lie in [0, 1]")
    rho = float(rho)

    states = enumerate_feasible_states(
        graph, max_variables=max_variables, chunk_size=chunk_size
    )
    probabilities, partition, log_partition = _normalized_uniform_weights(states)
    mine_counts = (
        states.sum(axis=1, dtype=np.int_)
        if states.size
        else np.zeros(states.shape[0], dtype=np.int_)
    )
    histogram = tuple(
        int(value)
        for value in np.bincount(
            mine_counts, minlength=graph.number_of_variables + 1
        )
    )

    if math.isfinite(log_partition):
        marginals = probabilities @ states.astype(np.float64)
        replica_overlap = _replica_overlap_distribution(states, probabilities)
    else:
        marginals = np.full(graph.number_of_variables, np.nan, dtype=np.float64)
        replica_overlap = None

    planted_overlap = None
    if planted_ground_truth is not None:
        truth = _as_bool_grid(planted_ground_truth, name="planted_ground_truth")
        if truth.shape != graph.shape:
            raise ValueError("planted_ground_truth shape must equal the graph shape")
        assert_csp_consistent(graph, truth)
        planted_assignment = np.fromiter(
            (bool(truth[cell]) for cell in graph.variable_cells),
            dtype=np.bool_,
            count=graph.number_of_variables,
        )
        if math.isfinite(log_partition):
            planted_overlap = _planted_overlap_distribution(
                states, probabilities, planted_assignment
            )

    return ExactResult(
        rho=rho,
        feasible_states=states,
        posterior_probabilities=probabilities,
        marginals=marginals,
        partition_function=partition,
        log_partition_function=log_partition,
        mine_count_histogram=histogram,
        planted_overlap=planted_overlap,
        replica_overlap=replica_overlap,
    )
