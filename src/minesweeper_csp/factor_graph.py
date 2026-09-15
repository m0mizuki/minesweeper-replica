"""Bipartite factor graph representation of a Minesweeper CSP."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .observation import Cell, _as_bool_grid, iter_neighbor_cells

IntVector = NDArray[np.int_]


@dataclass(frozen=True)
class FactorGraph:
    """Factor graph with unobserved cells as binary variables.

    Indices into ``factor_to_variables`` and ``variable_to_factors`` refer to
    positions in ``clue_cells`` and ``variable_cells``, respectively.
    """

    shape: tuple[int, int]
    variable_cells: tuple[Cell, ...]
    clue_cells: tuple[Cell, ...]
    clue_values: tuple[int, ...]
    factor_to_variables: tuple[tuple[int, ...], ...]
    variable_to_factors: tuple[tuple[int, ...], ...]
    cell_to_variable: Mapping[Cell, int]
    cell_to_factor: Mapping[Cell, int]

    @property
    def number_of_variables(self) -> int:
        return len(self.variable_cells)

    @property
    def number_of_constraints(self) -> int:
        return len(self.clue_cells)


def build_factor_graph(observation_mask: NDArray[np.bool_], clues: ArrayLike) -> FactorGraph:
    """Construct the factor graph induced by an observation mask and clues."""

    observed = _as_bool_grid(observation_mask, name="observation_mask")
    clue_array = np.asarray(clues)
    if clue_array.shape != observed.shape:
        raise ValueError("clues and observation_mask must have equal shapes")
    if not np.issubdtype(clue_array.dtype, np.integer):
        raise TypeError("clues must have integer dtype")
    if np.any(clue_array[observed] < 0) or np.any(clue_array[observed] > 8):
        raise ValueError("observed clue values must lie in [0, 8]")

    variable_cells = tuple(
        (int(cell[0]), int(cell[1])) for cell in np.argwhere(~observed)
    )
    clue_cells = tuple(
        (int(cell[0]), int(cell[1])) for cell in np.argwhere(observed)
    )
    cell_to_variable_dict = {cell: i for i, cell in enumerate(variable_cells)}
    cell_to_factor_dict = {cell: a for a, cell in enumerate(clue_cells)}

    factor_to_variables = tuple(
        tuple(
            cell_to_variable_dict[neighbor]
            for neighbor in iter_neighbor_cells(cell, observed.shape)
            if neighbor in cell_to_variable_dict
        )
        for cell in clue_cells
    )

    variable_to_factors_lists: list[list[int]] = [
        [] for _ in range(len(variable_cells))
    ]
    for factor, variables in enumerate(factor_to_variables):
        for variable in variables:
            variable_to_factors_lists[variable].append(factor)

    return FactorGraph(
        shape=(int(observed.shape[0]), int(observed.shape[1])),
        variable_cells=variable_cells,
        clue_cells=clue_cells,
        clue_values=tuple(int(clue_array[cell]) for cell in clue_cells),
        factor_to_variables=factor_to_variables,
        variable_to_factors=tuple(tuple(v) for v in variable_to_factors_lists),
        cell_to_variable=MappingProxyType(cell_to_variable_dict),
        cell_to_factor=MappingProxyType(cell_to_factor_dict),
    )


def _validate_assignment(graph: FactorGraph, assignment: ArrayLike) -> IntVector:
    values = np.asarray(assignment)
    if values.shape != (graph.number_of_variables,):
        raise ValueError(
            "assignment length must equal number of variables: "
            f"expected {graph.number_of_variables}, got shape {values.shape}"
        )
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("assignment must contain only binary values 0 or 1")
    return values.astype(np.int_, copy=False)


def constraint_residuals(graph: FactorGraph, assignment: ArrayLike) -> IntVector:
    """Return ``sum(neighbor variables) - clue`` for every constraint."""

    values = _validate_assignment(graph, assignment)
    residuals = np.empty(graph.number_of_constraints, dtype=np.int_)
    for factor, variables in enumerate(graph.factor_to_variables):
        residuals[factor] = int(values[list(variables)].sum()) - graph.clue_values[factor]
    return residuals


def _assignment_from_ground_truth(
    graph: FactorGraph, ground_truth: NDArray[np.bool_]
) -> IntVector:
    truth = _as_bool_grid(ground_truth, name="ground_truth")
    if truth.shape != graph.shape:
        raise ValueError("ground_truth shape must equal the factor graph shape")
    return np.fromiter(
        (int(truth[cell]) for cell in graph.variable_cells),
        dtype=np.int_,
        count=graph.number_of_variables,
    )


def check_csp_consistency(
    graph: FactorGraph, ground_truth: NDArray[np.bool_]
) -> bool:
    """Return whether the planted grid is safe at clues and satisfies all factors."""

    truth = _as_bool_grid(ground_truth, name="ground_truth")
    if truth.shape != graph.shape:
        raise ValueError("ground_truth shape must equal the factor graph shape")
    if any(bool(truth[cell]) for cell in graph.clue_cells):
        return False
    assignment = _assignment_from_ground_truth(graph, truth)
    return bool(np.all(constraint_residuals(graph, assignment) == 0))


def assert_csp_consistent(
    graph: FactorGraph, ground_truth: NDArray[np.bool_]
) -> None:
    """Raise ``ValueError`` with details unless the planted grid satisfies the CSP."""

    truth = _as_bool_grid(ground_truth, name="ground_truth")
    if truth.shape != graph.shape:
        raise ValueError("ground_truth shape must equal the factor graph shape")
    mined_clues = [cell for cell in graph.clue_cells if bool(truth[cell])]
    if mined_clues:
        raise ValueError(f"observed clue cells contain mines: {mined_clues}")

    assignment = _assignment_from_ground_truth(graph, truth)
    residuals = constraint_residuals(graph, assignment)
    inconsistent = np.flatnonzero(residuals)
    if inconsistent.size:
        details = [
            {
                "cell": graph.clue_cells[int(factor)],
                "clue": graph.clue_values[int(factor)],
                "residual": int(residuals[int(factor)]),
            }
            for factor in inconsistent
        ]
        raise ValueError(f"planted configuration violates constraints: {details}")
