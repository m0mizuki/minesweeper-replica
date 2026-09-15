"""Core types and functions for planted Minesweeper CSP instances."""

from .factor_graph import (
    FactorGraph,
    assert_csp_consistent,
    build_factor_graph,
    check_csp_consistency,
    constraint_residuals,
)
from .instance import PlantedInstance, generate_ground_truth
from .observation import (
    ObservationProtocol,
    generate_clues,
    generate_observation_mask,
    iter_neighbor_cells,
)

__all__ = [
    "FactorGraph",
    "ObservationProtocol",
    "PlantedInstance",
    "assert_csp_consistent",
    "build_factor_graph",
    "check_csp_consistency",
    "constraint_residuals",
    "generate_clues",
    "generate_ground_truth",
    "generate_observation_mask",
    "iter_neighbor_cells",
]

