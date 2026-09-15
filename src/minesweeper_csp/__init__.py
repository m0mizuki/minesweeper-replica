"""Core types and functions for planted Minesweeper CSP instances."""

from .bp import (
    BPConfig,
    BPInitialization,
    BPMarginalComparison,
    BPMultipleResult,
    BPResult,
    BPStatus,
    compare_bp_to_exact,
    run_bp,
    run_bp_multiple,
)
from .exact import (
    ExactEnumerationLimitError,
    ExactResult,
    OverlapDistribution,
    enumerate_feasible_states,
    solve_exact,
)
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
    "BPConfig",
    "BPInitialization",
    "BPMarginalComparison",
    "BPMultipleResult",
    "BPResult",
    "BPStatus",
    "FactorGraph",
    "ExactEnumerationLimitError",
    "ExactResult",
    "ObservationProtocol",
    "OverlapDistribution",
    "PlantedInstance",
    "assert_csp_consistent",
    "build_factor_graph",
    "check_csp_consistency",
    "compare_bp_to_exact",
    "constraint_residuals",
    "enumerate_feasible_states",
    "generate_clues",
    "generate_ground_truth",
    "generate_observation_mask",
    "iter_neighbor_cells",
    "run_bp",
    "run_bp_multiple",
    "solve_exact",
]
