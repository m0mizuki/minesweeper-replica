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
from .diagnostics import (
    effective_sample_size,
    gelman_rubin_r_hat,
    integrated_autocorrelation_time,
)
from .mcmc import (
    MCMCConfig,
    MCMCMarginalComparison,
    MCMCMultipleResult,
    MCMCResult,
    compare_mcmc_to_exact,
    run_blocked_gibbs,
    run_blocked_gibbs_chains,
)
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
    "compare_mcmc_to_exact",
    "effective_sample_size",
    "enumerate_feasible_states",
    "generate_clues",
    "generate_ground_truth",
    "generate_observation_mask",
    "gelman_rubin_r_hat",
    "integrated_autocorrelation_time",
    "iter_neighbor_cells",
    "run_bp",
    "run_bp_multiple",
    "run_blocked_gibbs",
    "run_blocked_gibbs_chains",
    "solve_exact",
    "MCMCConfig",
    "MCMCMarginalComparison",
    "MCMCMultipleResult",
    "MCMCResult",
]
