"""Blocked Gibbs sampling for hard-constraint Minesweeper posteriors."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .diagnostics import (
    effective_sample_size,
    gelman_rubin_r_hat,
    integrated_autocorrelation_time,
)
from .exact import ExactResult
from .factor_graph import FactorGraph, assert_csp_consistent, constraint_residuals
from .observation import _as_bool_grid

BoolMatrix = NDArray[np.bool_]
BoolVector = NDArray[np.bool_]
FloatVector = NDArray[np.float64]


def _frozen_copy(values: NDArray) -> NDArray:
    result = np.array(values, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class MCMCConfig:
    """Settings for one local blocked-Gibbs chain."""

    block_size: int = 4
    burn_in: int = 1_000
    samples: int = 5_000
    thinning: int = 1
    seed: int | None = None

    def __post_init__(self) -> None:
        integer_fields = {
            "block_size": self.block_size,
            "burn_in": self.burn_in,
            "samples": self.samples,
            "thinning": self.thinning,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
        if not 1 <= self.block_size <= 20:
            raise ValueError("block_size must lie in [1, 20]")
        if self.burn_in < 0:
            raise ValueError("burn_in must be nonnegative")
        if self.samples <= 0:
            raise ValueError("samples must be positive")
        if self.thinning <= 0:
            raise ValueError("thinning must be positive")

    @property
    def chain_length(self) -> int:
        return self.burn_in + self.samples * self.thinning


@dataclass(frozen=True)
class MCMCResult:
    """Samples and diagnostics from one completed blocked-Gibbs chain."""

    config: MCMCConfig
    rho: float
    sampler: str
    chain_length: int
    initial_assignment: BoolVector
    final_assignment: BoolVector
    retained_samples: BoolMatrix
    marginals: FloatVector
    marginal_standard_errors: FloatVector
    variable_autocorrelation_times: FloatVector
    variable_effective_sample_sizes: FloatVector
    mine_density_trace: FloatVector
    mine_density_autocorrelation_time: float
    mine_density_effective_sample_size: float
    planted_overlap_trace: FloatVector | None
    planted_overlap_autocorrelation_time: float | None
    planted_overlap_effective_sample_size: float | None
    changed_update_rate: float
    acceptance_rate: None = None

    def __post_init__(self) -> None:
        samples = np.asarray(self.retained_samples, dtype=np.bool_)
        if samples.shape != (self.config.samples, np.asarray(self.initial_assignment).size):
            raise ValueError(
                "retained_samples shape must equal (config.samples, number_of_variables)"
            )
        number_of_variables = samples.shape[1]
        variable_vectors = (
            "initial_assignment",
            "final_assignment",
            "marginals",
            "marginal_standard_errors",
            "variable_autocorrelation_times",
            "variable_effective_sample_sizes",
        )
        for name in variable_vectors:
            values = np.asarray(getattr(self, name))
            if values.shape != (number_of_variables,):
                raise ValueError(f"{name} must have one entry per variable")
            object.__setattr__(self, name, _frozen_copy(values))
        density = np.asarray(self.mine_density_trace, dtype=np.float64)
        if density.shape != (self.config.samples,):
            raise ValueError("mine_density_trace must have one entry per sample")
        object.__setattr__(self, "mine_density_trace", _frozen_copy(density))
        if self.chain_length != self.config.chain_length:
            raise ValueError("chain_length does not match config")
        if not 0.0 <= self.changed_update_rate <= 1.0:
            raise ValueError("changed_update_rate must lie in [0, 1]")
        if samples.ndim != 2:
            raise ValueError("retained_samples must be a matrix")
        object.__setattr__(self, "retained_samples", _frozen_copy(samples))
        if self.planted_overlap_trace is not None:
            overlap = np.asarray(self.planted_overlap_trace, dtype=np.float64)
            if overlap.shape != (self.config.samples,):
                raise ValueError("planted_overlap_trace must have one entry per sample")
            object.__setattr__(
                self,
                "planted_overlap_trace",
                _frozen_copy(overlap),
            )


@dataclass(frozen=True)
class MCMCMultipleResult:
    """Independent-chain comparison and replica-overlap diagnostics."""

    chains: tuple[MCMCResult, ...]
    pooled_marginals: FloatVector
    r_hat: FloatVector
    replica_overlap_trace: FloatVector

    def __post_init__(self) -> None:
        if len(self.chains) < 2:
            raise ValueError("at least two chains are required")
        number_of_variables = self.chains[0].marginals.size
        rho = self.chains[0].rho
        if any(chain.marginals.size != number_of_variables for chain in self.chains):
            raise ValueError("all chains must have the same number of variables")
        if any(chain.rho != rho for chain in self.chains):
            raise ValueError("all chains must use the same rho")
        if np.asarray(self.pooled_marginals).shape != (number_of_variables,):
            raise ValueError("pooled_marginals must have one entry per variable")
        if np.asarray(self.r_hat).shape != (number_of_variables,):
            raise ValueError("r_hat must have one entry per variable")
        if np.asarray(self.replica_overlap_trace).ndim != 1:
            raise ValueError("replica_overlap_trace must be one-dimensional")
        object.__setattr__(
            self, "pooled_marginals", _frozen_copy(np.asarray(self.pooled_marginals))
        )
        object.__setattr__(self, "r_hat", _frozen_copy(np.asarray(self.r_hat)))
        object.__setattr__(
            self,
            "replica_overlap_trace",
            _frozen_copy(np.asarray(self.replica_overlap_trace)),
        )

    @property
    def max_pairwise_marginal_difference(self) -> float:
        maximum = 0.0
        for left_index, left in enumerate(self.chains[:-1]):
            for right in self.chains[left_index + 1 :]:
                if left.marginals.size:
                    maximum = max(
                        maximum,
                        float(np.max(np.abs(left.marginals - right.marginals))),
                    )
        return maximum

    @property
    def replica_overlap_mean(self) -> float:
        return float(self.replica_overlap_trace.mean())


@dataclass(frozen=True)
class MCMCMarginalComparison:
    """Errors between empirical MCMC and exact posterior marginals."""

    mae: float
    rmse: float
    max_absolute_error: float


def _coerce_assignment(graph: FactorGraph, assignment: ArrayLike, *, name: str) -> BoolVector:
    values = np.asarray(assignment)
    if values.shape == graph.shape:
        if not np.all((values == 0) | (values == 1)):
            raise ValueError(f"{name} grid must be binary")
        if any(bool(values[cell]) for cell in graph.clue_cells):
            raise ValueError(f"{name} has a mine on an observed clue")
        result = np.fromiter(
            (bool(values[cell]) for cell in graph.variable_cells),
            dtype=np.bool_,
            count=graph.number_of_variables,
        )
    elif values.shape == (graph.number_of_variables,):
        if not np.all((values == 0) | (values == 1)):
            raise ValueError(f"{name} must be binary")
        result = values.astype(np.bool_, copy=True)
    else:
        raise ValueError(
            f"{name} must have shape {graph.shape} or ({graph.number_of_variables},)"
        )
    if np.any(constraint_residuals(graph, result) != 0):
        raise ValueError(f"{name} does not satisfy the CSP constraints")
    return result


def _variable_adjacency(graph: FactorGraph) -> tuple[tuple[int, ...], ...]:
    neighbors: list[set[int]] = [set() for _ in range(graph.number_of_variables)]
    for variables in graph.factor_to_variables:
        for variable in variables:
            neighbors[variable].update(other for other in variables if other != variable)
    return tuple(tuple(sorted(values)) for values in neighbors)


def _choose_local_block(
    number_of_variables: int,
    adjacency: tuple[tuple[int, ...], ...],
    block_size: int,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    seed_variable = int(rng.integers(number_of_variables))
    block: list[int] = []
    queued = {seed_variable}
    queue = [seed_variable]
    while queue and len(block) < block_size:
        variable = queue.pop(0)
        block.append(variable)
        candidates = [neighbor for neighbor in adjacency[variable] if neighbor not in queued]
        if candidates:
            for neighbor in rng.permutation(candidates):
                neighbor = int(neighbor)
                queued.add(neighbor)
                queue.append(neighbor)
    return tuple(block)


def _binary_states(number_of_variables: int) -> BoolMatrix:
    codes = np.arange(1 << number_of_variables, dtype=np.uint64)
    shifts = np.arange(number_of_variables, dtype=np.uint64)
    return ((codes[:, None] >> shifts[None, :]) & np.uint64(1)).astype(np.bool_)


def _sample_block_conditional(
    graph: FactorGraph,
    state: BoolVector,
    block: tuple[int, ...],
    rng: np.random.Generator,
) -> BoolVector:
    candidates = _binary_states(len(block))
    feasible = np.ones(candidates.shape[0], dtype=np.bool_)
    block_position = {variable: position for position, variable in enumerate(block)}
    affected_factors = sorted(
        {factor for variable in block for factor in graph.variable_to_factors[variable]}
    )
    for factor in affected_factors:
        inside_positions: list[int] = []
        outside_total = 0
        for variable in graph.factor_to_variables[factor]:
            position = block_position.get(variable)
            if position is None:
                outside_total += int(state[variable])
            else:
                inside_positions.append(position)
        totals = outside_total + candidates[:, inside_positions].sum(axis=1)
        feasible &= totals == graph.clue_values[factor]
        if not feasible.any():
            break

    feasible_candidates = candidates[feasible]
    if feasible_candidates.shape[0] == 0:
        raise RuntimeError("blocked conditional has no feasible assignment")
    selected = int(rng.integers(feasible_candidates.shape[0]))
    return feasible_candidates[selected]


def _overlap_trace(samples: BoolMatrix, planted: BoolVector) -> FloatVector:
    if samples.shape[1] == 0:
        return np.ones(samples.shape[0], dtype=np.float64)
    distances = np.count_nonzero(samples != planted[None, :], axis=1)
    return 1.0 - 2.0 * distances / samples.shape[1]


def run_blocked_gibbs(
    graph: FactorGraph,
    rho: float,
    initial_assignment: ArrayLike,
    *,
    config: MCMCConfig | None = None,
    planted_ground_truth: NDArray[np.bool_] | None = None,
) -> MCMCResult:
    """Run blocked Gibbs for the uniform measure on feasible assignments.

    ``rho`` is planted-instance metadata and does not enter transition weights.
    """

    if isinstance(rho, bool) or not isinstance(rho, (int, float, np.number)):
        raise TypeError("rho must be a real number")
    if not np.isfinite(rho) or not 0.0 <= float(rho) <= 1.0:
        raise ValueError("rho must lie in [0, 1]")
    rho = float(rho)
    config = config if config is not None else MCMCConfig()
    if not isinstance(config, MCMCConfig):
        raise TypeError("config must be an MCMCConfig")

    state = _coerce_assignment(graph, initial_assignment, name="initial_assignment")
    initial = state.copy()

    planted = None
    if planted_ground_truth is not None:
        truth = _as_bool_grid(planted_ground_truth, name="planted_ground_truth")
        assert_csp_consistent(graph, truth)
        planted = _coerce_assignment(graph, truth, name="planted_ground_truth")

    rng = np.random.default_rng(config.seed)
    retained = np.empty((config.samples, graph.number_of_variables), dtype=np.bool_)
    adjacency = _variable_adjacency(graph)
    retained_index = 0
    changed_updates = 0

    if graph.number_of_variables == 0:
        retained[:] = state
    else:
        for step in range(1, config.chain_length + 1):
            block = _choose_local_block(
                graph.number_of_variables, adjacency, config.block_size, rng
            )
            updated = _sample_block_conditional(graph, state, block, rng)
            if np.any(state[list(block)] != updated):
                changed_updates += 1
            state[list(block)] = updated
            if step > config.burn_in and (step - config.burn_in) % config.thinning == 0:
                retained[retained_index] = state
                retained_index += 1

    marginals = retained.mean(axis=0, dtype=np.float64)
    variable_taus = np.array(
        [
            integrated_autocorrelation_time(retained[:, variable])
            for variable in range(graph.number_of_variables)
        ],
        dtype=np.float64,
    )
    variable_ess = np.array(
        [
            effective_sample_size(retained[:, variable])
            for variable in range(graph.number_of_variables)
        ],
        dtype=np.float64,
    )
    standard_errors = np.full_like(marginals, np.nan)
    valid_ess = np.isfinite(variable_ess) & (variable_ess > 0.0)
    standard_errors[valid_ess] = np.sqrt(
        marginals[valid_ess]
        * (1.0 - marginals[valid_ess])
        / variable_ess[valid_ess]
    )
    density_trace = (
        retained.mean(axis=1, dtype=np.float64)
        if graph.number_of_variables
        else np.zeros(config.samples, dtype=np.float64)
    )
    density_tau = integrated_autocorrelation_time(density_trace)
    density_ess = effective_sample_size(density_trace)

    planted_trace = _overlap_trace(retained, planted) if planted is not None else None
    planted_tau = (
        integrated_autocorrelation_time(planted_trace)
        if planted_trace is not None
        else None
    )
    planted_ess = (
        effective_sample_size(planted_trace) if planted_trace is not None else None
    )

    return MCMCResult(
        config=config,
        rho=rho,
        sampler="local_bfs_blocked_gibbs",
        chain_length=config.chain_length,
        initial_assignment=initial,
        final_assignment=state,
        retained_samples=retained,
        marginals=marginals,
        marginal_standard_errors=standard_errors,
        variable_autocorrelation_times=variable_taus,
        variable_effective_sample_sizes=variable_ess,
        mine_density_trace=density_trace,
        mine_density_autocorrelation_time=density_tau,
        mine_density_effective_sample_size=density_ess,
        planted_overlap_trace=planted_trace,
        planted_overlap_autocorrelation_time=planted_tau,
        planted_overlap_effective_sample_size=planted_ess,
        changed_update_rate=(changed_updates / config.chain_length),
    )


def run_blocked_gibbs_chains(
    graph: FactorGraph,
    rho: float,
    initial_assignments: Sequence[ArrayLike],
    configs: Sequence[MCMCConfig],
    *,
    planted_ground_truth: NDArray[np.bool_] | None = None,
) -> MCMCMultipleResult:
    """Run independent chains and compute chain and replica diagnostics."""

    initial_assignments = tuple(initial_assignments)
    configs = tuple(configs)
    if len(initial_assignments) != len(configs):
        raise ValueError("one initial assignment is required per MCMCConfig")
    if len(configs) < 2:
        raise ValueError("at least two chains are required")
    chains = tuple(
        run_blocked_gibbs(
            graph,
            rho,
            initial,
            config=config,
            planted_ground_truth=planted_ground_truth,
        )
        for initial, config in zip(initial_assignments, configs)
    )
    pooled = np.concatenate([chain.retained_samples for chain in chains], axis=0)
    pooled_marginals = pooled.mean(axis=0, dtype=np.float64)

    common_samples = min(chain.retained_samples.shape[0] for chain in chains)
    stacked = np.stack(
        [chain.retained_samples[:common_samples] for chain in chains], axis=0
    )
    r_hat = gelman_rubin_r_hat(stacked)

    replica_traces: list[FloatVector] = []
    for left_index, left in enumerate(chains[:-1]):
        for right in chains[left_index + 1 :]:
            sample_count = min(
                left.retained_samples.shape[0], right.retained_samples.shape[0]
            )
            if graph.number_of_variables == 0:
                overlaps = np.ones(sample_count, dtype=np.float64)
            else:
                distances = np.count_nonzero(
                    left.retained_samples[:sample_count]
                    != right.retained_samples[:sample_count],
                    axis=1,
                )
                overlaps = 1.0 - 2.0 * distances / graph.number_of_variables
            replica_traces.append(overlaps)

    return MCMCMultipleResult(
        chains=chains,
        pooled_marginals=pooled_marginals,
        r_hat=r_hat,
        replica_overlap_trace=np.concatenate(replica_traces),
    )


def compare_mcmc_to_exact(
    mcmc: MCMCResult | MCMCMultipleResult,
    exact: ExactResult,
) -> MCMCMarginalComparison:
    """Compare empirical MCMC marginals with the normalized exact posterior."""

    if not exact.has_posterior_mass:
        raise ValueError("exact posterior has zero mass")
    if isinstance(mcmc, MCMCResult):
        marginals = mcmc.marginals
    elif isinstance(mcmc, MCMCMultipleResult):
        marginals = mcmc.pooled_marginals
    else:
        raise TypeError("mcmc must be an MCMCResult or MCMCMultipleResult")
    if marginals.shape != exact.marginals.shape:
        raise ValueError("MCMC and exact results have different numbers of variables")
    differences = marginals - exact.marginals
    if differences.size == 0:
        return MCMCMarginalComparison(0.0, 0.0, 0.0)
    return MCMCMarginalComparison(
        mae=float(np.mean(np.abs(differences))),
        rmse=float(np.sqrt(np.mean(differences * differences))),
        max_absolute_error=float(np.max(np.abs(differences))),
    )
