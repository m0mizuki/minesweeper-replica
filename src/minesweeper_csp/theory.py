"""Finite-instance RS/Bethe diagnostics and local BP stability analysis."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from .aggregate import summarize_samples
from .bp import (
    BPConfig,
    BPResult,
    _build_edge_layout,
    _factor_message,
    _variable_message,
    run_bp,
)
from .factor_graph import FactorGraph, build_factor_graph
from .io import read_json, utc_timestamp, write_json_atomic

FloatMatrix = NDArray[np.float64]


def _frozen_copy(values: NDArray) -> NDArray:
    result = np.array(values, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class FactorGraphTopology:
    number_of_variables: int
    number_of_factors: int
    number_of_edges: int
    connected_components: int
    cycle_rank: int
    four_cycle_count: int

    @property
    def cycle_rank_per_variable(self) -> float | None:
        if self.number_of_variables == 0:
            return None
        return self.cycle_rank / self.number_of_variables

    @property
    def four_cycles_per_variable(self) -> float | None:
        if self.number_of_variables == 0:
            return None
        return self.four_cycle_count / self.number_of_variables


@dataclass(frozen=True)
class BetheResult:
    log_partition_function: float
    free_entropy_density: float | None
    entropy: float
    entropy_density: float | None
    expected_mine_count: float
    variable_log_normalizers: tuple[float, ...]
    factor_log_normalizers: tuple[float, ...]
    edge_log_normalizers: tuple[float, ...]


@dataclass(frozen=True)
class RSStabilityResult:
    spectral_radius: float
    locally_stable: bool
    perturbation: float
    clipping_probability: float
    clipped_message_entries: int
    total_message_entries: int
    jacobian: FloatMatrix

    def __post_init__(self) -> None:
        jacobian = np.asarray(self.jacobian, dtype=np.float64)
        if jacobian.ndim != 2 or jacobian.shape[0] != jacobian.shape[1]:
            raise ValueError("jacobian must be square")
        object.__setattr__(self, "jacobian", _frozen_copy(jacobian))

    @property
    def boundary_fraction(self) -> float:
        if self.total_message_entries == 0:
            return 0.0
        return self.clipped_message_entries / self.total_message_entries


class RSStabilityLimitError(ValueError):
    """Raised when numerical BP-Jacobian construction exceeds its edge guard."""


def factor_graph_topology(graph: FactorGraph) -> FactorGraphTopology:
    """Count independent cycles and exact length-four cycles."""

    variable_count = graph.number_of_variables
    factor_count = graph.number_of_constraints
    node_count = variable_count + factor_count
    parent = list(range(node_count))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    edge_count = 0
    for factor, variables in enumerate(graph.factor_to_variables):
        factor_node = variable_count + factor
        for variable in variables:
            edge_count += 1
            left, right = find(variable), find(factor_node)
            if left != right:
                parent[left] = right
    connected_components = len({find(node) for node in range(node_count)}) if node_count else 0
    cycle_rank = edge_count - node_count + connected_components

    four_cycles = 0
    factor_sets = [set(variables) for variables in graph.factor_to_variables]
    for left in range(factor_count - 1):
        for right in range(left + 1, factor_count):
            shared = len(factor_sets[left] & factor_sets[right])
            four_cycles += shared * (shared - 1) // 2
    return FactorGraphTopology(
        number_of_variables=variable_count,
        number_of_factors=factor_count,
        number_of_edges=edge_count,
        connected_components=connected_components,
        cycle_rank=cycle_rank,
        four_cycle_count=four_cycles,
    )


def _log_probability(value: float) -> float:
    return math.log(value) if value > 0.0 else -math.inf


def _logsumexp_pair(left: float, right: float) -> float:
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    maximum = max(left, right)
    return maximum + math.log(math.exp(left - maximum) + math.exp(right - maximum))


def _factor_log_normalizer(
    graph: FactorGraph,
    result: BPResult,
    factor: int,
) -> float:
    edges = [
        edge for edge, edge_factor in enumerate(result.edge_factors) if edge_factor == factor
    ]
    log_distribution = np.array([0.0], dtype=np.float64)
    for edge in edges:
        message = result.variable_to_factor_messages[edge]
        updated = np.full(log_distribution.size + 1, -math.inf, dtype=np.float64)
        for count, log_mass in enumerate(log_distribution):
            updated[count] = _logsumexp_pair(
                float(updated[count]), log_mass + _log_probability(float(message[0]))
            )
            updated[count + 1] = _logsumexp_pair(
                float(updated[count + 1]),
                log_mass + _log_probability(float(message[1])),
            )
        log_distribution = updated
    clue = graph.clue_values[factor]
    return float(log_distribution[clue]) if 0 <= clue < log_distribution.size else -math.inf


def compute_bethe_result(
    graph: FactorGraph,
    bp: BPResult,
    *,
    require_converged: bool = True,
) -> BetheResult:
    """Compute the Bethe log partition and entropy at a BP fixed point."""

    if require_converged and not bp.converged:
        raise ValueError("Bethe quantities require a converged BP result")
    expected_edges = tuple(
        (factor, variable)
        for factor, variables in enumerate(graph.factor_to_variables)
        for variable in variables
    )
    if expected_edges != tuple(zip(bp.edge_factors, bp.edge_variables)):
        raise ValueError("BP result does not match the supplied factor graph")

    variable_logs = []
    for variable in range(graph.number_of_variables):
        log_terms = np.zeros(2, dtype=np.float64)
        for edge, edge_variable in enumerate(bp.edge_variables):
            if edge_variable == variable:
                message = bp.factor_to_variable_messages[edge]
                log_terms[0] += _log_probability(float(message[0]))
                log_terms[1] += _log_probability(float(message[1]))
        variable_logs.append(_logsumexp_pair(float(log_terms[0]), float(log_terms[1])))

    factor_logs = [
        _factor_log_normalizer(graph, bp, factor)
        for factor in range(graph.number_of_constraints)
    ]
    edge_logs = []
    for edge in range(len(bp.edge_factors)):
        mass = float(
            bp.variable_to_factor_messages[edge]
            @ bp.factor_to_variable_messages[edge]
        )
        edge_logs.append(_log_probability(mass))
    if any(value == -math.inf for value in variable_logs + factor_logs + edge_logs):
        raise ValueError("Bethe normalizer has zero mass")

    log_partition = float(sum(variable_logs) + sum(factor_logs) - sum(edge_logs))
    entropy = log_partition
    variable_count = graph.number_of_variables
    return BetheResult(
        log_partition_function=log_partition,
        free_entropy_density=(log_partition / variable_count if variable_count else None),
        entropy=entropy,
        entropy_density=(entropy / variable_count if variable_count else None),
        expected_mine_count=float(bp.marginals.sum()),
        variable_log_normalizers=tuple(variable_logs),
        factor_log_normalizers=tuple(factor_logs),
        edge_log_normalizers=tuple(edge_logs),
    )


def _logit_messages(messages: FloatMatrix, clipping: float) -> tuple[np.ndarray, int]:
    probabilities = messages[:, 1]
    clipped = np.clip(probabilities, clipping, 1.0 - clipping)
    clipped_count = int(np.count_nonzero(clipped != probabilities))
    return np.log(clipped) - np.log1p(-clipped), clipped_count


def _sigmoid(log_odds: np.ndarray) -> FloatMatrix:
    probability_one = np.empty_like(log_odds)
    positive = log_odds >= 0.0
    probability_one[positive] = 1.0 / (1.0 + np.exp(-log_odds[positive]))
    exponential = np.exp(log_odds[~positive])
    probability_one[~positive] = exponential / (1.0 + exponential)
    return np.column_stack((1.0 - probability_one, probability_one))


def _undamped_bp_map(
    graph: FactorGraph,
    variable_log_odds: np.ndarray,
    clipping: float,
) -> np.ndarray:
    layout = _build_edge_layout(graph)
    variable_to_factor = _sigmoid(variable_log_odds)
    factor_to_variable = np.empty_like(variable_to_factor)
    for edge, factor in enumerate(layout.edge_factors):
        message = _factor_message(
            edge,
            factor,
            graph.clue_values[factor],
            layout,
            variable_to_factor,
        )
        if message is None:
            raise ValueError("perturbed BP map reached zero factor mass")
        factor_to_variable[edge] = message

    updated = np.empty_like(variable_to_factor)
    for edge, variable in enumerate(layout.edge_variables):
        message = _variable_message(
            edge,
            variable,
            layout,
            factor_to_variable,
        )
        if message is None:
            raise ValueError("perturbed BP map reached zero variable mass")
        updated[edge] = message
    log_odds, _ = _logit_messages(updated, clipping)
    return log_odds


def compute_rs_stability(
    graph: FactorGraph,
    bp: BPResult,
    *,
    perturbation: float = 1e-6,
    clipping_probability: float = 1e-12,
    max_edges: int = 256,
) -> RSStabilityResult:
    """Numerically linearize the undamped BP map in cavity log-odds.

    The spectral radius is a finite-instance local fixed-point diagnostic.  It
    is not an Almeida–Thouless eigenvalue for the quenched infinite ensemble.
    Boundary messages are regularized and their fraction is reported.
    """

    if not bp.converged:
        raise ValueError("RS stability requires a converged BP fixed point")
    if isinstance(max_edges, bool) or not isinstance(max_edges, (int, np.integer)):
        raise TypeError("max_edges must be an integer")
    if max_edges < 0:
        raise ValueError("max_edges must be nonnegative")
    edge_count = len(bp.edge_factors)
    if edge_count > max_edges:
        raise RSStabilityLimitError(
            f"BP Jacobian has {edge_count} edges, exceeding max_edges={max_edges}"
        )
    if not math.isfinite(perturbation) or perturbation <= 0.0:
        raise ValueError("perturbation must be finite and positive")
    if not 0.0 < clipping_probability < 0.5:
        raise ValueError("clipping_probability must lie in (0, 0.5)")
    base_log_odds, clipped_count = _logit_messages(
        bp.variable_to_factor_messages, clipping_probability
    )
    if edge_count == 0:
        jacobian = np.empty((0, 0), dtype=np.float64)
        spectral_radius = 0.0
    else:
        jacobian = np.empty((edge_count, edge_count), dtype=np.float64)
        for source in range(edge_count):
            plus = base_log_odds.copy()
            minus = base_log_odds.copy()
            plus[source] += perturbation
            minus[source] -= perturbation
            mapped_plus = _undamped_bp_map(graph, plus, clipping_probability)
            mapped_minus = _undamped_bp_map(graph, minus, clipping_probability)
            jacobian[:, source] = (mapped_plus - mapped_minus) / (
                2.0 * perturbation
            )
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(jacobian))))
    return RSStabilityResult(
        spectral_radius=spectral_radius,
        locally_stable=spectral_radius < 1.0,
        perturbation=perturbation,
        clipping_probability=clipping_probability,
        clipped_message_entries=clipped_count,
        total_message_entries=edge_count,
        jacobian=jacobian,
    )


def analyze_rs_run(
    run_path: str | Path,
    *,
    max_edges: int = 256,
) -> dict[str, Any]:
    """Reconstruct one saved run and add finite-instance RS diagnostics."""

    run_path = Path(run_path)
    record = read_json(run_path)
    primary_index = int(record["bp"]["primary_run"])
    primary = record["bp"]["runs"][primary_index]
    with np.load(run_path.parent / record["files"]["arrays"]) as arrays:
        graph = build_factor_graph(arrays["observation_mask"], arrays["clues"])
        saved_marginals = np.array(
            arrays[f"bp_{primary_index}_marginals"], dtype=np.float64, copy=True
        )
    config = BPConfig(
        max_iterations=max(
            1,
            int(primary["iterations"] + 5)
            if primary["converged"]
            else int(primary["iterations"]),
        ),
        tolerance=float(primary["tolerance"]),
        damping=float(primary["damping"]),
        initialization=str(primary["initialization"]),
        seed=primary["seed"],
    )
    bp = run_bp(graph, float(record["model"]["rho"]), config=config)
    replay_difference = None
    if np.all(np.isfinite(saved_marginals)) and np.all(np.isfinite(bp.marginals)):
        replay_difference = (
            float(np.max(np.abs(saved_marginals - bp.marginals)))
            if saved_marginals.size
            else 0.0
        )
    topology = factor_graph_topology(graph)
    result: dict[str, Any] = {
        "run_id": record["run_id"],
        "rho": float(record["model"]["rho"]),
        "number_of_variables": graph.number_of_variables,
        "saved_bp_status": primary["status"],
        "bp_status": bp.status,
        "bp_replay_max_marginal_difference": replay_difference,
        "topology": {
            **vars(topology),
            "cycle_rank_per_variable": topology.cycle_rank_per_variable,
            "four_cycles_per_variable": topology.four_cycles_per_variable,
        },
        "bethe": None,
        "stability": None,
        "exact_comparison": None,
    }
    if bp.converged:
        bethe = compute_bethe_result(graph, bp)
        result["bethe"] = vars(bethe)
        if record["exact"]["available"]:
            exact_log_partition = record["exact"]["log_partition_function"]
            difference = bethe.log_partition_function - exact_log_partition
            result["exact_comparison"] = {
                "exact_log_partition_function": exact_log_partition,
                "bethe_log_partition_error": difference,
                "absolute_error_per_variable": (
                    abs(difference) / graph.number_of_variables
                    if graph.number_of_variables
                    else None
                ),
            }
        try:
            stability = compute_rs_stability(graph, bp, max_edges=max_edges)
            result["stability"] = {
                "spectral_radius": stability.spectral_radius,
                "locally_stable": stability.locally_stable,
                "perturbation": stability.perturbation,
                "clipping_probability": stability.clipping_probability,
                "clipped_message_entries": stability.clipped_message_entries,
                "total_message_entries": stability.total_message_entries,
                "boundary_fraction": stability.boundary_fraction,
            }
        except RSStabilityLimitError as error:
            result["stability"] = {"available": False, "reason": str(error)}
    write_json_atomic(run_path.parent / "theory.json", result)
    return read_json(run_path.parent / "theory.json")


def analyze_rs_candidate_directory(
    output_directory: str | Path,
    *,
    max_edges: int = 256,
) -> dict[str, Any]:
    """Aggregate Bethe and local-stability diagnostics over a candidate study."""

    output = Path(output_directory)
    plan = read_json(output / "candidate_plan.json")
    sizes = []
    for sweep in plan["sweeps"]:
        size_output = output / sweep["name"]
        grouped: dict[float, list[dict[str, Any]]] = {}
        for run_path in sorted(size_output.glob("rho_*/sample_*/run.json")):
            diagnostic = analyze_rs_run(run_path, max_edges=max_edges)
            grouped.setdefault(float(diagnostic["rho"]), []).append(diagnostic)
        points = []
        for rho in sorted(grouped):
            diagnostics = grouped[rho]

            def values(path: tuple[str, ...]) -> Iterable[float | None]:
                for diagnostic in diagnostics:
                    value: Any = diagnostic
                    for key in path:
                        if not isinstance(value, dict):
                            value = None
                            break
                        value = value.get(key)
                    yield (
                        float(value)
                        if isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(float(value))
                        else None
                    )

            stability_values = list(values(("stability", "spectral_radius")))
            stable_flags = [
                diagnostic["stability"].get("locally_stable")
                for diagnostic in diagnostics
                if isinstance(diagnostic.get("stability"), dict)
                and isinstance(diagnostic["stability"].get("locally_stable"), bool)
            ]
            points.append(
                {
                    "rho": rho,
                    "disorder_samples": len(diagnostics),
                    "bp_converged_fraction": sum(
                        diagnostic["bp_status"] == "converged"
                        for diagnostic in diagnostics
                    )
                    / len(diagnostics),
                    "spectral_radius": summarize_samples(stability_values),
                    "locally_unstable_fraction": (
                        sum(not flag for flag in stable_flags) / len(stable_flags)
                        if stable_flags
                        else None
                    ),
                    "boundary_message_fraction": summarize_samples(
                        values(("stability", "boundary_fraction"))
                    ),
                    "bethe_log_partition_error_per_variable": summarize_samples(
                        values(("exact_comparison", "absolute_error_per_variable"))
                    ),
                    "bethe_free_entropy_density": summarize_samples(
                        values(("bethe", "free_entropy_density"))
                    ),
                    "bethe_entropy_density": summarize_samples(
                        values(("bethe", "entropy_density"))
                    ),
                    "cycle_rank_per_variable": summarize_samples(
                        values(("topology", "cycle_rank_per_variable"))
                    ),
                    "four_cycles_per_variable": summarize_samples(
                        values(("topology", "four_cycles_per_variable"))
                    ),
                }
            )
        sizes.append(
            {
                "name": sweep["name"],
                "Lx": int(sweep["Lx"]),
                "Ly": int(sweep["Ly"]),
                "N": int(sweep["Lx"]) * int(sweep["Ly"]),
                "points": points,
            }
        )
    analysis = {
        "schema_version": 1,
        "generated_at_utc": utc_timestamp(),
        "max_edges": max_edges,
        "sizes": sizes,
        "interpretation": (
            "The BP Jacobian spectral radius is a local finite-instance RS diagnostic, "
            "not an ensemble AT line. Bethe error can also arise from short spatial "
            "loops, so stability, exact error, topology, mixing, and P(q) must be read jointly."
        ),
    }
    write_json_atomic(output / "rs_theory_analysis.json", analysis)
    from .theory_visualization import write_rs_theory_svg

    write_rs_theory_svg(analysis, output / "rs_theory_summary.svg")
    return read_json(output / "rs_theory_analysis.json")
