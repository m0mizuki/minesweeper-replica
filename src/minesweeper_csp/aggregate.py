"""Disorder aggregation for saved phase-scan runs."""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .io import read_json, utc_timestamp, write_json_atomic


METRIC_PATHS: dict[str, tuple[str, ...]] = {
    "bp_mcmc_mae": ("comparison", "bp_vs_mcmc", "mae"),
    "bp_exact_mae": ("comparison", "bp_vs_exact", "mae"),
    "mcmc_exact_mae": ("comparison", "mcmc_vs_exact", "mae"),
    "bp_convergence_fraction": ("bp", "convergence_fraction"),
    "bp_max_iterations": ("bp", "max_iterations"),
    "bp_fixed_point_spread": ("bp", "max_pairwise_marginal_difference"),
    "mcmc_chain_marginal_spread": (
        "mcmc",
        "max_pairwise_marginal_difference",
    ),
    "mcmc_max_r_hat": ("mcmc", "max_finite_r_hat"),
    "mcmc_density_tau": ("mcmc", "mean_density_autocorrelation_time"),
    "mcmc_min_variable_ess": ("mcmc", "minimum_variable_effective_sample_size"),
    "mcmc_planted_overlap": ("overlap", "mcmc_planted_mean"),
    "mcmc_replica_overlap": ("overlap", "mcmc_replica_mean"),
    "exact_planted_overlap": ("overlap", "exact_planted_mean"),
    "exact_replica_overlap": ("overlap", "exact_replica_mean"),
}


def _nested_value(record: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = record
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def summarize_samples(values: Iterable[float | None]) -> dict[str, Any]:
    raw = list(values)
    finite = np.array([value for value in raw if value is not None], dtype=np.float64)
    if finite.size == 0:
        return {
            "count": 0,
            "missing": len(raw),
            "mean": None,
            "standard_deviation": None,
            "standard_error": None,
            "quantiles": None,
        }
    standard_deviation = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
    quantiles = np.quantile(finite, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "count": int(finite.size),
        "missing": len(raw) - int(finite.size),
        "mean": float(finite.mean()),
        "standard_deviation": standard_deviation,
        "standard_error": standard_deviation / math.sqrt(finite.size),
        "quantiles": {
            "q05": float(quantiles[0]),
            "q25": float(quantiles[1]),
            "q50": float(quantiles[2]),
            "q75": float(quantiles[3]),
            "q95": float(quantiles[4]),
        },
    }


def aggregate_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    all_records = list(records)
    for record in all_records:
        grouped[float(record["model"]["rho"])].append(record)

    parameter_points = []
    for rho in sorted(grouped):
        group = grouped[rho]
        parameter_points.append(
            {
                "rho": rho,
                "disorder_samples": len(group),
                "metrics": {
                    name: summarize_samples(
                        _nested_value(record, path) for record in group
                    )
                    for name, path in METRIC_PATHS.items()
                },
                "bp_status_counts": {
                    status: sum(
                        run["status"] == status
                        for record in group
                        for run in record["bp"]["runs"]
                    )
                    for status in ("converged", "max_iterations", "infeasible")
                },
                "mcmc_infinite_r_hat_variables": sum(
                    int(record["mcmc"]["infinite_r_hat_variables"])
                    for record in group
                ),
            }
        )
    return {
        "schema_version": 1,
        "run_count": len(all_records),
        "generated_at_utc": utc_timestamp(),
        "parameter_points": parameter_points,
    }


def aggregate_directory(output_directory: str | Path) -> dict[str, Any]:
    output = Path(output_directory)
    paths = sorted(output.glob("rho_*/sample_*/run.json"))
    if not paths:
        raise ValueError(f"no run.json files found under {output}")
    aggregate = aggregate_records(read_json(path) for path in paths)
    config_path = output / "config.json"
    if config_path.exists():
        aggregate["config"] = read_json(config_path)
    write_json_atomic(output / "aggregate.json", aggregate)
    return aggregate

