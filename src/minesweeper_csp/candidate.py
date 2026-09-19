"""Focused finite-size and overlap analysis for possible RSB regions."""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .aggregate import aggregate_directory, summarize_samples
from .experiment import SweepExperimentConfig, derive_seed, execute_sweep
from .io import read_json, utc_timestamp, write_json_atomic


SCORE_COMPONENTS: dict[str, tuple[str, bool]] = {
    "bp_mcmc_mae": ("bp_mcmc_mae", True),
    "bp_nonconvergence": ("bp_convergence_fraction", False),
    "bp_fixed_point_spread": ("bp_fixed_point_spread", True),
    "mcmc_chain_spread": ("mcmc_chain_marginal_spread", True),
    "mcmc_r_hat": ("mcmc_max_r_hat", True),
    "mcmc_overlap_autocorrelation": ("mcmc_overlap_tau", True),
    "mcmc_low_ess": ("mcmc_min_variable_ess", False),
}


def _metric_mean(point: Mapping[str, Any], metric: str) -> float | None:
    value = point.get("metrics", {}).get(metric, {}).get("mean")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def score_candidate_points(aggregate: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Rank rho points by several normalized warning diagnostics.

    Scores are relative to the supplied scan and are prioritization aids, not
    probabilities or evidence that replica symmetry is broken.
    """

    points = list(aggregate.get("parameter_points", []))
    if not points:
        raise ValueError("aggregate contains no parameter points")

    normalized: dict[str, dict[int, float]] = {}
    raw_components: dict[str, dict[int, float | None]] = {}
    for component, (metric, high_is_suspicious) in SCORE_COMPONENTS.items():
        raw = {index: _metric_mean(point, metric) for index, point in enumerate(points)}
        raw_components[component] = raw
        finite = [value for value in raw.values() if value is not None]
        if not finite:
            normalized[component] = {}
            continue
        low, high = min(finite), max(finite)
        component_scores: dict[int, float] = {}
        for index, value in raw.items():
            if value is None:
                continue
            if high == low:
                score = 0.0
            else:
                score = (value - low) / (high - low)
                if not high_is_suspicious:
                    score = 1.0 - score
            component_scores[index] = float(score)
        normalized[component] = component_scores

    scored = []
    for index, point in enumerate(points):
        available = {
            component: values[index]
            for component, values in normalized.items()
            if index in values
        }
        score = float(np.mean(list(available.values()))) if available else 0.0
        scored.append(
            {
                "rho": float(point["rho"]),
                "score": score,
                "available_components": len(available),
                "components": available,
                "raw": {
                    component: raw_components[component][index]
                    for component in SCORE_COMPONENTS
                },
            }
        )
    return sorted(scored, key=lambda item: (-item["score"], item["rho"]))


def identify_candidate_region(
    aggregate: Mapping[str, Any],
    *,
    top_k: int = 3,
    padding_points: int = 1,
) -> dict[str, Any]:
    """Select and pad the highest-priority rho points from a coarse scan."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if padding_points < 0:
        raise ValueError("padding_points must be nonnegative")
    rhos = sorted(float(point["rho"]) for point in aggregate["parameter_points"])
    ranking = score_candidate_points(aggregate)
    chosen = ranking[: min(top_k, len(ranking))]
    chosen_indices = [rhos.index(item["rho"]) for item in chosen]
    low_index = max(0, min(chosen_indices) - padding_points)
    high_index = min(len(rhos) - 1, max(chosen_indices) + padding_points)
    return {
        "rho_min": rhos[low_index],
        "rho_max": rhos[high_index],
        "selected_rhos": sorted(item["rho"] for item in chosen),
        "top_k": top_k,
        "padding_points": padding_points,
        "ranking": ranking,
        "interpretation": (
            "prioritization score only; finite-size, loopy-BP, and MCMC mixing "
            "effects must be separated before any RSB claim"
        ),
    }


def _parse_size(size: Sequence[int]) -> tuple[int, int]:
    if len(size) != 2:
        raise ValueError("each size must contain Lx and Ly")
    Lx, Ly = int(size[0]), int(size[1])
    if Lx <= 0 or Ly <= 0:
        raise ValueError("system sizes must be positive")
    return Lx, Ly


def build_candidate_plan(
    aggregate: Mapping[str, Any],
    *,
    sizes: Sequence[Sequence[int]],
    dense_points: int = 11,
    top_k: int = 3,
    padding_points: int = 1,
    disorder_samples: int | None = None,
    bp_random_restarts: int = 6,
    mcmc_chains: int | None = None,
    base_seed: int | None = None,
    overlap_bins: int = 41,
) -> dict[str, Any]:
    """Build fully specified size-dependent sweep configs from a coarse scan."""

    if "config" not in aggregate:
        raise ValueError("aggregate must include its source config")
    parsed_sizes = tuple(_parse_size(size) for size in sizes)
    if not parsed_sizes:
        raise ValueError("at least one system size is required")
    if dense_points < 2:
        raise ValueError("dense_points must be at least two")
    if bp_random_restarts < 0:
        raise ValueError("bp_random_restarts must be nonnegative")
    if overlap_bins < 3:
        raise ValueError("overlap_bins must be at least three")

    selection = identify_candidate_region(
        aggregate, top_k=top_k, padding_points=padding_points
    )
    if selection["rho_min"] == selection["rho_max"]:
        coarse_rhos = sorted(
            float(point["rho"]) for point in aggregate["parameter_points"]
        )
        if len(coarse_rhos) < 2:
            raise ValueError("at least two coarse rho points are required for densification")
        index = coarse_rhos.index(selection["rho_min"])
        if index == 0:
            selection["rho_max"] = coarse_rhos[1]
        elif index == len(coarse_rhos) - 1:
            selection["rho_min"] = coarse_rhos[-2]
        else:
            selection["rho_min"] = coarse_rhos[index - 1]
            selection["rho_max"] = coarse_rhos[index + 1]
        selection["auto_expanded_single_point"] = True
    rho_values = np.linspace(
        selection["rho_min"], selection["rho_max"], dense_points
    ).tolist()
    source_config = aggregate["config"]
    source_sweep = source_config["sweep"]
    selected_disorder_samples = (
        int(disorder_samples)
        if disorder_samples is not None
        else int(source_sweep["disorder_samples"])
    )
    selected_chains = (
        int(mcmc_chains)
        if mcmc_chains is not None
        else int(source_config["mcmc"]["chains"])
    )
    selected_seed = int(base_seed) if base_seed is not None else int(source_sweep["base_seed"])
    if selected_disorder_samples <= 0:
        raise ValueError("disorder_samples must be positive")
    if selected_chains < 2:
        raise ValueError("mcmc_chains must be at least two")

    initializations = ["uniform"] + [
        "random" for _ in range(bp_random_restarts)
    ]
    sweeps = []
    for Lx, Ly in parsed_sizes:
        size_seed = derive_seed(selected_seed, Lx, Ly, "candidate_size")
        mapping = {
            "model": {"Lx": Lx, "Ly": Ly},
            "observation": dict(source_config["observation"]),
            "sweep": {
                "rho_values": rho_values,
                "disorder_samples": selected_disorder_samples,
                "base_seed": size_seed,
            },
            "exact": dict(source_config["exact"]),
            "bp": {
                **source_config["bp"],
                "initializations": initializations,
            },
            "mcmc": {
                **source_config["mcmc"],
                "chains": selected_chains,
            },
        }
        validated = SweepExperimentConfig.from_mapping(mapping)
        sweeps.append(
            {
                "name": f"Lx{Lx:03d}_Ly{Ly:03d}",
                "Lx": Lx,
                "Ly": Ly,
                "config": validated.to_mapping(),
            }
        )
    return {
        "schema_version": 1,
        "source_run_count": int(aggregate.get("run_count", 0)),
        "source_generated_at_utc": aggregate.get("generated_at_utc"),
        "selection": selection,
        "rho_values": rho_values,
        "sizes": [list(size) for size in parsed_sizes],
        "overlap_bins": overlap_bins,
        "sweeps": sweeps,
    }


def write_candidate_plan(path: str | Path, plan: Mapping[str, Any]) -> None:
    write_json_atomic(path, plan)


def execute_candidate_plan(
    plan: Mapping[str, Any],
    output_directory: str | Path,
    *,
    overwrite: bool = False,
    repository_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Execute every size sweep in a candidate plan and analyze the results."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "candidate_plan.json"
    if plan_path.exists() and read_json(plan_path) != dict(plan):
        raise ValueError("output directory contains a different candidate plan")
    write_json_atomic(plan_path, plan)
    for sweep in plan["sweeps"]:
        config = SweepExperimentConfig.from_mapping(sweep["config"])
        size_output = output / sweep["name"]
        execute_sweep(
            config,
            size_output,
            overwrite=overwrite,
            repository_directory=repository_directory,
        )
        aggregate_directory(size_output)
    return analyze_candidate_directory(output, overlap_bins=int(plan["overlap_bins"]))


def _weighted_distribution_statistics(
    values: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    total = float(probabilities.sum())
    if values.size == 0 or total <= 0.0:
        return {
            "mean": None,
            "variance": None,
            "q05": None,
            "q50": None,
            "q95": None,
            "width_q05_q95": None,
            "entropy": None,
            "mode_count": None,
        }
    probabilities = probabilities / total
    order = np.argsort(values)
    values = values[order]
    probabilities = probabilities[order]
    cumulative = np.cumsum(probabilities)

    def quantile(level: float) -> float:
        return float(values[min(int(np.searchsorted(cumulative, level)), values.size - 1)])

    mean = float(values @ probabilities)
    variance = float(((values - mean) ** 2) @ probabilities)
    positive = probabilities > 0.0
    entropy = float(-(probabilities[positive] * np.log(probabilities[positive])).sum())
    smoothed = np.convolve(probabilities, [0.25, 0.5, 0.25], mode="same")
    threshold = 0.05 * float(smoothed.max())
    modes = 0
    for index, mass in enumerate(smoothed):
        left = smoothed[index - 1] if index > 0 else -math.inf
        right = smoothed[index + 1] if index + 1 < smoothed.size else -math.inf
        if mass >= threshold and mass > left and mass >= right:
            modes += 1
    q05, q50, q95 = quantile(0.05), quantile(0.5), quantile(0.95)
    return {
        "mean": mean,
        "variance": variance,
        "q05": q05,
        "q50": q50,
        "q95": q95,
        "width_q05_q95": q95 - q05,
        "entropy": entropy,
            "mode_count": modes,
            "mode_count_is_histogram_heuristic": True,
    }


def _histogram(
    values: np.ndarray,
    edges: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    masses = np.histogram(values, bins=edges, weights=weights)[0].astype(np.float64)
    if masses.sum() > 0.0:
        masses /= masses.sum()
    return masses


def _analyze_point(
    run_paths: Sequence[Path],
    aggregate_point: Mapping[str, Any],
    edges: np.ndarray,
) -> dict[str, Any]:
    mcmc_histograms = []
    exact_histograms = []
    mcmc_run_widths: list[float | None] = []
    exact_run_widths: list[float | None] = []
    distinct_fixed_points = []
    changed_rates = []
    for path in run_paths:
        record = read_json(path)
        distinct_fixed_points.append(len(record["bp"]["fixed_point_groups"]))
        changed_rates.extend(
            float(chain["changed_update_rate"]) for chain in record["mcmc"]["chains"]
        )
        with np.load(path.parent / record["files"]["arrays"]) as arrays:
            mcmc_values = arrays["mcmc_replica_overlap_trace"]
            mcmc_histogram = _histogram(mcmc_values, edges)
            mcmc_histograms.append(mcmc_histogram)
            mcmc_statistics = _weighted_distribution_statistics(
                (edges[:-1] + edges[1:]) / 2.0, mcmc_histogram
            )
            mcmc_run_widths.append(mcmc_statistics["width_q05_q95"])
            if "exact_replica_overlap_values" in arrays:
                exact_histogram = _histogram(
                    arrays["exact_replica_overlap_values"],
                    edges,
                    arrays["exact_replica_overlap_probabilities"],
                )
                exact_histograms.append(exact_histogram)
                exact_statistics = _weighted_distribution_statistics(
                    (edges[:-1] + edges[1:]) / 2.0, exact_histogram
                )
                exact_run_widths.append(exact_statistics["width_q05_q95"])

    centers = (edges[:-1] + edges[1:]) / 2.0
    mcmc_mean_histogram = np.mean(mcmc_histograms, axis=0)
    exact_mean_histogram = (
        np.mean(exact_histograms, axis=0) if exact_histograms else None
    )
    metrics = aggregate_point["metrics"]
    return {
        "rho": float(aggregate_point["rho"]),
        "disorder_samples": len(run_paths),
        "bp_mcmc_mae": metrics["bp_mcmc_mae"],
        "bp_convergence_fraction": metrics["bp_convergence_fraction"],
        "bp_fixed_point_spread": metrics["bp_fixed_point_spread"],
        "bp_distinct_fixed_points": summarize_samples(distinct_fixed_points),
        "mcmc_chain_marginal_spread": metrics["mcmc_chain_marginal_spread"],
        "mcmc_max_r_hat": metrics["mcmc_max_r_hat"],
        "mcmc_density_tau": metrics["mcmc_density_tau"],
        "mcmc_overlap_tau": metrics["mcmc_overlap_tau"],
        "mcmc_min_variable_ess": metrics["mcmc_min_variable_ess"],
        "mcmc_changed_update_rate": summarize_samples(changed_rates),
        "mcmc_overlap": {
            "bin_centers": centers.tolist(),
            "probabilities": mcmc_mean_histogram.tolist(),
            "shape": _weighted_distribution_statistics(centers, mcmc_mean_histogram),
            "width_across_disorder": summarize_samples(mcmc_run_widths),
        },
        "exact_overlap": (
            {
                "bin_centers": centers.tolist(),
                "probabilities": exact_mean_histogram.tolist(),
                "shape": _weighted_distribution_statistics(
                    centers, exact_mean_histogram
                ),
                "width_across_disorder": summarize_samples(exact_run_widths),
            }
            if exact_mean_histogram is not None
            else None
        ),
    }


def analyze_candidate_directory(
    output_directory: str | Path,
    *,
    overlap_bins: int | None = None,
) -> dict[str, Any]:
    """Aggregate finite-size diagnostics and full P(q) from a completed study."""

    output = Path(output_directory)
    plan = read_json(output / "candidate_plan.json")
    bins = int(overlap_bins if overlap_bins is not None else plan["overlap_bins"])
    if bins < 3:
        raise ValueError("overlap_bins must be at least three")
    edges = np.linspace(-1.0, 1.0, bins + 1)
    sizes = []
    for sweep in plan["sweeps"]:
        size_output = output / sweep["name"]
        aggregate = aggregate_directory(size_output)
        run_paths_by_rho: dict[float, list[Path]] = defaultdict(list)
        for path in sorted(size_output.glob("rho_*/sample_*/run.json")):
            record = read_json(path)
            run_paths_by_rho[float(record["model"]["rho"])].append(path)
        aggregate_by_rho = {
            float(point["rho"]): point for point in aggregate["parameter_points"]
        }
        points = [
            _analyze_point(run_paths_by_rho[rho], aggregate_by_rho[rho], edges)
            for rho in sorted(run_paths_by_rho)
        ]
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
        "selection": plan["selection"],
        "overlap_bins": bins,
        "sizes": sizes,
        "interpretation": (
            "These are finite-size candidate diagnostics. Broad or multimodal P(q), "
            "BP multiplicity, and slow MCMC must co-occur and survive size/mixing "
            "checks before motivating an RSB interpretation."
        ),
    }
    write_json_atomic(output / "candidate_analysis.json", analysis)
    from .candidate_visualization import write_candidate_figures

    write_candidate_figures(analysis, output)
    return read_json(output / "candidate_analysis.json")
