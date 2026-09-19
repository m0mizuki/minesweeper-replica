"""Reproducible rho sweeps over planted disorder samples."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import platform
from typing import Any, Mapping

import numpy as np

from .bp import BPConfig, BPResult, compare_bp_to_exact, run_bp_multiple
from .exact import ExactEnumerationLimitError, ExactResult, solve_exact
from .factor_graph import build_factor_graph
from .instance import generate_ground_truth
from .io import (
    git_commit,
    git_is_dirty,
    read_json,
    utc_timestamp,
    write_json_atomic,
    write_npz_atomic,
)
from .mcmc import (
    MCMCConfig,
    MCMCMultipleResult,
    compare_mcmc_to_exact,
    run_blocked_gibbs_chains,
)
from .observation import generate_clues, generate_observation_mask


@dataclass(frozen=True)
class SweepExperimentConfig:
    """Validated configuration for a complete rho/disorder sweep."""

    Lx: int
    Ly: int
    observation_protocol: str
    observation_rate: float
    rho_values: tuple[float, ...]
    disorder_samples: int
    base_seed: int
    exact_max_variables: int
    exact_chunk_size: int
    bp_max_iterations: int
    bp_tolerance: float
    bp_damping: float
    bp_initializations: tuple[str, ...]
    mcmc_block_size: int
    mcmc_burn_in: int
    mcmc_samples: int
    mcmc_thinning: int
    mcmc_chains: int

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SweepExperimentConfig":
        try:
            model = data["model"]
            observation = data["observation"]
            sweep = data["sweep"]
            exact = data["exact"]
            bp = data["bp"]
            mcmc = data["mcmc"]
            if mcmc.get("sampler", "local_bfs_blocked_gibbs") != "local_bfs_blocked_gibbs":
                raise ValueError("unsupported MCMC sampler")
            config = cls(
                Lx=int(model["Lx"]),
                Ly=int(model["Ly"]),
                observation_protocol=str(observation["protocol"]),
                observation_rate=float(observation["observation_rate"]),
                rho_values=tuple(float(value) for value in sweep["rho_values"]),
                disorder_samples=int(sweep["disorder_samples"]),
                base_seed=int(sweep["base_seed"]),
                exact_max_variables=int(exact["max_variables"]),
                exact_chunk_size=int(exact.get("chunk_size", 65_536)),
                bp_max_iterations=int(bp["max_iterations"]),
                bp_tolerance=float(bp["tolerance"]),
                bp_damping=float(bp["damping"]),
                bp_initializations=tuple(str(value) for value in bp["initializations"]),
                mcmc_block_size=int(mcmc["block_size"]),
                mcmc_burn_in=int(mcmc["burn_in"]),
                mcmc_samples=int(mcmc["samples"]),
                mcmc_thinning=int(mcmc["thinning"]),
                mcmc_chains=int(mcmc["chains"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid sweep configuration: {error}") from error
        config.validate()
        return config

    def validate(self) -> None:
        if self.Lx <= 0 or self.Ly <= 0:
            raise ValueError("model dimensions must be positive")
        if self.observation_protocol not in ("all_safe", "bernoulli_safe"):
            raise ValueError("unsupported observation protocol")
        if not 0.0 <= self.observation_rate <= 1.0:
            raise ValueError("observation_rate must lie in [0, 1]")
        if self.observation_protocol == "all_safe" and self.observation_rate != 1.0:
            raise ValueError("all_safe requires observation_rate=1.0")
        if not self.rho_values or any(
            not math.isfinite(rho) or not 0.0 <= rho <= 1.0
            for rho in self.rho_values
        ):
            raise ValueError("rho_values must be a nonempty sequence in [0, 1]")
        if len(set(self.rho_values)) != len(self.rho_values):
            raise ValueError("rho_values must not contain duplicates")
        if self.disorder_samples <= 0:
            raise ValueError("disorder_samples must be positive")
        if self.base_seed < 0:
            raise ValueError("base_seed must be nonnegative")
        BPConfig(
            max_iterations=self.bp_max_iterations,
            tolerance=self.bp_tolerance,
            damping=self.bp_damping,
        )
        if not self.bp_initializations:
            raise ValueError("at least one BP initialization is required")
        for initialization in self.bp_initializations:
            BPConfig(initialization=initialization)
        MCMCConfig(
            block_size=self.mcmc_block_size,
            burn_in=self.mcmc_burn_in,
            samples=self.mcmc_samples,
            thinning=self.mcmc_thinning,
        )
        if self.mcmc_chains < 2:
            raise ValueError("mcmc.chains must be at least two")
        if not 0 <= self.exact_max_variables <= 62:
            raise ValueError("exact.max_variables must lie in [0, 62]")
        if self.exact_chunk_size <= 0:
            raise ValueError("exact.chunk_size must be positive")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "inference": {
                "target_distribution": "uniform_feasible_assignments",
            },
            "model": {"Lx": self.Lx, "Ly": self.Ly},
            "observation": {
                "protocol": self.observation_protocol,
                "observation_rate": self.observation_rate,
            },
            "sweep": {
                "rho_values": list(self.rho_values),
                "disorder_samples": self.disorder_samples,
                "base_seed": self.base_seed,
            },
            "exact": {
                "max_variables": self.exact_max_variables,
                "chunk_size": self.exact_chunk_size,
            },
            "bp": {
                "max_iterations": self.bp_max_iterations,
                "tolerance": self.bp_tolerance,
                "damping": self.bp_damping,
                "initializations": list(self.bp_initializations),
            },
            "mcmc": {
                "sampler": "local_bfs_blocked_gibbs",
                "block_size": self.mcmc_block_size,
                "burn_in": self.mcmc_burn_in,
                "samples": self.mcmc_samples,
                "thinning": self.mcmc_thinning,
                "chains": self.mcmc_chains,
            },
        }


def load_sweep_config(path: str | Path) -> SweepExperimentConfig:
    return SweepExperimentConfig.from_mapping(read_json(path))


def prepare_output_directory(
    config: SweepExperimentConfig, output_directory: str | Path
) -> Path:
    """Create an output root while preventing mixed experiment configurations."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    requested_config = config.to_mapping()
    if config_path.exists() and read_json(config_path) != requested_config:
        raise ValueError(
            "output directory contains results from a different configuration; "
            "choose another directory"
        )
    write_json_atomic(config_path, requested_config)
    return output


def derive_seed(
    base_seed: int,
    rho_index: int,
    disorder_index: int,
    stream: str,
    replicate: int = 0,
) -> int:
    """Derive stable, stream-separated 63-bit seeds from experiment indices."""

    payload = (
        f"{base_seed}|{rho_index}|{disorder_index}|{stream}|{replicate}".encode(
            "utf-8"
        )
    )
    digest = hashlib.blake2b(payload, digest_size=8, person=b"ms-csp-v1").digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def _metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    differences = np.asarray(left, dtype=np.float64) - np.asarray(
        right, dtype=np.float64
    )
    if differences.size == 0:
        return {"mae": 0.0, "rmse": 0.0, "max_absolute_error": 0.0}
    return {
        "mae": float(np.mean(np.abs(differences))),
        "rmse": float(np.sqrt(np.mean(differences * differences))),
        "max_absolute_error": float(np.max(np.abs(differences))),
    }


def _mean_or_none(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else None


def _max_or_none(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.max()) if finite.size else None


def _min_or_none(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.min()) if finite.size else None


def _exact_summary(exact: ExactResult | None, skip_reason: str | None) -> dict[str, Any]:
    if exact is None:
        return {"available": False, "skip_reason": skip_reason}
    return {
        "available": True,
        "number_of_feasible_solutions": exact.number_of_feasible_solutions,
        "partition_function": exact.partition_function,
        "log_partition_function": exact.log_partition_function,
        "mine_count_histogram": exact.mine_count_histogram,
        "planted_overlap_mean": (
            exact.planted_overlap.mean if exact.planted_overlap is not None else None
        ),
        "replica_overlap_mean": (
            exact.replica_overlap.mean if exact.replica_overlap is not None else None
        ),
    }


def _bp_summary(bp: Any) -> dict[str, Any]:
    runs: tuple[BPResult, ...] = bp.runs
    return {
        "runs": [
            {
                "initialization": run.config.initialization,
                "seed": run.config.seed,
                "status": run.status,
                "converged": run.converged,
                "iterations": run.iterations,
                "tolerance": run.config.tolerance,
                "damping": run.config.damping,
                "final_max_message_delta": run.final_max_message_delta,
                "failure_reason": run.failure_reason,
            }
            for run in runs
        ],
        "primary_run": 0,
        "convergence_fraction": sum(run.converged for run in runs) / len(runs),
        "max_iterations": max(run.iterations for run in runs),
        "max_pairwise_marginal_difference": bp.max_pairwise_marginal_difference,
        "fixed_point_groups": bp.fixed_point_groups(),
    }


def _mcmc_summary(mcmc: MCMCMultipleResult) -> dict[str, Any]:
    finite_r_hat = mcmc.r_hat[np.isfinite(mcmc.r_hat)]
    overlap_traces = [
        chain.planted_overlap_trace
        for chain in mcmc.chains
        if chain.planted_overlap_trace is not None
    ]
    planted_mean = (
        float(np.concatenate(overlap_traces).mean()) if overlap_traces else None
    )
    return {
        "sampler": mcmc.chains[0].sampler,
        "chains": [
            {
                "seed": chain.config.seed,
                "chain_length": chain.chain_length,
                "burn_in": chain.config.burn_in,
                "samples": chain.config.samples,
                "thinning": chain.config.thinning,
                "block_size": chain.config.block_size,
                "acceptance_rate": chain.acceptance_rate,
                "changed_update_rate": chain.changed_update_rate,
                "mine_density_autocorrelation_time": (
                    chain.mine_density_autocorrelation_time
                ),
                "mine_density_effective_sample_size": (
                    chain.mine_density_effective_sample_size
                ),
                "planted_overlap_autocorrelation_time": (
                    chain.planted_overlap_autocorrelation_time
                ),
                "planted_overlap_effective_sample_size": (
                    chain.planted_overlap_effective_sample_size
                ),
                "minimum_variable_effective_sample_size": _min_or_none(
                    chain.variable_effective_sample_sizes
                ),
            }
            for chain in mcmc.chains
        ],
        "max_pairwise_marginal_difference": (
            mcmc.max_pairwise_marginal_difference
        ),
        "max_finite_r_hat": (
            float(finite_r_hat.max()) if finite_r_hat.size else None
        ),
        "infinite_r_hat_variables": int(np.isinf(mcmc.r_hat).sum()),
        "mean_density_autocorrelation_time": _mean_or_none(
            np.array(
                [chain.mine_density_autocorrelation_time for chain in mcmc.chains]
            )
        ),
        "mean_planted_overlap_autocorrelation_time": _mean_or_none(
            np.array(
                [
                    chain.planted_overlap_autocorrelation_time
                    for chain in mcmc.chains
                    if chain.planted_overlap_autocorrelation_time is not None
                ]
            )
        ),
        "minimum_variable_effective_sample_size": _min_or_none(
            np.concatenate(
                [chain.variable_effective_sample_sizes for chain in mcmc.chains]
            )
        ),
        "planted_overlap_mean": planted_mean,
        "replica_overlap_mean": mcmc.replica_overlap_mean,
    }


def _raw_arrays(
    instance: Any,
    observed: np.ndarray,
    clues: np.ndarray,
    exact: ExactResult | None,
    bp: Any,
    mcmc: MCMCMultipleResult,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "ground_truth": instance.ground_truth,
        "observation_mask": observed,
        "clues": clues,
        "mcmc_pooled_marginals": mcmc.pooled_marginals,
        "mcmc_r_hat": mcmc.r_hat,
        "mcmc_replica_overlap_trace": mcmc.replica_overlap_trace,
    }
    for index, run in enumerate(bp.runs):
        arrays[f"bp_{index}_marginals"] = run.marginals
        arrays[f"bp_{index}_residual_history"] = np.asarray(run.residual_history)
    for index, chain in enumerate(mcmc.chains):
        arrays[f"mcmc_{index}_initial_assignment"] = chain.initial_assignment
        arrays[f"mcmc_{index}_retained_samples"] = chain.retained_samples
        arrays[f"mcmc_{index}_marginals"] = chain.marginals
        arrays[f"mcmc_{index}_marginal_standard_errors"] = (
            chain.marginal_standard_errors
        )
        arrays[f"mcmc_{index}_variable_autocorrelation_times"] = (
            chain.variable_autocorrelation_times
        )
        arrays[f"mcmc_{index}_variable_effective_sample_sizes"] = (
            chain.variable_effective_sample_sizes
        )
        arrays[f"mcmc_{index}_mine_density_trace"] = chain.mine_density_trace
        if chain.planted_overlap_trace is not None:
            arrays[f"mcmc_{index}_planted_overlap_trace"] = (
                chain.planted_overlap_trace
            )
    if exact is not None:
        arrays.update(
            {
                "exact_feasible_states": exact.feasible_states,
                "exact_posterior_probabilities": exact.posterior_probabilities,
                "exact_marginals": exact.marginals,
            }
        )
        if exact.planted_overlap is not None:
            arrays["exact_planted_overlap_values"] = exact.planted_overlap.values
            arrays["exact_planted_overlap_probabilities"] = (
                exact.planted_overlap.probabilities
            )
        if exact.replica_overlap is not None:
            arrays["exact_replica_overlap_values"] = exact.replica_overlap.values
            arrays["exact_replica_overlap_probabilities"] = (
                exact.replica_overlap.probabilities
            )
    return arrays


def run_disorder_sample(
    config: SweepExperimentConfig,
    rho_index: int,
    disorder_index: int,
    output_directory: str | Path,
    *,
    overwrite: bool = False,
    repository_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Run and persist one reproducible parameter/disorder point."""

    rho = config.rho_values[rho_index]
    output_directory = prepare_output_directory(config, output_directory)
    run_directory = (
        output_directory
        / f"rho_{rho_index:03d}_{rho:.6f}"
        / f"sample_{disorder_index:04d}"
    )
    summary_path = run_directory / "run.json"
    if summary_path.exists() and not overwrite:
        return read_json(summary_path)

    seed_instance = derive_seed(
        config.base_seed, rho_index, disorder_index, "instance"
    )
    seed_observation = derive_seed(
        config.base_seed, rho_index, disorder_index, "observation"
    )
    bp_seeds = [
        derive_seed(config.base_seed, rho_index, disorder_index, "bp", index)
        for index in range(len(config.bp_initializations))
    ]
    mcmc_seeds = [
        derive_seed(config.base_seed, rho_index, disorder_index, "mcmc", index)
        for index in range(config.mcmc_chains)
    ]
    mcmc_initial_seeds: list[int | None] = [None] + [
        derive_seed(
            config.base_seed,
            rho_index,
            disorder_index,
            "mcmc_initial",
            index,
        )
        for index in range(1, config.mcmc_chains)
    ]

    instance = generate_ground_truth(
        config.Lx, config.Ly, rho, seed=seed_instance
    )
    observed = generate_observation_mask(
        instance,
        protocol=config.observation_protocol,
        observation_rate=config.observation_rate,
        seed=seed_observation,
    )
    clues = generate_clues(instance.ground_truth, observed)
    graph = build_factor_graph(observed, clues)

    exact: ExactResult | None
    exact_skip_reason: str | None = None
    try:
        exact = solve_exact(
            graph,
            rho,
            planted_ground_truth=instance.ground_truth,
            max_variables=config.exact_max_variables,
            chunk_size=config.exact_chunk_size,
        )
    except ExactEnumerationLimitError as error:
        exact = None
        exact_skip_reason = str(error)

    bp_configs = [
        BPConfig(
            max_iterations=config.bp_max_iterations,
            tolerance=config.bp_tolerance,
            damping=config.bp_damping,
            initialization=initialization,
            seed=seed,
        )
        for initialization, seed in zip(config.bp_initializations, bp_seeds)
    ]
    bp = run_bp_multiple(graph, rho, bp_configs)

    initial_assignments: list[np.ndarray] = [instance.ground_truth]
    initial_sources = ["planted"]
    if exact is not None and exact.has_posterior_mass:
        for chain_index in range(1, config.mcmc_chains):
            rng = np.random.default_rng(mcmc_initial_seeds[chain_index])
            selected = int(
                rng.choice(
                    exact.number_of_feasible_solutions,
                    p=exact.posterior_probabilities,
                )
            )
            initial_assignments.append(exact.feasible_states[selected])
            initial_sources.append("exact_posterior")
    while len(initial_assignments) < config.mcmc_chains:
        initial_assignments.append(instance.ground_truth)
        initial_sources.append("planted")

    mcmc_configs = [
        MCMCConfig(
            block_size=config.mcmc_block_size,
            burn_in=config.mcmc_burn_in,
            samples=config.mcmc_samples,
            thinning=config.mcmc_thinning,
            seed=seed,
        )
        for seed in mcmc_seeds
    ]
    mcmc = run_blocked_gibbs_chains(
        graph,
        rho,
        initial_assignments,
        mcmc_configs,
        planted_ground_truth=instance.ground_truth,
    )

    primary_bp = bp.runs[0]
    comparisons: dict[str, Any] = {
        "bp_vs_mcmc": (
            _metrics(primary_bp.marginals, mcmc.pooled_marginals)
            if np.all(np.isfinite(primary_bp.marginals))
            else None
        ),
        "bp_vs_exact": None,
        "mcmc_vs_exact": None,
    }
    if exact is not None and exact.has_posterior_mass:
        if np.all(np.isfinite(primary_bp.marginals)):
            comparisons["bp_vs_exact"] = vars(
                compare_bp_to_exact(primary_bp, exact)
            )
        comparisons["mcmc_vs_exact"] = vars(compare_mcmc_to_exact(mcmc, exact))

    mcmc_summary = _mcmc_summary(mcmc)
    for chain, source in zip(mcmc_summary["chains"], initial_sources):
        chain["initialization"] = source
    summary: dict[str, Any] = {
        "schema_version": 2,
        "run_id": f"rho-{rho_index:03d}-sample-{disorder_index:04d}",
        "model": {
            "Lx": config.Lx,
            "Ly": config.Ly,
            "N": config.Lx * config.Ly,
            "rho": rho,
        },
        "inference": {
            "target_distribution": "uniform_feasible_assignments",
            "rho_used_as_inference_prior": False,
        },
        "observation": {
            "protocol": config.observation_protocol,
            "observation_rate": config.observation_rate,
        },
        "instance": {
            "disorder_index": disorder_index,
            "seed_instance": seed_instance,
            "seed_observation": seed_observation,
            "number_of_mines": instance.number_of_mines,
            "number_of_variables": graph.number_of_variables,
            "number_of_constraints": graph.number_of_constraints,
        },
        "exact": _exact_summary(exact, exact_skip_reason),
        "bp": _bp_summary(bp),
        "mcmc": mcmc_summary,
        "comparison": comparisons,
        "overlap": {
            "exact_planted_mean": (
                exact.planted_overlap.mean
                if exact is not None and exact.planted_overlap is not None
                else None
            ),
            "exact_replica_mean": (
                exact.replica_overlap.mean
                if exact is not None and exact.replica_overlap is not None
                else None
            ),
            "mcmc_planted_mean": mcmc_summary["planted_overlap_mean"],
            "mcmc_replica_mean": mcmc.replica_overlap_mean,
        },
        "seeds": {
            "instance": seed_instance,
            "observation": seed_observation,
            "bp": bp_seeds,
            "mcmc": mcmc_seeds,
            "mcmc_initial": mcmc_initial_seeds,
        },
        "files": {"arrays": "arrays.npz"},
        "metadata": {
            "git_commit": git_commit(repository_directory),
            "git_dirty": git_is_dirty(repository_directory),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "package_version": "0.1.0",
            "timestamp_utc": utc_timestamp(),
        },
    }
    write_npz_atomic(
        run_directory / "arrays.npz",
        _raw_arrays(instance, observed, clues, exact, bp, mcmc),
    )
    write_json_atomic(summary_path, summary)
    # Return the same strict-JSON representation on fresh and resumed runs.
    return read_json(summary_path)


def execute_sweep(
    config: SweepExperimentConfig,
    output_directory: str | Path,
    *,
    overwrite: bool = False,
    repository_directory: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Execute all rho/disorder points, resuming completed runs by default."""

    output = prepare_output_directory(config, output_directory)
    summaries = []
    for rho_index in range(len(config.rho_values)):
        for disorder_index in range(config.disorder_samples):
            summaries.append(
                run_disorder_sample(
                    config,
                    rho_index,
                    disorder_index,
                    output,
                    overwrite=overwrite,
                    repository_directory=repository_directory,
                )
            )
    return summaries
