from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from minesweeper_csp.aggregate import aggregate_directory
from minesweeper_csp.candidate import (
    build_candidate_plan,
    execute_candidate_plan,
    identify_candidate_region,
    score_candidate_points,
)
from minesweeper_csp.experiment import SweepExperimentConfig, execute_sweep


def metric(mean: float) -> dict:
    return {
        "count": 2,
        "missing": 0,
        "mean": mean,
        "standard_deviation": 0.0,
        "standard_error": 0.0,
        "quantiles": {
            "q05": mean,
            "q25": mean,
            "q50": mean,
            "q75": mean,
            "q95": mean,
        },
    }


def synthetic_aggregate() -> dict:
    base_metrics = {
        "bp_mcmc_mae": 0.01,
        "bp_convergence_fraction": 1.0,
        "bp_fixed_point_spread": 0.0,
        "mcmc_chain_marginal_spread": 0.01,
        "mcmc_max_r_hat": 1.01,
        "mcmc_density_tau": 1.0,
        "mcmc_overlap_tau": 1.0,
        "mcmc_min_variable_ess": 100.0,
    }
    points = []
    for rho in (0.1, 0.2, 0.3):
        values = dict(base_metrics)
        if rho == 0.2:
            values.update(
                {
                    "bp_mcmc_mae": 0.3,
                    "bp_convergence_fraction": 0.2,
                    "bp_fixed_point_spread": 0.4,
                    "mcmc_chain_marginal_spread": 0.25,
                    "mcmc_max_r_hat": 1.8,
                    "mcmc_density_tau": 20.0,
                    "mcmc_overlap_tau": 25.0,
                    "mcmc_min_variable_ess": 5.0,
                }
            )
        points.append(
            {"rho": rho, "metrics": {name: metric(value) for name, value in values.items()}}
        )
    return {"parameter_points": points}


def coarse_config() -> SweepExperimentConfig:
    return SweepExperimentConfig.from_mapping(
        {
            "model": {"Lx": 2, "Ly": 2},
            "observation": {
                "protocol": "bernoulli_safe",
                "observation_rate": 0.5,
            },
            "sweep": {
                "rho_values": [0.15, 0.35],
                "disorder_samples": 1,
                "base_seed": 9876,
            },
            "exact": {"max_variables": 6, "chunk_size": 64},
            "bp": {
                "max_iterations": 100,
                "tolerance": 1e-9,
                "damping": 0.2,
                "initializations": ["prior", "random"],
            },
            "mcmc": {
                "sampler": "local_bfs_blocked_gibbs",
                "block_size": 4,
                "burn_in": 5,
                "samples": 20,
                "thinning": 1,
                "chains": 2,
            },
        }
    )


class CandidateAnalysisTests(unittest.TestCase):
    def test_combined_score_prioritizes_joint_warning_point(self) -> None:
        aggregate = synthetic_aggregate()
        ranking = score_candidate_points(aggregate)
        selection = identify_candidate_region(aggregate, top_k=1, padding_points=1)

        self.assertEqual(ranking[0]["rho"], 0.2)
        self.assertEqual(selection["selected_rhos"], [0.2])
        self.assertEqual(selection["rho_min"], 0.1)
        self.assertEqual(selection["rho_max"], 0.3)
        self.assertIn("prioritization", selection["interpretation"])

    def test_candidate_plan_uses_distinct_size_seed_streams(self) -> None:
        aggregate = synthetic_aggregate()
        aggregate["config"] = coarse_config().to_mapping()
        aggregate["run_count"] = 2
        plan = build_candidate_plan(
            aggregate,
            sizes=[(2, 2), (2, 3)],
            dense_points=5,
            top_k=1,
            padding_points=0,
            bp_random_restarts=2,
        )

        self.assertEqual(len(plan["rho_values"]), 5)
        self.assertEqual(plan["rho_values"][0], 0.1)
        self.assertEqual(plan["rho_values"][-1], 0.3)
        seeds = [sweep["config"]["sweep"]["base_seed"] for sweep in plan["sweeps"]]
        self.assertEqual(len(set(seeds)), 2)
        self.assertEqual(
            plan["sweeps"][0]["config"]["bp"]["initializations"],
            ["prior", "uniform", "random", "random"],
        )

    def test_candidate_study_runs_size_analysis_and_pq_figures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            coarse_output = root / "coarse"
            execute_sweep(coarse_config(), coarse_output)
            aggregate = aggregate_directory(coarse_output)
            plan = build_candidate_plan(
                aggregate,
                sizes=[(2, 2), (2, 3)],
                dense_points=2,
                top_k=1,
                padding_points=0,
                disorder_samples=1,
                bp_random_restarts=1,
                mcmc_chains=2,
                overlap_bins=9,
            )
            output = root / "candidate"
            analysis = execute_candidate_plan(plan, output)

            self.assertEqual(len(analysis["sizes"]), 2)
            self.assertTrue((output / "candidate_analysis.json").exists())
            for name in ("candidate_summary.svg", "overlap_distributions.svg"):
                path = output / name
                self.assertTrue(path.exists())
                ET.parse(path)
            for size in analysis["sizes"]:
                self.assertEqual(len(size["points"]), 2)
                for point in size["points"]:
                    probabilities = point["mcmc_overlap"]["probabilities"]
                    self.assertAlmostEqual(sum(probabilities), 1.0)
                    self.assertIn("mode_count", point["mcmc_overlap"]["shape"])


if __name__ == "__main__":
    unittest.main()
