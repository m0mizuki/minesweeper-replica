import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from minesweeper_csp.aggregate import aggregate_directory, summarize_samples
from minesweeper_csp.experiment import (
    SweepExperimentConfig,
    derive_seed,
    execute_sweep,
)
from minesweeper_csp.io import read_json
from minesweeper_csp.visualization import write_phase_scan_svg


def tiny_config() -> SweepExperimentConfig:
    return SweepExperimentConfig.from_mapping(
        {
            "model": {"Lx": 2, "Ly": 3},
            "observation": {
                "protocol": "bernoulli_safe",
                "observation_rate": 0.5,
            },
            "sweep": {
                "rho_values": [0.2, 0.4],
                "disorder_samples": 2,
                "base_seed": 123456,
            },
            "exact": {"max_variables": 6, "chunk_size": 64},
            "bp": {
                "max_iterations": 100,
                "tolerance": 1e-9,
                "damping": 0.2,
                "initializations": ["prior", "random"],
            },
            "mcmc": {
                "block_size": 6,
                "burn_in": 10,
                "samples": 40,
                "thinning": 1,
                "chains": 2,
            },
        }
    )


class ExperimentTests(unittest.TestCase):
    def test_seed_derivation_is_stable_and_stream_separated(self) -> None:
        first = derive_seed(10, 2, 3, "instance")
        self.assertEqual(first, derive_seed(10, 2, 3, "instance"))
        self.assertNotEqual(first, derive_seed(10, 2, 3, "observation"))
        self.assertNotEqual(first, derive_seed(10, 2, 4, "instance"))

    def test_sweep_writes_reproducible_run_and_raw_data(self) -> None:
        config = tiny_config()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            first = execute_sweep(config, output, repository_directory=Path.cwd())
            second = execute_sweep(config, output, repository_directory=Path.cwd())

            self.assertEqual(first, second)
            self.assertEqual(len(first), 4)
            self.assertEqual(len({run["seeds"]["instance"] for run in first}), 4)
            self.assertTrue((output / "config.json").exists())
            for run in first:
                rho = run["model"]["rho"]
                rho_index = config.rho_values.index(rho)
                sample_index = run["instance"]["disorder_index"]
                run_directory = (
                    output
                    / f"rho_{rho_index:03d}_{rho:.6f}"
                    / f"sample_{sample_index:04d}"
                )
                self.assertTrue((run_directory / "run.json").exists())
                with np.load(run_directory / "arrays.npz") as arrays:
                    self.assertIn("ground_truth", arrays)
                    self.assertIn("exact_marginals", arrays)
                    self.assertIn("mcmc_replica_overlap_trace", arrays)
                    self.assertEqual(arrays["ground_truth"].shape, (2, 3))

                raw_json = (run_directory / "run.json").read_text(encoding="utf-8")
                json.loads(raw_json, parse_constant=lambda value: self.fail(value))

    def test_aggregate_contains_disorder_statistics_and_svg(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            execute_sweep(tiny_config(), output)
            aggregate = aggregate_directory(output)
            svg_path = write_phase_scan_svg(aggregate, output / "phase_scan.svg")

            self.assertEqual(aggregate["run_count"], 4)
            self.assertEqual(len(aggregate["parameter_points"]), 2)
            for point in aggregate["parameter_points"]:
                statistics = point["metrics"]["bp_mcmc_mae"]
                self.assertEqual(statistics["count"], 2)
                self.assertIn("standard_error", statistics)
                self.assertIn("q50", statistics["quantiles"])
            self.assertTrue(svg_path.exists())
            svg = svg_path.read_text(encoding="utf-8")
            self.assertIn("Minesweeper CSP phase scan", svg)
            self.assertIn("Marginal disagreement", svg)
            self.assertEqual(read_json(output / "aggregate.json")["run_count"], 4)

    def test_summary_statistics_record_missing_values(self) -> None:
        summary = summarize_samples([1.0, 2.0, None])
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["missing"], 1)
        self.assertAlmostEqual(summary["mean"], 1.5)

    def test_invalid_sweep_configuration_is_rejected(self) -> None:
        mapping = tiny_config().to_mapping()
        mapping["mcmc"]["chains"] = 1
        with self.assertRaises(ValueError):
            SweepExperimentConfig.from_mapping(mapping)

    def test_resume_rejects_a_different_configuration(self) -> None:
        first = tiny_config()
        changed_mapping = first.to_mapping()
        changed_mapping["sweep"]["base_seed"] += 1
        changed = SweepExperimentConfig.from_mapping(changed_mapping)
        with tempfile.TemporaryDirectory() as directory:
            execute_sweep(first, directory)
            with self.assertRaises(ValueError):
                execute_sweep(changed, directory)
            with self.assertRaises(ValueError):
                execute_sweep(changed, directory, overwrite=True)


if __name__ == "__main__":
    unittest.main()
