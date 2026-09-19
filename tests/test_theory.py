from pathlib import Path
import math
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from minesweeper_csp import (
    BPConfig,
    RSStabilityLimitError,
    analyze_rs_run,
    build_factor_graph,
    compute_bethe_result,
    compute_rs_stability,
    factor_graph_topology,
    generate_clues,
    run_bp,
    solve_exact,
)
from minesweeper_csp.experiment import SweepExperimentConfig, execute_sweep
from minesweeper_csp.theory_visualization import write_rs_theory_svg


def one_of_two_graph():
    truth = np.array([[True, False, False]], dtype=np.bool_)
    observed = np.array([[False, True, False]], dtype=np.bool_)
    return build_factor_graph(observed, generate_clues(truth, observed)), truth


def four_cycle_graph():
    truth = np.array([[False, True], [False, False]], dtype=np.bool_)
    observed = np.array([[True, False], [False, True]], dtype=np.bool_)
    return build_factor_graph(observed, generate_clues(truth, observed)), truth


def summary(mean: float | None) -> dict:
    return {
        "count": 1 if mean is not None else 0,
        "missing": 0 if mean is not None else 1,
        "mean": mean,
        "standard_deviation": 0.0 if mean is not None else None,
        "standard_error": 0.0 if mean is not None else None,
        "quantiles": {},
    }


class TheoryTests(unittest.TestCase):
    def test_bethe_is_exact_and_stable_on_tree(self) -> None:
        graph, truth = one_of_two_graph()
        rho = 0.17
        exact = solve_exact(graph, rho, planted_ground_truth=truth)
        bp = run_bp(
            graph,
            rho,
            config=BPConfig(tolerance=1e-13, initialization="random", seed=4),
        )
        bethe = compute_bethe_result(graph, bp)
        stability = compute_rs_stability(graph, bp)

        self.assertAlmostEqual(
            bethe.log_partition_function,
            exact.log_partition_function,
            places=11,
        )
        self.assertAlmostEqual(bethe.entropy, math.log(2.0), places=11)
        self.assertAlmostEqual(stability.spectral_radius, 0.0, places=10)
        self.assertTrue(stability.locally_stable)

    def test_topology_counts_k22_cycle(self) -> None:
        graph, _ = four_cycle_graph()
        topology = factor_graph_topology(graph)

        self.assertEqual(topology.number_of_variables, 2)
        self.assertEqual(topology.number_of_factors, 2)
        self.assertEqual(topology.number_of_edges, 4)
        self.assertEqual(topology.connected_components, 1)
        self.assertEqual(topology.cycle_rank, 1)
        self.assertEqual(topology.four_cycle_count, 1)

    def test_loopy_stability_is_finite_and_edge_guarded(self) -> None:
        graph, _ = four_cycle_graph()
        bp = run_bp(
            graph,
            0.3,
            config=BPConfig(tolerance=1e-12, damping=0.2),
        )
        stability = compute_rs_stability(graph, bp)

        self.assertTrue(math.isfinite(stability.spectral_radius))
        self.assertEqual(stability.jacobian.shape, (4, 4))
        with self.assertRaises(RSStabilityLimitError):
            compute_rs_stability(graph, bp, max_edges=3)
        with self.assertRaises(ValueError):
            compute_rs_stability(graph, bp, max_edges=-1)

    def test_unconstrained_graph_counts_all_assignments(self) -> None:
        observed = np.zeros((2, 2), dtype=np.bool_)
        clues = np.full((2, 2), -1, dtype=np.int_)
        graph = build_factor_graph(observed, clues)
        bp = run_bp(graph, 0.23)
        bethe = compute_bethe_result(graph, bp)
        stability = compute_rs_stability(graph, bp)

        self.assertAlmostEqual(bethe.log_partition_function, math.log(16.0))
        self.assertAlmostEqual(bethe.entropy, math.log(16.0))
        self.assertAlmostEqual(stability.spectral_radius, 0.0)
        self.assertEqual(stability.jacobian.shape, (0, 0))

    def test_saved_run_analysis_and_svg(self) -> None:
        config = SweepExperimentConfig.from_mapping(
            {
                "model": {"Lx": 2, "Ly": 2},
                "observation": {
                    "protocol": "bernoulli_safe",
                    "observation_rate": 0.5,
                },
                "sweep": {
                    "rho_values": [0.25],
                    "disorder_samples": 1,
                    "base_seed": 777,
                },
                "exact": {"max_variables": 6, "chunk_size": 64},
                "bp": {
                    "max_iterations": 100,
                    "tolerance": 1e-10,
                    "damping": 0.2,
                    "initializations": ["uniform"],
                },
                "mcmc": {
                    "sampler": "local_bfs_blocked_gibbs",
                    "block_size": 4,
                    "burn_in": 2,
                    "samples": 5,
                    "thinning": 1,
                    "chains": 2,
                },
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            execute_sweep(config, output)
            run_path = next(output.glob("rho_*/sample_*/run.json"))
            diagnostic = analyze_rs_run(run_path)

            self.assertEqual(diagnostic["bp_status"], "converged")
            self.assertEqual(diagnostic["bp_replay_max_marginal_difference"], 0.0)
            self.assertIsNotNone(diagnostic["bethe"])
            self.assertIsNotNone(diagnostic["stability"])
            self.assertTrue((run_path.parent / "theory.json").exists())

            point = {
                "rho": 0.25,
                "spectral_radius": summary(0.4),
                "bethe_log_partition_error_per_variable": summary(0.01),
                "bethe_entropy_density": summary(0.2),
                "four_cycles_per_variable": summary(0.5),
            }
            analysis = {
                "sizes": [{"Lx": 2, "Ly": 2, "points": [point]}]
            }
            svg = write_rs_theory_svg(analysis, output / "theory.svg")
            ET.parse(svg)
            self.assertIn("local threshold", svg.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
