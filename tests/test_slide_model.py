"""Check the slide model on solutions with different total mine counts."""

import unittest

import numpy as np

from minesweeper_csp import (
    BPConfig,
    MCMCConfig,
    build_factor_graph,
    generate_clues,
    run_blocked_gibbs,
    run_bp,
    solve_exact,
)


class SlideModelTests(unittest.TestCase):
    def test_uniform_solutions_without_prior_or_total_mine_constraint(self) -> None:
        # The two local clues impose x0+x1=1 and x1+x2=1.
        # The feasible solutions 010 and 101 have different total mine counts.
        truth = np.array([[False, False, True, False, False]])
        observed = np.array([[False, True, False, True, False]])
        graph = build_factor_graph(observed, generate_clues(truth, observed))
        bp_config = BPConfig(initialization="random", seed=7, tolerance=1e-12)
        mcmc_config = MCMCConfig(block_size=3, burn_in=100, samples=2_000, seed=8)
        previous_samples = None

        for rho in (0.0, 0.2, 0.8, 1.0):
            with self.subTest(rho=rho):
                exact = solve_exact(graph, rho, planted_ground_truth=truth)
                self.assertEqual(exact.mine_count_histogram, (0, 1, 1, 0))
                np.testing.assert_allclose(exact.posterior_probabilities, [0.5, 0.5])
                np.testing.assert_allclose(exact.marginals, [0.5, 0.5, 0.5])

                bp = run_bp(graph, rho, config=bp_config)
                self.assertTrue(bp.converged)
                np.testing.assert_allclose(bp.marginals, exact.marginals, atol=1e-10)

                mcmc = run_blocked_gibbs(
                    graph, rho, truth, config=mcmc_config, planted_ground_truth=truth
                )
                np.testing.assert_allclose(mcmc.marginals, exact.marginals, atol=0.04)
                self.assertEqual(set(mcmc.retained_samples.sum(axis=1)), {1, 2})
                np.testing.assert_array_equal(
                    mcmc.planted_overlap_trace,
                    (mcmc.retained_samples == [False, True, False]).mean(axis=1),
                )
                if previous_samples is not None:
                    np.testing.assert_array_equal(mcmc.retained_samples, previous_samples)
                previous_samples = mcmc.retained_samples


if __name__ == "__main__":
    unittest.main()
