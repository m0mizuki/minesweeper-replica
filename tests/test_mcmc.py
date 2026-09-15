import math
import unittest

import numpy as np

from minesweeper_csp import (
    MCMCConfig,
    build_factor_graph,
    compare_mcmc_to_exact,
    effective_sample_size,
    generate_clues,
    gelman_rubin_r_hat,
    integrated_autocorrelation_time,
    run_blocked_gibbs,
    run_blocked_gibbs_chains,
    solve_exact,
)


def one_of_two_graph():
    truth = np.array([[True, False, False]], dtype=np.bool_)
    observed = np.array([[False, True, False]], dtype=np.bool_)
    graph = build_factor_graph(observed, generate_clues(truth, observed))
    return graph, truth


def unconstrained_graph(shape: tuple[int, int]):
    observed = np.zeros(shape, dtype=np.bool_)
    clues = np.full(shape, -1, dtype=np.int_)
    return build_factor_graph(observed, clues)


class MCMCTests(unittest.TestCase):
    def test_unconstrained_chain_respects_bernoulli_prior(self) -> None:
        graph = unconstrained_graph((1, 2))
        initial = np.array([False, False], dtype=np.bool_)
        exact = solve_exact(graph, 0.23)
        result = run_blocked_gibbs(
            graph,
            0.23,
            initial,
            config=MCMCConfig(block_size=2, burn_in=500, samples=8_000, seed=17),
        )

        comparison = compare_mcmc_to_exact(result, exact)
        self.assertLess(comparison.max_absolute_error, 0.03)
        self.assertTrue(np.all(result.variable_effective_sample_sizes > 0.0))

    def test_blocked_chain_matches_exact_on_joint_move(self) -> None:
        graph, truth = one_of_two_graph()
        exact = solve_exact(graph, 0.3, planted_ground_truth=truth)
        result = run_blocked_gibbs(
            graph,
            0.3,
            truth,
            config=MCMCConfig(block_size=2, burn_in=200, samples=5_000, seed=4),
            planted_ground_truth=truth,
        )
        comparison = compare_mcmc_to_exact(result, exact)

        self.assertLess(comparison.max_absolute_error, 0.03)
        self.assertGreater(result.changed_update_rate, 0.4)
        self.assertIsNone(result.acceptance_rate)
        self.assertEqual(result.chain_length, 5_200)
        self.assertEqual(result.planted_overlap_trace.shape, (5_000,))
        np.testing.assert_array_equal(result.retained_samples.sum(axis=1), 1)

    def test_single_site_chain_exposes_nonergodicity(self) -> None:
        graph, truth = one_of_two_graph()
        result = run_blocked_gibbs(
            graph,
            0.3,
            truth,
            config=MCMCConfig(block_size=1, burn_in=10, samples=100, seed=2),
        )

        self.assertEqual(result.changed_update_rate, 0.0)
        np.testing.assert_array_equal(
            result.retained_samples,
            np.tile([True, False], (100, 1)),
        )

    def test_multiple_stuck_chains_flag_initialization_dependence(self) -> None:
        graph, truth = one_of_two_graph()
        opposite = np.array([False, True], dtype=np.bool_)
        multiple = run_blocked_gibbs_chains(
            graph,
            0.3,
            [truth, opposite],
            [
                MCMCConfig(block_size=1, burn_in=5, samples=50, seed=1),
                MCMCConfig(block_size=1, burn_in=5, samples=50, seed=2),
            ],
            planted_ground_truth=truth,
        )

        self.assertEqual(multiple.max_pairwise_marginal_difference, 1.0)
        self.assertTrue(np.isinf(multiple.r_hat).all())
        self.assertEqual(set(multiple.replica_overlap_trace), {-1.0})

    def test_independent_blocked_chains_have_replica_trace(self) -> None:
        graph, truth = one_of_two_graph()
        multiple = run_blocked_gibbs_chains(
            graph,
            0.3,
            [truth, truth],
            [
                MCMCConfig(block_size=2, burn_in=100, samples=2_000, seed=10),
                MCMCConfig(block_size=2, burn_in=100, samples=2_000, seed=11),
            ],
        )

        self.assertEqual(multiple.replica_overlap_trace.shape, (2_000,))
        self.assertLess(abs(multiple.replica_overlap_mean), 0.08)
        self.assertLess(multiple.max_pairwise_marginal_difference, 0.08)

    def test_seed_reproduces_chain(self) -> None:
        graph, truth = one_of_two_graph()
        config = MCMCConfig(block_size=2, burn_in=20, samples=100, seed=99)
        first = run_blocked_gibbs(graph, 0.3, truth, config=config)
        second = run_blocked_gibbs(graph, 0.3, truth, config=config)

        np.testing.assert_array_equal(first.retained_samples, second.retained_samples)
        np.testing.assert_array_equal(first.final_assignment, second.final_assignment)

    def test_invalid_initial_assignment_is_rejected(self) -> None:
        graph, truth = one_of_two_graph()
        with self.assertRaises(ValueError):
            run_blocked_gibbs(graph, 0.3, np.array([False, False]))
        with self.assertRaises(ValueError):
            run_blocked_gibbs(graph, 0.0, truth)

    def test_diagnostics_handle_correlation_and_constant_chains(self) -> None:
        alternating = np.tile([0.0, 1.0], 100)
        constant = np.ones(200)
        self.assertGreaterEqual(integrated_autocorrelation_time(alternating), 1.0)
        self.assertEqual(integrated_autocorrelation_time(constant), 1.0)
        self.assertEqual(effective_sample_size(constant), 200.0)

        agreeing = np.ones((2, 20, 1))
        disagreeing = np.stack([np.zeros((20, 1)), np.ones((20, 1))])
        np.testing.assert_allclose(gelman_rubin_r_hat(agreeing), [1.0])
        self.assertTrue(math.isinf(float(gelman_rubin_r_hat(disagreeing)[0])))

    def test_invalid_config_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MCMCConfig(block_size=0)
        with self.assertRaises(ValueError):
            MCMCConfig(samples=0)
        with self.assertRaises(ValueError):
            MCMCConfig(thinning=0)

    def test_result_arrays_are_immutable(self) -> None:
        graph, truth = one_of_two_graph()
        result = run_blocked_gibbs(
            graph,
            0.3,
            truth,
            config=MCMCConfig(block_size=2, burn_in=2, samples=5, seed=3),
        )
        with self.assertRaises(ValueError):
            result.marginals[0] = 0.0
        with self.assertRaises(ValueError):
            result.retained_samples[0, 0] = False

    def test_one_sample_reports_unavailable_standard_error(self) -> None:
        graph, truth = one_of_two_graph()
        result = run_blocked_gibbs(
            graph,
            0.3,
            truth,
            config=MCMCConfig(block_size=2, burn_in=0, samples=1, seed=3),
        )
        self.assertTrue(np.isnan(result.variable_autocorrelation_times).all())
        self.assertTrue(np.isnan(result.variable_effective_sample_sizes).all())
        self.assertTrue(np.isnan(result.marginal_standard_errors).all())


if __name__ == "__main__":
    unittest.main()
