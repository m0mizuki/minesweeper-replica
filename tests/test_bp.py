import unittest

import numpy as np

from minesweeper_csp import (
    BPConfig,
    build_factor_graph,
    compare_bp_to_exact,
    generate_clues,
    run_bp,
    run_bp_multiple,
    solve_exact,
)


def unconstrained_graph(shape: tuple[int, int]):
    observed = np.zeros(shape, dtype=np.bool_)
    clues = np.full(shape, -1, dtype=np.int_)
    return build_factor_graph(observed, clues)


def one_of_two_graph():
    truth = np.array([[True, False, False]], dtype=np.bool_)
    observed = np.array([[False, True, False]], dtype=np.bool_)
    graph = build_factor_graph(observed, generate_clues(truth, observed))
    return graph, truth


class BeliefPropagationTests(unittest.TestCase):
    def test_unconstrained_marginals_equal_prior(self) -> None:
        result = run_bp(unconstrained_graph((2, 2)), 0.23)

        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 0)
        np.testing.assert_allclose(result.marginals, np.full(4, 0.23))

    def test_tree_factor_matches_exact_marginals(self) -> None:
        graph, truth = one_of_two_graph()
        exact = solve_exact(graph, 0.17, planted_ground_truth=truth)
        bp = run_bp(
            graph,
            0.17,
            config=BPConfig(tolerance=1e-13, initialization="random", seed=9),
        )
        comparison = compare_bp_to_exact(bp, exact)

        self.assertTrue(bp.converged)
        np.testing.assert_allclose(bp.marginals, exact.marginals, atol=1e-12)
        self.assertLess(comparison.mae, 1e-12)
        self.assertLess(comparison.rmse, 1e-12)

    def test_every_message_is_normalized(self) -> None:
        graph, _ = one_of_two_graph()
        result = run_bp(
            graph,
            0.3,
            config=BPConfig(damping=0.4, initialization="random", seed=4),
        )

        np.testing.assert_allclose(
            result.variable_to_factor_messages.sum(axis=1), 1.0
        )
        np.testing.assert_allclose(
            result.factor_to_variable_messages.sum(axis=1), 1.0
        )
        self.assertTrue(np.all(result.variable_to_factor_messages >= 0.0))
        self.assertTrue(np.all(result.factor_to_variable_messages >= 0.0))

    def test_random_initialization_is_reproducible(self) -> None:
        graph, _ = one_of_two_graph()
        config = BPConfig(max_iterations=1, initialization="random", seed=123)
        first = run_bp(graph, 0.3, config=config)
        second = run_bp(graph, 0.3, config=config)

        np.testing.assert_array_equal(
            first.variable_to_factor_messages,
            second.variable_to_factor_messages,
        )
        np.testing.assert_array_equal(first.residual_history, second.residual_history)

    def test_max_iteration_status_is_recorded(self) -> None:
        graph, _ = one_of_two_graph()
        result = run_bp(
            graph,
            0.3,
            config=BPConfig(
                max_iterations=1,
                tolerance=1e-15,
                damping=0.5,
                initialization="random",
                seed=8,
            ),
        )

        self.assertEqual(result.status, "max_iterations")
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(len(result.residual_history), 1)

    def test_locally_impossible_constraint_is_reported(self) -> None:
        observed = np.array([[False, True, False]], dtype=np.bool_)
        graph = build_factor_graph(
            observed, np.array([[-1, 3, -1]], dtype=np.int_)
        )
        result = run_bp(graph, 0.5)

        self.assertEqual(result.status, "infeasible")
        self.assertEqual(result.iterations, 0)
        self.assertTrue(np.isnan(result.marginals).all())
        self.assertIn("exceeds factor degree", result.failure_reason)

    def test_zero_prior_mass_is_not_replaced_by_uniform_message(self) -> None:
        graph, _ = one_of_two_graph()
        result = run_bp(graph, 0.0)

        self.assertEqual(result.status, "infeasible")
        self.assertTrue(np.isnan(result.marginals).all())

    def test_multiple_initializations_reach_same_tree_fixed_point(self) -> None:
        graph, _ = one_of_two_graph()
        multiple = run_bp_multiple(
            graph,
            0.3,
            [
                BPConfig(initialization="uniform"),
                BPConfig(initialization="prior"),
                BPConfig(initialization="random", seed=1),
                BPConfig(initialization="random", seed=2),
            ],
        )

        self.assertTrue(multiple.all_converged)
        self.assertLess(multiple.max_pairwise_marginal_difference, 1e-9)
        self.assertEqual(multiple.fixed_point_groups(tolerance=1e-9), ((0, 1, 2, 3),))

    def test_invalid_config_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BPConfig(damping=1.0)
        with self.assertRaises(ValueError):
            BPConfig(tolerance=0.0)
        with self.assertRaises(ValueError):
            BPConfig(max_iterations=0)

    def test_comparison_rejects_different_prior(self) -> None:
        graph, truth = one_of_two_graph()
        exact = solve_exact(graph, 0.2, planted_ground_truth=truth)
        bp = run_bp(graph, 0.3)

        with self.assertRaises(ValueError):
            compare_bp_to_exact(bp, exact)


if __name__ == "__main__":
    unittest.main()
