import math
import unittest

import numpy as np

from minesweeper_csp import (
    ExactEnumerationLimitError,
    build_factor_graph,
    enumerate_feasible_states,
    generate_clues,
    solve_exact,
)


def unconstrained_graph(shape: tuple[int, int]):
    observed = np.zeros(shape, dtype=np.bool_)
    clues = np.full(shape, -1, dtype=np.int_)
    return build_factor_graph(observed, clues)


class ExactSolverTests(unittest.TestCase):
    def test_unconstrained_nonuniform_prior(self) -> None:
        graph = unconstrained_graph((1, 2))
        planted = np.array([[False, False]], dtype=np.bool_)
        result = solve_exact(graph, 0.25, planted_ground_truth=planted)

        self.assertEqual(result.number_of_feasible_solutions, 4)
        self.assertEqual(result.mine_count_histogram, (1, 2, 1))
        self.assertAlmostEqual(result.partition_function, 1.0)
        self.assertAlmostEqual(result.log_partition_function, 0.0)
        np.testing.assert_allclose(result.marginals, [0.25, 0.25])
        np.testing.assert_allclose(
            result.planted_overlap.values, [1.0, 0.0, -1.0]
        )
        np.testing.assert_allclose(
            result.planted_overlap.probabilities, [0.5625, 0.375, 0.0625]
        )
        np.testing.assert_allclose(
            result.replica_overlap.probabilities,
            [0.390625, 0.46875, 0.140625],
        )
        self.assertAlmostEqual(result.replica_overlap.mean, 0.25)

    def test_equality_constraint_enumerates_only_feasible_states(self) -> None:
        truth = np.array([[True, False, False]], dtype=np.bool_)
        observed = np.array([[False, True, False]], dtype=np.bool_)
        clues = generate_clues(truth, observed)
        graph = build_factor_graph(observed, clues)

        states = enumerate_feasible_states(graph)
        self.assertEqual({tuple(row) for row in states.tolist()}, {(False, True), (True, False)})

        result = solve_exact(graph, 0.2, planted_ground_truth=truth)
        self.assertEqual(result.number_of_feasible_solutions, 2)
        self.assertAlmostEqual(result.partition_function, 2 * 0.2 * 0.8)
        np.testing.assert_allclose(result.posterior_probabilities, [0.5, 0.5])
        np.testing.assert_allclose(result.marginals, [0.5, 0.5])
        np.testing.assert_allclose(
            result.replica_overlap.probabilities, [0.5, 0.0, 0.5]
        )
        np.testing.assert_allclose(
            result.planted_overlap.probabilities, [0.5, 0.0, 0.5]
        )

    def test_no_variables_has_one_empty_assignment(self) -> None:
        truth = np.zeros((1, 1), dtype=np.bool_)
        observed = np.ones((1, 1), dtype=np.bool_)
        clues = generate_clues(truth, observed)
        graph = build_factor_graph(observed, clues)
        result = solve_exact(graph, 0.3, planted_ground_truth=truth)

        self.assertEqual(result.feasible_states.shape, (1, 0))
        self.assertEqual(result.number_of_feasible_solutions, 1)
        self.assertAlmostEqual(result.partition_function, 1.0)
        np.testing.assert_array_equal(result.marginals, np.empty(0))
        np.testing.assert_allclose(result.replica_overlap.values, [1.0])
        np.testing.assert_allclose(result.replica_overlap.probabilities, [1.0])

    def test_unsatisfiable_graph_reports_zero_mass(self) -> None:
        observed = np.array([[False, True, False]], dtype=np.bool_)
        clues = np.array([[-1, 3, -1]], dtype=np.int_)
        graph = build_factor_graph(observed, clues)
        result = solve_exact(graph, 0.5)

        self.assertEqual(result.number_of_feasible_solutions, 0)
        self.assertEqual(result.partition_function, 0.0)
        self.assertEqual(result.log_partition_function, -math.inf)
        self.assertFalse(result.has_posterior_mass)
        self.assertTrue(np.isnan(result.marginals).all())
        self.assertIsNone(result.replica_overlap)

    def test_prior_boundary_can_give_feasible_states_zero_mass(self) -> None:
        truth = np.array([[True, False, False]], dtype=np.bool_)
        observed = np.array([[False, True, False]], dtype=np.bool_)
        graph = build_factor_graph(observed, generate_clues(truth, observed))
        result = solve_exact(graph, 0.0)

        self.assertEqual(result.number_of_feasible_solutions, 2)
        self.assertFalse(result.has_posterior_mass)
        np.testing.assert_array_equal(result.posterior_probabilities, [0.0, 0.0])

    def test_rho_endpoints_select_the_supported_assignment(self) -> None:
        graph = unconstrained_graph((1, 3))
        zero = solve_exact(graph, 0.0)
        one = solve_exact(graph, 1.0)

        np.testing.assert_allclose(zero.marginals, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(one.marginals, [1.0, 1.0, 1.0])
        self.assertAlmostEqual(zero.partition_function, 1.0)
        self.assertAlmostEqual(one.partition_function, 1.0)

    def test_size_guard_prevents_accidental_large_enumeration(self) -> None:
        graph = unconstrained_graph((1, 4))
        with self.assertRaises(ExactEnumerationLimitError):
            solve_exact(graph, 0.5, max_variables=3)

    def test_replica_overlap_matches_direct_pair_sum(self) -> None:
        result = solve_exact(unconstrained_graph((1, 3)), 0.37)
        direct = np.zeros(4, dtype=np.float64)
        for state_a, probability_a in zip(
            result.feasible_states, result.posterior_probabilities
        ):
            for state_b, probability_b in zip(
                result.feasible_states, result.posterior_probabilities
            ):
                distance = int(np.count_nonzero(state_a != state_b))
                direct[distance] += probability_a * probability_b

        np.testing.assert_allclose(
            result.replica_overlap.probabilities, direct, rtol=1e-14, atol=1e-14
        )

    def test_inconsistent_planted_configuration_is_rejected(self) -> None:
        truth = np.array([[True, False, False]], dtype=np.bool_)
        observed = np.array([[False, True, False]], dtype=np.bool_)
        graph = build_factor_graph(observed, generate_clues(truth, observed))
        inconsistent = np.zeros((1, 3), dtype=np.bool_)

        with self.assertRaises(ValueError):
            solve_exact(graph, 0.2, planted_ground_truth=inconsistent)

    def test_arrays_in_result_are_immutable(self) -> None:
        result = solve_exact(unconstrained_graph((1, 1)), 0.5)
        with self.assertRaises(ValueError):
            result.marginals[0] = 0.0
        with self.assertRaises(ValueError):
            result.feasible_states[0, 0] = True


if __name__ == "__main__":
    unittest.main()
