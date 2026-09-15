import unittest

import numpy as np

from minesweeper_csp import (
    PlantedInstance,
    assert_csp_consistent,
    build_factor_graph,
    check_csp_consistency,
    constraint_residuals,
    generate_clues,
    generate_ground_truth,
)


class FactorGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        truth = np.array(
            [
                [True, False, False],
                [False, False, True],
                [False, False, False],
            ],
            dtype=np.bool_,
        )
        self.instance = PlantedInstance(3, 3, 2 / 9, 10, truth)
        self.observed = np.array(
            [
                [False, True, False],
                [False, True, False],
                [False, False, True],
            ],
            dtype=np.bool_,
        )
        self.clues = generate_clues(truth, self.observed)
        self.graph = build_factor_graph(self.observed, self.clues)

    def test_graph_partition_and_adjacency(self) -> None:
        self.assertEqual(self.graph.number_of_variables, 6)
        self.assertEqual(self.graph.number_of_constraints, 3)
        self.assertEqual(set(self.graph.variable_cells), set(map(tuple, np.argwhere(~self.observed))))
        self.assertEqual(set(self.graph.clue_cells), set(map(tuple, np.argwhere(self.observed))))

        for factor, variables in enumerate(self.graph.factor_to_variables):
            clue_cell = self.graph.clue_cells[factor]
            for variable in variables:
                variable_cell = self.graph.variable_cells[variable]
                self.assertLessEqual(abs(clue_cell[0] - variable_cell[0]), 1)
                self.assertLessEqual(abs(clue_cell[1] - variable_cell[1]), 1)
                self.assertIn(factor, self.graph.variable_to_factors[variable])

    def test_planted_configuration_satisfies_every_constraint(self) -> None:
        self.assertTrue(check_csp_consistency(self.graph, self.instance.ground_truth))
        assert_csp_consistent(self.graph, self.instance.ground_truth)

        assignment = np.array(
            [int(self.instance.ground_truth[cell]) for cell in self.graph.variable_cells]
        )
        np.testing.assert_array_equal(
            constraint_residuals(self.graph, assignment),
            np.zeros(self.graph.number_of_constraints, dtype=int),
        )

    def test_modified_assignment_can_violate_constraints(self) -> None:
        assignment = np.array(
            [int(self.instance.ground_truth[cell]) for cell in self.graph.variable_cells]
        )
        assignment[0] = 1 - assignment[0]
        self.assertTrue(np.any(constraint_residuals(self.graph, assignment) != 0))

    def test_corrupted_clue_is_detected(self) -> None:
        corrupted = self.clues.copy()
        corrupted.flags.writeable = True
        cell = self.graph.clue_cells[0]
        corrupted[cell] += 1
        graph = build_factor_graph(self.observed, corrupted)

        self.assertFalse(check_csp_consistency(graph, self.instance.ground_truth))
        with self.assertRaises(ValueError):
            assert_csp_consistent(graph, self.instance.ground_truth)

    def test_empty_observation_is_valid(self) -> None:
        instance = generate_ground_truth(2, 3, 0.5, seed=9)
        observed = np.zeros(instance.shape, dtype=np.bool_)
        clues = generate_clues(instance.ground_truth, observed)
        graph = build_factor_graph(observed, clues)

        self.assertEqual(graph.number_of_variables, 6)
        self.assertEqual(graph.number_of_constraints, 0)
        self.assertTrue(check_csp_consistency(graph, instance.ground_truth))


if __name__ == "__main__":
    unittest.main()
