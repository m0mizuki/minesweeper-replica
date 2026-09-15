import unittest

import numpy as np

from minesweeper_csp import generate_ground_truth


class GroundTruthTests(unittest.TestCase):
    def test_seed_reproduces_independent_bernoulli_draws(self) -> None:
        first = generate_ground_truth(7, 9, 0.31, seed=42)
        second = generate_ground_truth(7, 9, 0.31, seed=42)

        np.testing.assert_array_equal(first.ground_truth, second.ground_truth)
        self.assertEqual(first.shape, (7, 9))
        self.assertEqual(first.N, 63)
        self.assertEqual(first.number_of_mines, int(first.ground_truth.sum()))

    def test_rho_boundaries(self) -> None:
        self.assertFalse(generate_ground_truth(3, 4, 0.0, seed=1).ground_truth.any())
        self.assertTrue(generate_ground_truth(3, 4, 1.0, seed=1).ground_truth.all())

    def test_generated_grid_is_immutable(self) -> None:
        instance = generate_ground_truth(2, 2, 0.5, seed=1)
        with self.assertRaises(ValueError):
            instance.ground_truth[0, 0] = True

    def test_invalid_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            generate_ground_truth(0, 2, 0.5)
        with self.assertRaises(ValueError):
            generate_ground_truth(2, 2, 1.01)
        with self.assertRaises(TypeError):
            generate_ground_truth(2, 2, True)


if __name__ == "__main__":
    unittest.main()

