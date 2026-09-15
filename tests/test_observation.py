import unittest

import numpy as np

from minesweeper_csp import (
    PlantedInstance,
    generate_clues,
    generate_ground_truth,
    generate_observation_mask,
)


class ObservationTests(unittest.TestCase):
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

    def test_all_safe_observes_exactly_safe_cells(self) -> None:
        observed = generate_observation_mask(self.instance, protocol="all_safe")
        np.testing.assert_array_equal(observed, ~self.instance.ground_truth)

    def test_bernoulli_safe_is_seeded_and_never_observes_mines(self) -> None:
        first = generate_observation_mask(
            self.instance,
            protocol="bernoulli_safe",
            observation_rate=0.5,
            seed=123,
        )
        second = generate_observation_mask(
            self.instance,
            protocol="bernoulli_safe",
            observation_rate=0.5,
            seed=123,
        )
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.any(first & self.instance.ground_truth))

    def test_clues_use_bounded_moore_neighborhood(self) -> None:
        observed = generate_observation_mask(self.instance, protocol="all_safe")
        clues = generate_clues(self.instance.ground_truth, observed)

        expected = np.array(
            [
                [-1, 2, 1],
                [1, 2, -1],
                [0, 1, 1],
            ]
        )
        np.testing.assert_array_equal(clues, expected)

    def test_mine_cannot_be_used_as_clue(self) -> None:
        invalid_mask = np.zeros((3, 3), dtype=np.bool_)
        invalid_mask[0, 0] = True
        with self.assertRaises(ValueError):
            generate_clues(self.instance.ground_truth, invalid_mask)

    def test_protocol_configuration_is_not_silently_ignored(self) -> None:
        with self.assertRaises(ValueError):
            generate_observation_mask(
                self.instance, protocol="all_safe", observation_rate=0.5
            )
        with self.assertRaises(ValueError):
            generate_observation_mask(
                self.instance, protocol="bernoulli_safe", observation_rate=-0.1
            )


if __name__ == "__main__":
    unittest.main()

