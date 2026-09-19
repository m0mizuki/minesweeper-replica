from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
from pydantic import ValidationError


GAME_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("minesweeper_bp_game", GAME_ROOT / "main.py")
assert SPEC is not None and SPEC.loader is not None
game_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = game_module
SPEC.loader.exec_module(game_module)

Game = game_module.Game
GameStatus = game_module.GameStatus
SolveRequest = game_module.SolveRequest


class BernoulliGameTests(unittest.TestCase):
    def test_boundary_densities_generate_independent_extremes(self) -> None:
        self.assertEqual(Game.create(size=4, rho=0.0, seed=1).mines, set())
        self.assertEqual(len(Game.create(size=4, rho=1.0, seed=1).mines), 16)

    def test_seed_reproduces_the_planted_board(self) -> None:
        first = Game.create(size=9, rho=0.27, seed=123)
        second = Game.create(size=9, rho=0.27, seed=123)
        self.assertEqual(first.mines, second.mines)

    def test_first_click_is_not_conditioned_safe(self) -> None:
        game = Game.create(size=1, rho=1.0, seed=9)
        game.reveal(0, 0)
        self.assertEqual(game.status, GameStatus.lost)
        self.assertEqual(game.hit_cell, (0, 0))

    def test_bp_without_clues_returns_uniform_marginals(self) -> None:
        game = Game.create(size=3, rho=0.23, seed=7)
        game.predict_with_bp(SolveRequest())
        self.assertEqual(game.bp_summary.status, "converged")
        self.assertEqual(game.bp_summary.constraints, 0)
        np.testing.assert_allclose(list(game.mine_probabilities.values()), 0.5)

    def test_bp_uses_revealed_clue_but_not_flags_or_total_count(self) -> None:
        game = Game(size=3, rho=0.2, mines={(0, 0)})
        game.revealed = {(1, 1)}
        game.status = GameStatus.playing
        game.toggle_flag(2, 2)  # Deliberately wrong; flags are not evidence.
        game.predict_with_bp(SolveRequest(tolerance=1e-12))

        self.assertEqual(game.bp_summary.constraints, 1)
        self.assertEqual(len(game.mine_probabilities), 8)
        self.assertAlmostEqual(sum(game.mine_probabilities.values()), 1.0, places=9)
        self.assertAlmostEqual(game.mine_probabilities[(2, 2)], 1 / 8, places=9)

    def test_response_hides_realized_mine_count_until_game_over(self) -> None:
        game = Game(size=2, rho=0.5, mines={(0, 0)})
        self.assertIsNone(game.response("id").realized_mines)
        game.reveal(0, 0)
        self.assertEqual(game.response("id").realized_mines, 1)

    def test_probabilities_remain_without_rerunning_bp_after_reveal(self) -> None:
        game = Game(size=3, rho=0.2, mines={(0, 0)})
        game.predict_with_bp(SolveRequest(tolerance=1e-12))
        self.assertAlmostEqual(game.mine_probabilities[(0, 1)], 0.5)

        game.reveal(1, 1)

        self.assertEqual(game.status, GameStatus.playing)
        self.assertNotIn((1, 1), game.mine_probabilities)
        self.assertEqual(game.bp_summary.constraints, 0)
        self.assertAlmostEqual(game.mine_probabilities[(0, 1)], 0.5)


class GameApiTests(unittest.TestCase):
    def setUp(self) -> None:
        game_module.games.clear()

    def test_create_and_predict_api(self) -> None:
        created = game_module.new_game(
            game_module.NewGameRequest(size=3, rho=0.25, seed=42)
        )
        initial = created.model_dump(mode="json")
        self.assertEqual(initial["size"], 3)
        self.assertEqual(initial["rho"], 0.25)
        self.assertIsNone(initial["realized_mines"])
        self.assertNotIn("mines", initial)

        predicted = game_module.solve(
            initial["game_id"], game_module.SolveRequest()
        )
        payload = predicted.model_dump(mode="json")
        self.assertEqual(payload["bp"]["status"], "converged")
        self.assertEqual(payload["bp"]["constraints"], 0)
        probabilities = [
            cell["mine_probability"]
            for row in payload["cells"]
            for cell in row
        ]
        np.testing.assert_allclose(probabilities, 0.5)

    def test_invalid_new_game_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            game_module.NewGameRequest(size=0, rho=1.2)


if __name__ == "__main__":
    unittest.main()
