"""Interactive Minesweeper with Bernoulli planting and BP mine prediction."""

from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


# Prefer the repository source tree so this game always uses the BP implementation
# being developed alongside it.  Installing the project editable is still the
# recommended setup (see requirements.txt).
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from minesweeper_csp import BPConfig, build_factor_graph, run_bp  # noqa: E402


DEFAULT_SIZE = 9
DEFAULT_RHO = 10 / 81
MAX_SIZE = 30
STATIC_DIR = Path(__file__).parent / "static"
Cell = tuple[int, int]


class GameStatus(str, Enum):
    ready = "ready"
    playing = "playing"
    won = "won"
    lost = "lost"


class CellState(BaseModel):
    row: int
    col: int
    revealed: bool
    flagged: bool
    value: int | Literal["mine", "wrong-flag", "hit-mine"] | None
    mine_probability: float | None = None


class BPSummary(BaseModel):
    status: Literal["converged", "max_iterations", "infeasible"]
    iterations: int
    final_max_message_delta: float
    variables: int
    constraints: int


class GameResponse(BaseModel):
    game_id: str
    size: int
    rho: float
    flags: int
    status: GameStatus
    started: bool
    elapsed: int
    realized_mines: int | None
    bp: BPSummary | None
    cells: list[list[CellState]]


class NewGameRequest(BaseModel):
    size: int = Field(default=DEFAULT_SIZE, ge=1, le=MAX_SIZE)
    rho: float = Field(default=DEFAULT_RHO, ge=0.0, le=1.0)
    seed: int | None = None


class MoveRequest(BaseModel):
    row: int
    col: int


class SolveRequest(BaseModel):
    max_iterations: int = Field(default=500, ge=1, le=10_000)
    tolerance: float = Field(default=1e-9, gt=0.0)
    damping: float = Field(default=0.2, ge=0.0, lt=1.0)


@dataclass
class Game:
    """One game whose mine field is an unconditioned Bernoulli sample.

    Each cell is planted independently with probability ``rho``.  In
    particular, the first click is not forced safe and no fixed-mine-count
    condition is used by either generation or inference.
    """

    size: int
    rho: float
    mines: set[Cell]
    seed: int | None = None
    revealed: set[Cell] = field(default_factory=set)
    flagged: set[Cell] = field(default_factory=set)
    status: GameStatus = GameStatus.ready
    started_at: float | None = None
    finished_at: float | None = None
    hit_cell: Cell | None = None
    mine_probabilities: dict[Cell, float] = field(default_factory=dict)
    bp_summary: BPSummary | None = None
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @classmethod
    def create(
        cls,
        *,
        size: int = DEFAULT_SIZE,
        rho: float = DEFAULT_RHO,
        seed: int | None = None,
    ) -> "Game":
        if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= MAX_SIZE:
            raise ValueError(f"size must be an integer in [1, {MAX_SIZE}]")
        if isinstance(rho, bool) or not isinstance(rho, (int, float)):
            raise TypeError("rho must be a real number")
        if not math.isfinite(float(rho)) or not 0.0 <= float(rho) <= 1.0:
            raise ValueError("rho must lie in [0, 1]")

        rng = np.random.default_rng(seed)
        planted = rng.random((size, size)) < float(rho)
        mines = {
            (int(row), int(col))
            for row, col in np.argwhere(planted)
        }
        return cls(size=size, rho=float(rho), mines=mines, seed=seed)

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size

    def neighbors(self, row: int, col: int) -> list[Cell]:
        return [
            (next_row, next_col)
            for next_row in range(max(0, row - 1), min(self.size, row + 2))
            for next_col in range(max(0, col - 1), min(self.size, col + 2))
            if (next_row, next_col) != (row, col)
        ]

    def adjacent_mines(self, row: int, col: int) -> int:
        return sum(cell in self.mines for cell in self.neighbors(row, col))

    def reveal(self, row: int, col: int) -> None:
        self.validate_cell(row, col)
        if self.status in {GameStatus.won, GameStatus.lost} or (row, col) in self.flagged:
            return

        if self.status is GameStatus.ready:
            self.status = GameStatus.playing
            self.started_at = time.monotonic()

        previous_revealed = len(self.revealed)
        previous_status = self.status
        if (row, col) in self.revealed:
            self.chord(row, col)
        elif (row, col) in self.mines:
            self.hit_cell = (row, col)
            self.finish(GameStatus.lost)
        else:
            self.flood_reveal(row, col)
            self.check_win()

        if len(self.revealed) != previous_revealed or self.status is not previous_status:
            self.clear_bp_prediction()

    def chord(self, row: int, col: int) -> None:
        neighbors = self.neighbors(row, col)
        flagged_neighbors = sum(cell in self.flagged for cell in neighbors)
        if self.adjacent_mines(row, col) != flagged_neighbors:
            return

        for next_row, next_col in neighbors:
            cell = (next_row, next_col)
            if cell in self.revealed or cell in self.flagged:
                continue
            if cell in self.mines:
                self.hit_cell = cell
                self.finish(GameStatus.lost)
                return
            self.flood_reveal(next_row, next_col)
        self.check_win()

    def flood_reveal(self, row: int, col: int) -> None:
        stack = [(row, col)]
        while stack:
            cell = stack.pop()
            if cell in self.revealed or cell in self.flagged or cell in self.mines:
                continue
            self.revealed.add(cell)
            if self.adjacent_mines(*cell) == 0:
                stack.extend(self.neighbors(*cell))

    def toggle_flag(self, row: int, col: int) -> None:
        self.validate_cell(row, col)
        cell = (row, col)
        if self.status in {GameStatus.won, GameStatus.lost} or cell in self.revealed:
            return
        if cell in self.flagged:
            self.flagged.remove(cell)
        else:
            self.flagged.add(cell)

    def predict_with_bp(self, request: SolveRequest) -> None:
        """Infer hidden-cell marginals from revealed clues using sum-product BP.

        Flags are deliberately ignored: they are UI annotations, not evidence.
        The factor graph has no global factor for the realized number of mines.
        """

        if self.status in {GameStatus.won, GameStatus.lost}:
            raise HTTPException(status_code=400, detail="Finished games cannot be solved.")

        observed = np.zeros((self.size, self.size), dtype=np.bool_)
        clues = np.full((self.size, self.size), -1, dtype=np.int_)
        for cell in self.revealed:
            observed[cell] = True
            clues[cell] = self.adjacent_mines(*cell)

        graph = build_factor_graph(observed, clues)
        result = run_bp(
            graph,
            self.rho,
            config=BPConfig(
                max_iterations=request.max_iterations,
                tolerance=request.tolerance,
                damping=request.damping,
                initialization="prior",
            ),
        )

        self.bp_summary = BPSummary(
            status=result.status,
            iterations=result.iterations,
            final_max_message_delta=result.final_max_message_delta,
            variables=graph.number_of_variables,
            constraints=graph.number_of_constraints,
        )
        if not np.all(np.isfinite(result.marginals)):
            self.mine_probabilities = {}
            raise HTTPException(
                status_code=409,
                detail=result.failure_reason or "BP could not produce finite marginals.",
            )

        self.mine_probabilities = {
            cell: float(result.marginals[index])
            for index, cell in enumerate(graph.variable_cells)
        }

    def clear_bp_prediction(self) -> None:
        self.mine_probabilities.clear()
        self.bp_summary = None

    def check_win(self) -> None:
        if len(self.revealed) == self.size * self.size - len(self.mines):
            self.flagged = set(self.mines)
            self.finish(GameStatus.won)

    def finish(self, status: GameStatus) -> None:
        self.status = status
        self.finished_at = time.monotonic()

    def elapsed(self) -> int:
        if self.started_at is None:
            return 0
        return min(999, int((self.finished_at or time.monotonic()) - self.started_at))

    def validate_cell(self, row: int, col: int) -> None:
        if not self.in_bounds(row, col):
            raise HTTPException(status_code=400, detail="Cell is outside the board.")

    def response(self, game_id: str) -> GameResponse:
        game_over = self.status in {GameStatus.won, GameStatus.lost}
        cells: list[list[CellState]] = []
        for row in range(self.size):
            view_row: list[CellState] = []
            for col in range(self.size):
                cell = (row, col)
                is_revealed = cell in self.revealed
                is_flagged = cell in self.flagged
                value: int | Literal["mine", "wrong-flag", "hit-mine"] | None = None
                if is_revealed:
                    value = self.adjacent_mines(row, col)
                elif game_over and cell == self.hit_cell:
                    value = "hit-mine"
                elif game_over and cell in self.mines:
                    value = "mine"
                elif game_over and is_flagged:
                    value = "wrong-flag"

                view_row.append(
                    CellState(
                        row=row,
                        col=col,
                        revealed=is_revealed,
                        flagged=is_flagged,
                        value=value,
                        mine_probability=self.mine_probabilities.get(cell),
                    )
                )
            cells.append(view_row)

        return GameResponse(
            game_id=game_id,
            size=self.size,
            rho=self.rho,
            flags=len(self.flagged),
            status=self.status,
            started=self.started_at is not None,
            elapsed=self.elapsed(),
            realized_mines=len(self.mines) if game_over else None,
            bp=self.bp_summary,
            cells=cells,
        )


app = FastAPI(title="Bernoulli Minesweeper with Belief Propagation")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
games: dict[str, Game] = {}
games_lock = threading.Lock()


def get_game(game_id: str) -> Game:
    with games_lock:
        game = games.get(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="Game not found.")
    return game


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/new", response_model=GameResponse)
def new_game(request: NewGameRequest | None = None) -> GameResponse:
    settings = request or NewGameRequest()
    game = Game.create(size=settings.size, rho=settings.rho, seed=settings.seed)
    game_id = uuid4().hex
    with games_lock:
        games[game_id] = game
    return game.response(game_id)


@app.post("/api/{game_id}/reveal", response_model=GameResponse)
def reveal(game_id: str, move: MoveRequest) -> GameResponse:
    game = get_game(game_id)
    with game.lock:
        game.reveal(move.row, move.col)
        return game.response(game_id)


@app.post("/api/{game_id}/flag", response_model=GameResponse)
def flag(game_id: str, move: MoveRequest) -> GameResponse:
    game = get_game(game_id)
    with game.lock:
        game.toggle_flag(move.row, move.col)
        return game.response(game_id)


@app.post("/api/{game_id}/solve", response_model=GameResponse)
def solve(game_id: str, request: SolveRequest | None = None) -> GameResponse:
    game = get_game(game_id)
    with game.lock:
        game.predict_with_bp(request or SolveRequest())
        return game.response(game_id)
