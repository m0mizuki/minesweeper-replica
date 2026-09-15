"""Observation masks and clue generation, kept separate from planting."""

from __future__ import annotations

from typing import Iterator, Literal

import numpy as np
from numpy.typing import NDArray

from .instance import PlantedInstance

BoolGrid = NDArray[np.bool_]
IntGrid = NDArray[np.int_]
Cell = tuple[int, int]
ObservationProtocol = Literal["all_safe", "bernoulli_safe"]


def _as_bool_grid(values: NDArray[np.bool_], *, name: str) -> BoolGrid:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional grid")
    if array.dtype != np.bool_:
        raise TypeError(f"{name} must have boolean dtype")
    return array


def iter_neighbor_cells(cell: Cell, shape: tuple[int, int]) -> Iterator[Cell]:
    """Yield in-bounds Moore-neighbor cells, excluding ``cell`` itself."""

    i, j = cell
    Lx, Ly = shape
    if not (0 <= i < Lx and 0 <= j < Ly):
        raise IndexError(f"cell {cell} is outside grid with shape {shape}")
    for ni in range(max(0, i - 1), min(Lx, i + 2)):
        for nj in range(max(0, j - 1), min(Ly, j + 2)):
            if (ni, nj) != cell:
                yield (ni, nj)


def generate_observation_mask(
    instance: PlantedInstance,
    *,
    protocol: ObservationProtocol = "all_safe",
    observation_rate: float = 1.0,
    seed: int | None = None,
) -> BoolGrid:
    """Choose clue cells independently of ground-truth generation.

    Returned true entries are observed clue cells.  Both supported protocols
    only observe safe cells.  ``bernoulli_safe`` selects every safe cell
    independently with probability ``observation_rate``.
    """

    if protocol not in ("all_safe", "bernoulli_safe"):
        raise ValueError(f"unknown observation protocol: {protocol!r}")
    if isinstance(observation_rate, bool) or not isinstance(
        observation_rate, (int, float, np.number)
    ):
        raise TypeError("observation_rate must be a real number")
    if not np.isfinite(observation_rate) or not 0.0 <= float(observation_rate) <= 1.0:
        raise ValueError("observation_rate must lie in [0, 1]")
    if protocol == "all_safe" and float(observation_rate) != 1.0:
        raise ValueError("all_safe requires observation_rate=1.0")

    safe = ~instance.ground_truth
    if protocol == "all_safe":
        observed = safe.copy()
    else:
        rng = np.random.default_rng(seed)
        observed = safe & (rng.random(instance.shape) < float(observation_rate))

    observed.flags.writeable = False
    return observed


def generate_clues(ground_truth: BoolGrid, observation_mask: BoolGrid) -> IntGrid:
    """Generate clue values for observed safe cells.

    Unobserved cells receive the sentinel value ``-1``.  A clue is the number
    of mines among all in-bounds Moore neighbors in the planted configuration.
    """

    truth = _as_bool_grid(ground_truth, name="ground_truth")
    observed = _as_bool_grid(observation_mask, name="observation_mask")
    if observed.shape != truth.shape:
        raise ValueError("ground_truth and observation_mask must have equal shapes")
    if np.any(observed & truth):
        raise ValueError("a mine cell cannot be observed as a clue")

    clues = np.full(truth.shape, -1, dtype=np.int_)
    for raw_cell in np.argwhere(observed):
        cell = (int(raw_cell[0]), int(raw_cell[1]))
        clues[cell] = sum(bool(truth[neighbor]) for neighbor in iter_neighbor_cells(cell, truth.shape))

    clues.flags.writeable = False
    return clues

