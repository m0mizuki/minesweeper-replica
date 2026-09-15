"""Generation of planted Bernoulli Minesweeper configurations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

BoolGrid = NDArray[np.bool_]


@dataclass(frozen=True)
class PlantedInstance:
    """A reproducible planted configuration from the Bernoulli ensemble.

    ``ground_truth[i, j]`` is true exactly when the cell contains a mine.
    The two axes have lengths ``Lx`` and ``Ly`` respectively.  No constraint
    is imposed on the realized total number of mines.
    """

    Lx: int
    Ly: int
    rho: float
    seed: int | None
    ground_truth: BoolGrid

    def __post_init__(self) -> None:
        if self.Lx <= 0 or self.Ly <= 0:
            raise ValueError("Lx and Ly must both be positive")
        if not 0.0 <= self.rho <= 1.0:
            raise ValueError("rho must lie in [0, 1]")

        ground_truth = np.asarray(self.ground_truth)
        if ground_truth.shape != (self.Lx, self.Ly):
            raise ValueError(
                "ground_truth shape must equal (Lx, Ly): "
                f"expected {(self.Lx, self.Ly)}, got {ground_truth.shape}"
            )
        if ground_truth.dtype != np.bool_:
            raise TypeError("ground_truth must have boolean dtype")

        # Own and freeze the data so metadata cannot silently diverge from it.
        frozen = ground_truth.copy()
        frozen.flags.writeable = False
        object.__setattr__(self, "ground_truth", frozen)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.Lx, self.Ly)

    @property
    def N(self) -> int:
        return self.Lx * self.Ly

    @property
    def number_of_mines(self) -> int:
        return int(np.count_nonzero(self.ground_truth))


def generate_ground_truth(
    Lx: int,
    Ly: int,
    rho: float,
    *,
    seed: int | None = None,
) -> PlantedInstance:
    """Draw every cell independently from Bernoulli(``rho``).

    This deliberately does not condition on a fixed total mine count.
    """

    if isinstance(Lx, bool) or not isinstance(Lx, (int, np.integer)) or Lx <= 0:
        raise ValueError("Lx must be a positive integer")
    if isinstance(Ly, bool) or not isinstance(Ly, (int, np.integer)) or Ly <= 0:
        raise ValueError("Ly must be a positive integer")
    if isinstance(rho, bool) or not isinstance(rho, (int, float, np.number)):
        raise TypeError("rho must be a real number")
    if not np.isfinite(rho) or not 0.0 <= float(rho) <= 1.0:
        raise ValueError("rho must lie in [0, 1]")

    rng = np.random.default_rng(seed)
    ground_truth = rng.random((int(Lx), int(Ly))) < float(rho)
    return PlantedInstance(
        Lx=int(Lx),
        Ly=int(Ly),
        rho=float(rho),
        seed=seed,
        ground_truth=ground_truth,
    )

