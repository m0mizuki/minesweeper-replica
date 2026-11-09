"""Bernoulli prior weights, including deterministic density endpoints."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def log_prior_weights(states: NDArray, rho: float) -> NDArray[np.float64]:
    """Log product-prior mass for each row; the empty product is one."""

    counts = states.sum(axis=1, dtype=np.int64)
    size = states.shape[1]
    if rho == 0.0:
        return np.where(counts == 0, 0.0, -math.inf)
    if rho == 1.0:
        return np.where(counts == size, 0.0, -math.inf)
    return counts * math.log(rho) + (size - counts) * math.log1p(-rho)


def normalized_prior_weights(
    states: NDArray, rho: float
) -> tuple[NDArray[np.float64], float]:
    """Normalize prior mass on candidate states using log-sum-exp.

    A zero-mass set returns zero probabilities and log normalizer -infinity.
    """

    logs = log_prior_weights(states, rho)
    maximum = float(np.max(logs)) if logs.size else -math.inf
    if not math.isfinite(maximum):
        return np.zeros(states.shape[0], dtype=np.float64), -math.inf
    weights = np.exp(logs - maximum)
    total = float(weights.sum())
    return weights / total, maximum + math.log(total)
