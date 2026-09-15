"""Statistical diagnostics shared by Monte Carlo experiments."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def integrated_autocorrelation_time(trace: ArrayLike) -> float:
    """Estimate integrated autocorrelation time by the initial-positive sequence.

    A constant trace is assigned ``tau=1``.  This convention is useful for
    exactly fixed observables, but should always be interpreted together with
    update-change rates and independent-chain diagnostics because a stuck chain
    is also constant.
    """

    values = np.asarray(trace, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("trace must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("trace must contain only finite values")
    sample_count = values.size
    if sample_count < 2:
        return math.nan

    centered = values - values.mean()
    variance = float(centered @ centered / sample_count)
    if variance <= np.finfo(np.float64).eps:
        return 1.0

    transform_size = 1 << (2 * sample_count - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=transform_size)
    autocovariance = np.fft.irfft(spectrum * spectrum.conjugate(), n=transform_size)[
        :sample_count
    ]
    autocovariance /= np.arange(sample_count, 0, -1, dtype=np.float64)
    autocorrelation = autocovariance / autocovariance[0]

    tau = 1.0
    lag = 1
    while lag + 1 < sample_count:
        pair_sum = float(autocorrelation[lag] + autocorrelation[lag + 1])
        if pair_sum <= 0.0:
            break
        tau += 2.0 * pair_sum
        lag += 2
    return min(max(tau, 1.0), float(sample_count))


def effective_sample_size(trace: ArrayLike) -> float:
    """Return sample count divided by estimated autocorrelation time."""

    values = np.asarray(trace)
    if values.ndim != 1:
        raise ValueError("trace must be one-dimensional")
    tau = integrated_autocorrelation_time(values)
    if not math.isfinite(tau):
        return math.nan
    return float(values.size / tau)


def gelman_rubin_r_hat(chains: ArrayLike) -> FloatArray:
    """Compute classic between/within-chain R-hat for each trailing observable.

    Input shape is ``(number_of_chains, samples, ...)``.  If all chains are
    identically constant R-hat is one; if constant chains disagree it is
    infinite, explicitly flagging initialization dependence.
    """

    values = np.asarray(chains, dtype=np.float64)
    if values.ndim < 2:
        raise ValueError("chains must have shape (chains, samples, ...)")
    chain_count, sample_count = values.shape[:2]
    if chain_count < 2:
        raise ValueError("at least two chains are required for R-hat")
    output_shape = values.shape[2:]
    if sample_count < 2:
        return np.full(output_shape, np.nan, dtype=np.float64)

    chain_means = values.mean(axis=1)
    within = values.var(axis=1, ddof=1).mean(axis=0)
    between = sample_count * chain_means.var(axis=0, ddof=1)
    variance_estimate = ((sample_count - 1) / sample_count) * within + (
        between / sample_count
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        r_hat = np.sqrt(variance_estimate / within)
    both_zero = (within == 0.0) & (between == 0.0)
    disagreeing_constants = (within == 0.0) & (between > 0.0)
    r_hat = np.where(both_zero, 1.0, r_hat)
    r_hat = np.where(disagreeing_constants, math.inf, r_hat)
    return np.asarray(r_hat, dtype=np.float64)

