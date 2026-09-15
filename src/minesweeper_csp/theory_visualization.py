"""Static figures for finite-instance Bethe and RS diagnostics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .visualization import _panel


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _size_series(
    analysis: dict[str, Any], key: str
) -> list[tuple[str, list[tuple[float, float, float]]]]:
    series = []
    for size in analysis["sizes"]:
        values = []
        for point in size["points"]:
            statistics = point[key]
            mean = _finite(statistics.get("mean"))
            error = _finite(statistics.get("standard_error"))
            if mean is not None:
                values.append((float(point["rho"]), mean, error or 0.0))
        series.append((f"{size['Lx']}×{size['Ly']}", values))
    return series


def _positive_domain(
    series: list[tuple[str, list[tuple[float, float, float]]]],
    minimum: float,
    *,
    include: float | None = None,
) -> tuple[float, float]:
    upper = max(
        (value + error for _, values in series for _, value, error in values),
        default=minimum,
    )
    if include is not None:
        upper = max(upper, include)
    return (0.0, max(minimum, upper * 1.1))


def _data_domain(
    series: list[tuple[str, list[tuple[float, float, float]]]],
    minimum_span: float = 0.1,
) -> tuple[float, float]:
    lows = [value - error for _, values in series for _, value, error in values]
    highs = [value + error for _, values in series for _, value, error in values]
    if not lows:
        return (0.0, minimum_span)
    low, high = min(lows), max(highs)
    span = max(high - low, minimum_span)
    return (low - 0.08 * span, high + 0.08 * span)


def write_rs_theory_svg(analysis: dict[str, Any], path: str | Path) -> Path:
    """Write a four-panel finite-instance RS/Bethe diagnostic figure."""

    stability = _size_series(analysis, "spectral_radius")
    bethe_error = _size_series(
        analysis, "bethe_log_partition_error_per_variable"
    )
    entropy = _size_series(analysis, "bethe_entropy_density")
    short_loops = _size_series(analysis, "four_cycles_per_variable")
    panels = [
        _panel(
            20,
            50,
            570,
            360,
            "Local BP fixed-point stability",
            "Jacobian spectral radius",
            stability,
            _positive_domain(stability, 1.1, include=1.0),
            reference_y=1.0,
            reference_label="local threshold λ=1",
        ),
        _panel(
            610,
            50,
            570,
            360,
            "Bethe error against Exact",
            "|log Z_B − log Z| / variables",
            bethe_error,
            _positive_domain(bethe_error, 0.05),
        ),
        _panel(
            20,
            430,
            570,
            360,
            "Bethe entropy density",
            "H_B / variables",
            entropy,
            _data_domain(entropy),
        ),
        _panel(
            610,
            430,
            570,
            360,
            "Spatial short-loop density",
            "length-4 cycles / variables",
            short_loops,
            _positive_domain(short_loops, 0.1),
        ),
    ]
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="820" '
        'viewBox="0 0 1200 820" role="img" aria-labelledby="title desc">'
        '<title id="title">Finite-instance RS and Bethe diagnostics</title>'
        '<desc id="desc">System-size comparison of local BP Jacobian stability, '
        'Bethe partition-function error, Bethe entropy density, and spatial '
        'length-four loop density versus mine density.</desc>'
        '<rect width="1200" height="820" fill="#f8fafc"/>'
        '<text x="600" y="28" text-anchor="middle" font-family="Arial, sans-serif" '
        'font-size="20" font-weight="600" fill="#0f172a">Finite-instance RS / Bethe diagnostics</text>'
        '<g font-family="Arial, sans-serif" fill="#0f172a">'
        + "".join(panels)
        + "</g></svg>"
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(svg, encoding="utf-8", newline="\n")
    return destination
