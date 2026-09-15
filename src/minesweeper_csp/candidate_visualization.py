"""Static scientific figures for focused candidate-region studies."""

from __future__ import annotations

from html import escape
import math
from pathlib import Path
from typing import Any

from .visualization import COLORS, _panel


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _size_series(
    analysis: dict[str, Any],
    key: str,
    *,
    nested: tuple[str, ...] = (),
) -> list[tuple[str, list[tuple[float, float, float]]]]:
    series = []
    for size in analysis["sizes"]:
        values = []
        for point in size["points"]:
            statistics: Any = point[key]
            for child in nested:
                statistics = statistics[child]
            mean = _finite(statistics.get("mean"))
            error = _finite(statistics.get("standard_error"))
            if mean is not None:
                values.append((float(point["rho"]), mean, error or 0.0))
        series.append((f"{size['Lx']}×{size['Ly']}", values))
    return series


def _positive_domain(
    series: list[tuple[str, list[tuple[float, float, float]]]], minimum: float
) -> tuple[float, float]:
    upper = max(
        (value + error for _, values in series for _, value, error in values),
        default=minimum,
    )
    return (0.0, max(minimum, upper * 1.1))


def write_candidate_summary_svg(analysis: dict[str, Any], path: str | Path) -> Path:
    """Plot size dependence of four candidate diagnostics."""

    discrepancy = _size_series(analysis, "bp_mcmc_mae")
    fixed_points = _size_series(analysis, "bp_fixed_point_spread")
    relaxation = _size_series(analysis, "mcmc_overlap_tau")
    overlap_width = _size_series(
        analysis, "mcmc_overlap", nested=("width_across_disorder",)
    )
    panels = [
        _panel(
            20,
            50,
            570,
            360,
            "BP–MCMC disagreement",
            "marginal MAE",
            discrepancy,
            _positive_domain(discrepancy, 0.05),
        ),
        _panel(
            610,
            50,
            570,
            360,
            "BP fixed-point dependence",
            "max marginal spread",
            fixed_points,
            _positive_domain(fixed_points, 0.05),
        ),
        _panel(
            20,
            430,
            570,
            360,
            "MCMC relaxation",
            "planted-overlap autocorrelation time",
            relaxation,
            _positive_domain(relaxation, 1.0),
        ),
        _panel(
            610,
            430,
            570,
            360,
            "Replica-overlap breadth",
            "q95 − q05",
            overlap_width,
            (0.0, 2.0),
        ),
    ]
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="820" '
        'viewBox="0 0 1200 820" role="img" aria-labelledby="title desc">'
        '<title id="title">Candidate-region finite-size analysis</title>'
        '<desc id="desc">Size-dependent BP disagreement, fixed-point spread, MCMC '
        'relaxation, and replica-overlap width versus mine density.</desc>'
        '<rect width="1200" height="820" fill="#f8fafc"/>'
        '<text x="600" y="28" text-anchor="middle" font-family="Arial, sans-serif" '
        'font-size="20" font-weight="600" fill="#0f172a">Candidate-region finite-size analysis</text>'
        '<g font-family="Arial, sans-serif" fill="#0f172a">'
        + "".join(panels)
        + "</g></svg>"
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(svg, encoding="utf-8", newline="\n")
    return destination


def _selected_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indices = sorted({0, len(points) // 2, len(points) - 1})
    return [points[index] for index in indices]


def _distribution_panel(
    x: float,
    y: float,
    width: float,
    height: float,
    size: dict[str, Any],
) -> str:
    points = _selected_points(size["points"])
    left, right, top, bottom = 65, 18, 48, 52
    px, py = x + left, y + top
    pw, ph = width - left - right, height - top - bottom
    maximum = max(
        (
            probability
            for point in points
            for probability in point["mcmc_overlap"]["probabilities"]
        ),
        default=1.0,
    )
    for point in points:
        exact = point.get("exact_overlap")
        if exact is not None:
            maximum = max(maximum, max(exact["probabilities"], default=0.0))
    maximum = max(maximum * 1.08, 0.05)

    def sx(value: float) -> float:
        return px + (value + 1.0) * pw / 2.0

    def sy(value: float) -> float:
        return py + ph - value / maximum * ph

    chunks = [
        f'<g><text x="{x + width / 2:.1f}" y="{y + 20:.1f}" text-anchor="middle" '
        f'font-size="15" font-weight="600">{size["Lx"]}×{size["Ly"]}</text>',
        f'<rect x="{px:.1f}" y="{py:.1f}" width="{pw:.1f}" height="{ph:.1f}" '
        'fill="white" stroke="#94a3b8"/>',
    ]
    for index in range(5):
        value = index * maximum / 4.0
        position = sy(value)
        chunks.append(
            f'<line x1="{px:.1f}" x2="{px + pw:.1f}" y1="{position:.1f}" '
            f'y2="{position:.1f}" stroke="#e2e8f0"/>'
            f'<text x="{px - 7:.1f}" y="{position + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#475569">{value:.2g}</text>'
        )
    for value in (-1.0, -0.5, 0.0, 0.5, 1.0):
        position = sx(value)
        chunks.append(
            f'<line x1="{position:.1f}" x2="{position:.1f}" y1="{py + ph:.1f}" '
            f'y2="{py + ph + 5:.1f}" stroke="#64748b"/>'
            f'<text x="{position:.1f}" y="{py + ph + 20:.1f}" text-anchor="middle" '
            f'font-size="11" fill="#475569">{value:g}</text>'
        )
    chunks.append(
        f'<text x="{px + pw / 2:.1f}" y="{y + height - 7:.1f}" text-anchor="middle" '
        'font-size="12">replica overlap q</text>'
        f'<text x="{x + 13:.1f}" y="{py + ph / 2:.1f}" text-anchor="middle" '
        f'font-size="12" transform="rotate(-90 {x + 13:.1f} {py + ph / 2:.1f})">P(q)</text>'
    )
    for index, point in enumerate(points):
        color = COLORS[index % len(COLORS)]
        centers = point["mcmc_overlap"]["bin_centers"]
        masses = point["mcmc_overlap"]["probabilities"]
        coordinates = " ".join(
            f"{sx(float(q)):.1f},{sy(float(mass)):.1f}"
            for q, mass in zip(centers, masses)
        )
        legend_y = py + 14 + index * 16
        chunks.append(
            f'<line x1="{px + 6:.1f}" x2="{px + 22:.1f}" y1="{legend_y - 4:.1f}" '
            f'y2="{legend_y - 4:.1f}" stroke="{color}" stroke-width="2"/>'
            f'<text x="{px + 27:.1f}" y="{legend_y:.1f}" font-size="11" '
            f'fill="#334155">ρ={float(point["rho"]):.3g}</text>'
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        exact = point.get("exact_overlap")
        if exact is not None:
            exact_coordinates = " ".join(
                f"{sx(float(q)):.1f},{sy(float(mass)):.1f}"
                for q, mass in zip(
                    exact["bin_centers"], exact["probabilities"]
                )
            )
            chunks.append(
                f'<polyline points="{exact_coordinates}" fill="none" stroke="{color}" '
                'stroke-width="1.5" stroke-dasharray="5 4"/>'
            )
    chunks.append(
        f'<text x="{px + pw - 5:.1f}" y="{py + 14:.1f}" text-anchor="end" '
        'font-size="11" fill="#475569">solid MCMC · dashed Exact</text></g>'
    )
    return "".join(chunks)


def write_overlap_distributions_svg(
    analysis: dict[str, Any], path: str | Path
) -> Path:
    """Plot representative full P(q) curves for every system size."""

    columns = 2
    panel_width, panel_height = 590, 330
    rows = math.ceil(len(analysis["sizes"]) / columns)
    height = 50 + rows * panel_height
    panels = []
    for index, size in enumerate(analysis["sizes"]):
        column, row = index % columns, index // columns
        panels.append(
            _distribution_panel(
                10 + column * panel_width,
                40 + row * panel_height,
                panel_width - 10,
                panel_height - 10,
                size,
            )
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" '
        f'viewBox="0 0 1200 {height}" role="img" aria-labelledby="title desc">'
        '<title id="title">Replica-overlap distributions</title>'
        '<desc id="desc">MCMC and Exact replica-overlap distributions at representative '
        'mine densities for each system size.</desc>'
        f'<rect width="1200" height="{height}" fill="#f8fafc"/>'
        '<text x="600" y="27" text-anchor="middle" font-family="Arial, sans-serif" '
        'font-size="20" font-weight="600" fill="#0f172a">Replica-overlap distributions</text>'
        '<g font-family="Arial, sans-serif" fill="#0f172a">'
        + "".join(panels)
        + "</g></svg>"
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(svg, encoding="utf-8", newline="\n")
    return destination


def write_candidate_figures(analysis: dict[str, Any], output: str | Path) -> None:
    output = Path(output)
    write_candidate_summary_svg(analysis, output / "candidate_summary.svg")
    write_overlap_distributions_svg(analysis, output / "overlap_distributions.svg")
