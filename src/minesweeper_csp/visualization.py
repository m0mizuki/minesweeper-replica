"""Dependency-free SVG plots for aggregated phase scans."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any, Sequence


COLORS = ("#2563eb", "#dc2626", "#059669", "#7c3aed")


def _series(
    aggregate: dict[str, Any], metric: str
) -> list[tuple[float, float, float]]:
    result = []
    for point in aggregate["parameter_points"]:
        statistics = point["metrics"][metric]
        if statistics["mean"] is not None:
            result.append(
                (
                    float(point["rho"]),
                    float(statistics["mean"]),
                    float(statistics["standard_error"]),
                )
            )
    return result


def _ticks(low: float, high: float, count: int = 5) -> list[float]:
    if low == high:
        return [low]
    return [low + index * (high - low) / (count - 1) for index in range(count)]


def _panel(
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    y_label: str,
    series: Sequence[tuple[str, list[tuple[float, float, float]]]],
    y_domain: tuple[float, float] | None = None,
    reference_y: float | None = None,
    reference_label: str | None = None,
) -> str:
    available = [(label, values) for label, values in series if values]
    margin_left, margin_right, margin_top, margin_bottom = 72, 20, 46, 58
    plot_x = x + margin_left
    plot_y = y + margin_top
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    chunks = [
        f'<g><text x="{x + width / 2:.1f}" y="{y + 20:.1f}" text-anchor="middle" '
        f'font-size="15" font-weight="600">{escape(title)}</text>',
        f'<rect x="{plot_x:.1f}" y="{plot_y:.1f}" width="{plot_width:.1f}" '
        f'height="{plot_height:.1f}" fill="white" stroke="#94a3b8"/>',
    ]
    if not available:
        chunks.append(
            f'<text x="{plot_x + plot_width / 2:.1f}" y="{plot_y + plot_height / 2:.1f}" '
            'text-anchor="middle" fill="#64748b">no data</text></g>'
        )
        return "".join(chunks)

    all_points = [point for _, values in available for point in values]
    x_low = min(point[0] for point in all_points)
    x_high = max(point[0] for point in all_points)
    y_low = min(point[1] - point[2] for point in all_points)
    y_high = max(point[1] + point[2] for point in all_points)
    if x_low == x_high:
        x_low, x_high = x_low - 0.05, x_high + 0.05
    if y_domain is None:
        y_padding = (
            0.08 * (y_high - y_low)
            if y_high > y_low
            else max(0.05, abs(y_high) * 0.1)
        )
        y_low -= y_padding
        y_high += y_padding
    else:
        y_low, y_high = y_domain

    def sx(value: float) -> float:
        return plot_x + (value - x_low) / (x_high - x_low) * plot_width

    def sy(value: float) -> float:
        return plot_y + plot_height - (value - y_low) / (y_high - y_low) * plot_height

    for tick in _ticks(y_low, y_high):
        position = sy(tick)
        chunks.append(
            f'<line x1="{plot_x:.1f}" x2="{plot_x + plot_width:.1f}" '
            f'y1="{position:.1f}" y2="{position:.1f}" stroke="#e2e8f0"/>'
            f'<text x="{plot_x - 8:.1f}" y="{position + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#475569">{tick:.3g}</text>'
        )
    for tick in _ticks(x_low, x_high):
        position = sx(tick)
        chunks.append(
            f'<line x1="{position:.1f}" x2="{position:.1f}" '
            f'y1="{plot_y + plot_height:.1f}" y2="{plot_y + plot_height + 5:.1f}" '
            'stroke="#64748b"/>'
            f'<text x="{position:.1f}" y="{plot_y + plot_height + 20:.1f}" '
            f'text-anchor="middle" font-size="11" fill="#475569">{tick:.3g}</text>'
        )
    chunks.append(
        f'<text x="{plot_x + plot_width / 2:.1f}" y="{y + height - 8:.1f}" '
        'text-anchor="middle" font-size="12">mine density ρ</text>'
        f'<text x="{x + 14:.1f}" y="{plot_y + plot_height / 2:.1f}" '
        f'text-anchor="middle" font-size="12" transform="rotate(-90 {x + 14:.1f} '
        f'{plot_y + plot_height / 2:.1f})">{escape(y_label)}</text>'
    )

    if reference_y is not None and y_low <= reference_y <= y_high:
        position = sy(reference_y)
        chunks.append(
            f'<line x1="{plot_x:.1f}" x2="{plot_x + plot_width:.1f}" '
            f'y1="{position:.1f}" y2="{position:.1f}" stroke="#64748b" '
            'stroke-width="1.5" stroke-dasharray="6 5"/>'
        )
        if reference_label:
            chunks.append(
                f'<text x="{plot_x + plot_width - 5:.1f}" y="{position - 6:.1f}" '
                f'text-anchor="end" font-size="11" fill="#475569">'
                f'{escape(reference_label)}</text>'
            )

    legend_x = plot_x + 6
    for index, (label, values) in enumerate(available):
        color = COLORS[index % len(COLORS)]
        legend_y = plot_y + 15 + index * 16
        chunks.append(
            f'<line x1="{legend_x:.1f}" x2="{legend_x + 18:.1f}" '
            f'y1="{legend_y - 4:.1f}" y2="{legend_y - 4:.1f}" '
            f'stroke="{color}" stroke-width="2"/>'
            f'<text x="{legend_x + 24:.1f}" y="{legend_y:.1f}" font-size="11" '
            f'fill="#334155">{escape(label)}</text>'
        )
        coordinates = [
            (
                sx(px),
                sy(py),
                sy(max(y_low, py - error)),
                sy(min(y_high, py + error)),
            )
            for px, py, error in values
        ]
        chunks.append(
            f'<polyline points="{" ".join(f"{px:.1f},{py:.1f}" for px, py, _, _ in coordinates)}" '
            f'fill="none" stroke="{color}" stroke-width="2"/>'
        )
        for px, py, error_low, error_high in coordinates:
            chunks.append(
                f'<line x1="{px:.1f}" x2="{px:.1f}" y1="{error_low:.1f}" '
                f'y2="{error_high:.1f}" stroke="{color}"/>'
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="{color}"/>'
            )
    chunks.append("</g>")
    return "".join(chunks)


def write_phase_scan_svg(aggregate: dict[str, Any], path: str | Path) -> Path:
    """Write a four-panel scientific summary of an aggregate result."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        _panel(
            20,
            50,
            570,
            360,
            "Marginal disagreement",
            "mean absolute error",
            [
                ("BP vs MCMC", _series(aggregate, "bp_mcmc_mae")),
                ("BP vs Exact", _series(aggregate, "bp_exact_mae")),
                ("MCMC vs Exact", _series(aggregate, "mcmc_exact_mae")),
            ],
            (0.0, max(0.05, max(
                (
                    point[1] + point[2]
                    for metric in ("bp_mcmc_mae", "bp_exact_mae", "mcmc_exact_mae")
                    for point in _series(aggregate, metric)
                ),
                default=0.05,
            ) * 1.1)),
        ),
        _panel(
            610,
            50,
            570,
            360,
            "BP convergence",
            "converged fraction",
            [("BP", _series(aggregate, "bp_convergence_fraction"))],
            (0.0, 1.0),
        ),
        _panel(
            20,
            430,
            570,
            360,
            "MCMC relaxation",
            "integrated autocorrelation time",
            [("mine density", _series(aggregate, "mcmc_density_tau"))],
            (0.0, max(1.0, max(
                (point[1] + point[2] for point in _series(aggregate, "mcmc_density_tau")),
                default=1.0,
            ) * 1.1)),
        ),
        _panel(
            610,
            430,
            570,
            360,
            "Overlap",
            "mean overlap q",
            [
                ("MCMC planted", _series(aggregate, "mcmc_planted_overlap")),
                ("MCMC replica", _series(aggregate, "mcmc_replica_overlap")),
                ("Exact replica", _series(aggregate, "exact_replica_overlap")),
            ],
            (-1.0, 1.0),
        ),
    ]
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="820" '
        'viewBox="0 0 1200 820" role="img" aria-labelledby="title desc">'
        '<title id="title">Minesweeper CSP phase scan</title>'
        '<desc id="desc">Four panels showing marginal errors, BP convergence, MCMC '
        'autocorrelation time, and overlap versus mine density.</desc>'
        '<rect width="1200" height="820" fill="#f8fafc"/>'
        '<text x="600" y="28" text-anchor="middle" font-size="20" font-weight="600" '
        'fill="#0f172a">Minesweeper CSP phase scan</text>'
        '<g font-family="Arial, sans-serif" fill="#0f172a">'
        + "".join(panels)
        + "</g></svg>"
    )
    destination.write_text(svg, encoding="utf-8", newline="\n")
    return destination
