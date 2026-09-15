"""Re-aggregate already completed phase-scan runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from minesweeper_csp.aggregate import aggregate_directory
from minesweeper_csp.visualization import write_phase_scan_svg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    aggregate = aggregate_directory(args.results)
    write_phase_scan_svg(aggregate, args.results / "phase_scan.svg")
    print(f"aggregated {aggregate['run_count']} runs")


if __name__ == "__main__":
    main()

