"""Run a configured rho/disorder sweep and produce aggregate outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from minesweeper_csp.aggregate import aggregate_directory
from minesweeper_csp.experiment import execute_sweep, load_sweep_config
from minesweeper_csp.visualization import write_phase_scan_svg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="JSON sweep configuration")
    parser.add_argument("--output", type=Path, default=Path("results/phase_scan"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_sweep_config(args.config)
    execute_sweep(
        config,
        args.output,
        overwrite=args.overwrite,
        repository_directory=Path.cwd(),
    )
    aggregate = aggregate_directory(args.output)
    write_phase_scan_svg(aggregate, args.output / "phase_scan.svg")
    print(f"completed {aggregate['run_count']} runs in {args.output}")


if __name__ == "__main__":
    main()

