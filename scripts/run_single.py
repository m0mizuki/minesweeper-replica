"""Run one indexed point from a configured rho/disorder sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

from minesweeper_csp.experiment import load_sweep_config, run_disorder_sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="JSON sweep configuration")
    parser.add_argument("--rho-index", type=int, required=True)
    parser.add_argument("--sample-index", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/phase_scan"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_sweep_config(args.config)
    if not 0 <= args.rho_index < len(config.rho_values):
        parser.error("rho-index is outside the configured rho_values")
    if not 0 <= args.sample_index < config.disorder_samples:
        parser.error("sample-index is outside the configured disorder range")
    summary = run_disorder_sample(
        config,
        args.rho_index,
        args.sample_index,
        args.output,
        overwrite=args.overwrite,
        repository_directory=Path.cwd(),
    )
    print(f"completed {summary['run_id']}")


if __name__ == "__main__":
    main()

