"""Execute and analyze all finite-size sweeps in a candidate plan."""

from __future__ import annotations

import argparse
from pathlib import Path

from minesweeper_csp.candidate import execute_candidate_plan
from minesweeper_csp.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/candidate_study"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    analysis = execute_candidate_plan(
        read_json(args.plan),
        args.output,
        overwrite=args.overwrite,
        repository_directory=Path.cwd(),
    )
    print(f"completed candidate analysis for {len(analysis['sizes'])} sizes")


if __name__ == "__main__":
    main()

