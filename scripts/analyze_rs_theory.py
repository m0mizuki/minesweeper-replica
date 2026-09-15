"""Compute finite-instance Bethe and local RS diagnostics for a candidate study."""

from __future__ import annotations

import argparse
from pathlib import Path

from minesweeper_csp.theory import analyze_rs_candidate_directory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--max-edges",
        type=int,
        default=256,
        help="maximum factor-graph edge count for the dense numerical Jacobian",
    )
    args = parser.parse_args()
    analysis = analyze_rs_candidate_directory(args.results, max_edges=args.max_edges)
    print(f"analyzed {len(analysis['sizes'])} sizes")


if __name__ == "__main__":
    main()
