"""Rebuild candidate diagnostics and figures from completed size sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path

from minesweeper_csp.candidate import analyze_candidate_directory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--overlap-bins", type=int)
    args = parser.parse_args()
    analysis = analyze_candidate_directory(
        args.results, overlap_bins=args.overlap_bins
    )
    print(f"analyzed {len(analysis['sizes'])} sizes")


if __name__ == "__main__":
    main()

