"""Rank coarse-scan points and prepare a finite-size candidate plan."""

from __future__ import annotations

import argparse
from pathlib import Path

from minesweeper_csp.candidate import build_candidate_plan, write_candidate_plan
from minesweeper_csp.io import read_json


def parse_sizes(value: str) -> list[tuple[int, int]]:
    try:
        return [tuple(int(part) for part in item.lower().split("x")) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("sizes must look like 4x4,6x6,8x8") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("aggregate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=parse_sizes, default=parse_sizes("4x4,6x6,8x8"))
    parser.add_argument("--dense-points", type=int, default=11)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--padding-points", type=int, default=1)
    parser.add_argument("--disorder-samples", type=int)
    parser.add_argument("--bp-random-restarts", type=int, default=6)
    parser.add_argument("--mcmc-chains", type=int)
    parser.add_argument("--base-seed", type=int)
    parser.add_argument("--overlap-bins", type=int, default=41)
    args = parser.parse_args()

    plan = build_candidate_plan(
        read_json(args.aggregate),
        sizes=args.sizes,
        dense_points=args.dense_points,
        top_k=args.top_k,
        padding_points=args.padding_points,
        disorder_samples=args.disorder_samples,
        bp_random_restarts=args.bp_random_restarts,
        mcmc_chains=args.mcmc_chains,
        base_seed=args.base_seed,
        overlap_bins=args.overlap_bins,
    )
    write_candidate_plan(args.output, plan)
    selection = plan["selection"]
    print(
        f"candidate rho region [{selection['rho_min']}, {selection['rho_max']}] "
        f"written to {args.output}"
    )


if __name__ == "__main__":
    main()

