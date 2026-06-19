#!/usr/bin/env python
"""Benchmark _CalcFitness at increasing N to establish a baseline.

Run before and after vectorisation to measure the speedup:

    uv run python benchmarks/bench_calc_fitness.py
    uv run python benchmarks/bench_calc_fitness.py --output benchmarks/baseline.csv

Results are printed as a table and optionally saved to CSV.
N is capped at 3000 by default because the current O(N²) Python loop becomes
prohibitively slow beyond that — adjust with --max-n if needed.
"""
import argparse
import csv
import sys
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from BayesBlocks import bblock

logger.disable("BayesBlocks")

SIZES = [100, 250, 500, 1000, 2000, 3000]
REPEATS = 3  # take the best of this many runs
DATA_DIR = Path(__file__).parent / "data"


def load_tte_df(n: int) -> pd.DataFrame:
    """Load the pre-generated fixed TTE dataset for size n.

    Run benchmarks/generate_data.py first if the file is missing.
    """
    path = DATA_DIR / f"tte_n{n}.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: uv run python benchmarks/generate_data.py"
        )
    return pd.DataFrame({"TIME": np.load(path)})


def bench_calc_fitness(sizes: list[int], repeats: int) -> list[dict]:
    results = []

    print(f"\n{'N':>7}  {'_CalcFitness':>14}  {'find_blocks':>13}")
    print("-" * 42)

    for n in sizes:
        df = load_tte_df(n)

        # Prep for isolated _CalcFitness timing
        bb_prep = bblock(df, data_mode=1)
        bb_prep._ProcessData()
        bb_prep._CalcPrior()

        def run_fitness():
            bb_prep._CalcFitness()

        t_fitness = min(timeit.repeat(run_fitness, repeat=repeats, number=1))

        # Time the full pipeline
        def full_run():
            b = bblock(df, data_mode=1)
            b.find_blocks()

        t_full = min(timeit.repeat(full_run, repeat=repeats, number=1))

        results.append({
            "N": n,
            "calc_fitness_ms": round(t_fitness * 1000, 2),
            "find_blocks_ms": round(t_full * 1000, 2),
        })

        print(f"{n:>7}  {t_fitness*1000:>12.1f}ms  {t_full*1000:>11.1f}ms")

    return results


def save_csv(results: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["N", "calc_fitness_ms", "find_blocks_ms"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "-o", help="Save results to this CSV path")
    parser.add_argument("--max-n", type=int, default=3000,
                        help="Skip sizes above this value (default: 3000)")
    parser.add_argument("--repeats", type=int, default=REPEATS,
                        help=f"Timing repeats per size (default: {REPEATS})")
    args = parser.parse_args()

    sizes = [n for n in SIZES if n <= args.max_n]
    print(f"Benchmarking _CalcFitness (best of {args.repeats} runs each)")

    results = bench_calc_fitness(sizes, args.repeats)

    if args.output:
        save_csv(results, args.output)
