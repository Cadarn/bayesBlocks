#!/usr/bin/env python
"""Generate fixed benchmark datasets and save them to benchmarks/data/.

Run once to create the canonical datasets used by all benchmark scripts:

    uv run python benchmarks/generate_data.py

Each file is a 1-D numpy array of TIME values for a two-rate TTE dataset.
The first half of events has a low rate; the second half a 5× higher rate,
producing a clear changepoint at the midpoint.

Files are committed to the repository so benchmark runs are reproducible
across machines and implementations without any RNG at timing time.
"""
from pathlib import Path
import numpy as np

SIZES = [100, 250, 500, 1000, 2000, 3000]
SEED = 42
OUTPUT_DIR = Path(__file__).parent / "data"


def generate_two_rate_tte(n: int, rng: np.random.Generator) -> np.ndarray:
    """Two-rate TTE: n//2 events at low rate, n - n//2 at 5× higher rate."""
    half = n // 2
    low = np.sort(rng.uniform(0.0, half * 0.10, half))
    high = np.sort(rng.uniform(half * 0.10, half * 0.10 + (n - half) * 0.02, n - half))
    return np.concatenate([low, high])


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)

    for n in SIZES:
        times = generate_two_rate_tte(n, rng)
        path = OUTPUT_DIR / f"tte_n{n}.npy"
        np.save(path, times)
        print(f"Saved {path}  ({len(times)} events, "
              f"t=[{times[0]:.3f}, {times[-1]:.3f}])")

    print(f"\nAll datasets written to {OUTPUT_DIR}")
