# BayesBlocks Improvement Plan

## Overview

Modernise and extend the BayesBlocks implementation with a clean Python API, Typer CLI, incremental streaming support, a Rust-accelerated core via PyO3/Maturin, and a Textual TUI for live monitoring.

## Current state

- **Branch model**: `master` (stable, tagged) ← `dev` (integration) ← feature branches
- **Latest tag**: `v0.1.2`
- **Resume point**: start Task 2 on a new branch `perf/vectorise-fitness` cut from `dev`

---

## Phase 1 — Python Foundation ✅

### Task 1: Fix bugs in existing Python code ✅ `v0.1.1`

All fixed on branch `fix/python-cleanup`, merged to `dev` → `master`.

- Fixed `bblock_multi._processTimeMarkers`: missing `self.` on `tt_start_vec`, `tt_stop_vec`, `ncp_prior_vec`; `dataList` replaced with `self.bbData`
- Fixed DataFrame column detection: `hasattr()` replaced with `'col' in self.data.columns` throughout
- Fixed mode 3 time: `np.arange(npoints)` replaced with `np.array(self.data.TIME)`
- Fixed block-count off-by-one: `num_blocks = len(change_points)` (was `+1`); rewrote `_ProcessBlocks` with clean `block_starts` boundary array
- `bblock_multi` now raises `ValueError`/`TypeError` on bad input instead of printing and continuing
- Replaced all `print()` with `loguru` logger (`logger.info/warning/error`)
- Deprecated `normed=True` → `density=True` in `testBayes()`
- Replaced `import pylab` with `import matplotlib.pyplot as plt`

### Infrastructure completed alongside Phase 1 ✅ `v0.1.2`

- `pyproject.toml` with `hatchling` build backend; `uv` for environment management
- `pytest` test suite: `tests/conftest.py` (stable fixtures with fixed sample data), `tests/test_bblock.py`, `tests/test_bblock_multi.py` — 20 tests, all passing
- GitHub Actions CI (`.github/workflows/ci.yml`) — runs on push/PR against Python 3.10, 3.11, 3.12
- `benchmarks/` directory:
  - `generate_data.py` — generates fixed seeded datasets; run once to populate `benchmarks/data/`
  - `benchmarks/data/tte_n{N}.npy` — committed fixed TTE datasets at N = 100, 250, 500, 1000, 2000, 3000
  - `bench_calc_fitness.py` — loads fixed data, times `_CalcFitness` and `find_blocks`, saves CSV
  - `baseline.csv` — captured Python loop timings (see RESULTS.md)
  - `RESULTS.md` — permanent record of timings per implementation; update after each phase
- `tests/test_vectorisation.py` — 7 correctness tests embedding the original loop as a reference; any new implementation must produce bit-identical `best[]`/`last[]`

---

## Phase 2 — Python optimisation and API ← next up

### Task 2: Vectorise `_CalcFitness` ← **start here**

Branch: `perf/vectorise-fitness` from `dev`

- Pre-allocate `best` and `last` as `np.zeros(N)` — eliminates O(N²) memory allocations from `np.append` in loop
- Eliminate the Python `for` loop using numpy broadcasting: construct the full `(N, N)` cumsum and log-likelihood in one pass, then `np.max`/`np.argmax` along the axis — pushes all O(N²) work into C
- After implementing, run `benchmarks/bench_calc_fitness.py` and fill in the numpy row in `benchmarks/RESULTS.md`
- All 7 tests in `tests/test_vectorisation.py` must pass (bit-identical output vs reference)

### Task 3: Expose a clean top-level API function

Branch: `feat/clean-api` from `dev`

- Add a single `bayesian_blocks(t, x=None, sigma=None, mode='events', p0=0.003)` function as the primary public interface, wrapping `bblock` internally
- Remove or deprecate the legacy top-level `find_blocks()` function
- Return a clean result dataclass or namedtuple (`change_points`, `rate_vec`, `block_edges`) rather than relying on instance attributes
- Keep the class available for users who want the step-by-step pipeline

---

## Phase 3 — CLI

### Task 4: Build Typer CLI with `analyse` subcommand

Branch: `feat/typer-cli` from `dev`

- Add `typer` to `pyproject.toml` and create a `cli.py` entry point
- `bayesblocks analyse <file.csv>` — reads CSV, accepts `--mode`, `--fp-rate`, `--time-col` flags
- Output: print block table to stdout by default; `--output <file.csv>` writes changepoints + rates + block edges; `--json` for newline-delimited JSON
- Wire up as a `console_scripts` entry point in `pyproject.toml`

### Task 5: Implement `stream` subcommand with incremental algorithm

Branch: `feat/stream` from `dev`

- `bayesblocks stream` reads newline-delimited CSV/JSON events from stdin, one event per line
- Implement the incremental update: store `best[]` and `last[]` between events, compute only `best[N+1]` on each new event — O(N) per event rather than reprocessing from scratch
- Emit a JSON changepoint event to stdout whenever a new changepoint is confirmed
- `--window <int>` flag for sliding window to bound per-event cost for long-running streams (Scargle 2012, Section 4.3)
- Optional: `--kafka-topic` / `--bootstrap-servers` flags to consume directly from a Kafka topic instead of stdin

---

## Phase 4 — Rust Core

### Task 6: Set up Maturin + PyO3 project structure

Branch: `feat/rust-scaffold` from `dev`

- Install maturin: `uv add --dev maturin`
- `maturin init` to scaffold the Rust extension alongside the Python package
- Configure `pyproject.toml` for a mixed Python/Rust project (maturin build backend)
- Verify the round-trip: a trivial Rust function callable from Python
- Set up `maturin develop` workflow for iterative development
- Learning milestone: understand the PyO3 `#[pyfunction]` and `#[pymodule]` macros

### Task 7: Port `_CalcFitness` hot loop to Rust

Branch: `feat/rust-core` from `dev`

- Port the O(N²) fitness computation to Rust, accepting numpy arrays via `pyo3-numpy`
- Implement cumulative sum and log-likelihood in Rust — introduces iterators, `Vec`, and the `ndarray` crate
- Explore `rayon` for parallelising the outer loop (each iteration is independent given `best[0..j]`)
- Return `best[]` and `last[]` arrays back to Python as numpy arrays
- Run `benchmarks/bench_calc_fitness.py` and fill in the Rust row in `benchmarks/RESULTS.md`
- All 7 tests in `tests/test_vectorisation.py` must pass

---

## Phase 5 — Textual TUI

### Task 8: Build Textual TUI `watch` subcommand

Branch: `feat/textual-tui` from `dev`

- `bayesblocks watch` — long-running Textual dashboard for human monitoring of a live stream
- Add `textual` to `pyproject.toml`
- Display panels: current event count and active block rate; sparkline of recent event rate; scrolling log of confirmed changepoints with timestamp and rate delta
- Consumes the same stdin/Kafka source as the `stream` subcommand — shared detection logic, Textual is purely display

---

## Notes

- The Scargle et al. 2012 paper (arXiv:1207.5578) explicitly addresses real-time/incremental use in Section 4.3 "Real Time Analysis: Triggers". The incremental O(N)-per-event update is the intended design, not an adaptation. `tStop` cancels out of the fitness calculation so previously stored `best[]` values remain valid when new events arrive.
- For perpetual streams, `--window` bounds latency at the cost of not considering very old history as candidate block starts.
- The Rust step is motivated by learning Maturin/PyO3 as much as raw performance. Once `_CalcFitness` is numpy-vectorised the Python version is already fast; Rust adds further headroom for large N and unlocks `rayon` parallelism.
- Benchmark data in `benchmarks/data/` was generated with `generate_data.py` (seed=42) — regenerate only if the benchmark sizes change, and update `baseline.csv` and `RESULTS.md` accordingly.
