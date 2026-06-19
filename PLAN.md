# BayesBlocks Improvement Plan

## Overview

Modernise and extend the BayesBlocks implementation with a clean Python API, Typer CLI, incremental streaming support, a Rust-accelerated core via PyO3/Maturin, and a Textual TUI for live monitoring.

---

## Phase 1 — Python Foundation

### Task 1: Fix bugs in existing Python code

- Fix `bblock_multi._processTimeMarkers`: references `dataList`, `tt_start_vec`, `ncp_prior_vec` without `self.` — would crash immediately
- Fix DataFrame column detection: replace `hasattr(self.data, 'EXPOSURE')` pattern with `'EXPOSURE' in self.data.columns`
- Fix mode 3: `self.time = np.arange(self.npoints)` silently ignores real time coordinates
- Replace deprecated `normed=True` in `testBayes()` plot with `density=True`
- Replace bare `print` statements with `logging`

### Task 2: Vectorise `_CalcFitness` and add `pyproject.toml`

- Replace `np.append(best, ...)` / `np.append(last, ...)` inside the for loop with pre-allocated `np.zeros(N)` arrays — eliminates O(N²) memory allocations
- Optionally eliminate the Python for loop entirely using numpy broadcasting (as astropy does), pushing all O(N²) work into C
- Add `pyproject.toml` via `uv init` with dependencies: `pandas`, `numpy`, `matplotlib`
- Add a benchmark script to measure before/after — establishes the baseline for the Rust comparison in Phase 3

### Task 3: Expose a clean top-level API function

- Add a single `bayesian_blocks(t, x=None, sigma=None, mode='events', p0=0.003)` function as the primary public interface, wrapping `bblock` internally
- Remove or mark the legacy top-level `find_blocks()` function as deprecated
- Return a clean result object or namedtuple (`change_points`, `rate_vec`, `block_edges`) rather than relying on instance attributes
- Keep the class available for users who want the step-by-step pipeline

---

## Phase 2 — CLI

### Task 4: Build Typer CLI with `analyse` subcommand

- Add `typer` to `pyproject.toml` and create a `cli.py` entry point
- `bayesblocks analyse <file.csv>` — reads CSV, accepts `--mode`, `--fp-rate`, `--time-col` flags
- Output: print block table to stdout by default
- `--output <file.csv>` writes changepoints + rates + block edges
- `--json` flag for newline-delimited JSON output (composable with other tools)
- Wire up as a `console_scripts` entry point in `pyproject.toml`

### Task 5: Implement `stream` subcommand with incremental algorithm

- `bayesblocks stream` reads newline-delimited CSV/JSON events from stdin, one event per line
- Implement the incremental update: store `best[]` and `last[]` between events, compute only `best[N+1]` on each new event — O(N) per event rather than reprocessing from scratch
- Emit a JSON changepoint event to stdout whenever a new changepoint is confirmed
- `--window <int>` flag for sliding window to bound per-event cost for long-running streams (recommended in Scargle 2012, Section 4.3)
- `--fp-rate` flag
- Optional: `--kafka-topic` / `--bootstrap-servers` flags to consume directly from a Kafka topic instead of stdin

---

## Phase 3 — Rust Core

### Task 6: Set up Maturin + PyO3 project structure

- Install maturin: `uv add --dev maturin`
- `maturin init` to scaffold the Rust extension alongside the Python package
- Configure `pyproject.toml` for a mixed Python/Rust project (maturin build backend)
- Verify the round-trip: a trivial Rust function callable from Python
- Set up `maturin develop` workflow for iterative development
- Learning milestone: understand the PyO3 `#[pyfunction]` and `#[pymodule]` macros

### Task 7: Port `_CalcFitness` hot loop to Rust

- Port the O(N²) fitness computation to Rust, accepting numpy arrays via `pyo3-numpy`
- Implement cumulative sum and log-likelihood calculation in Rust — introduces iterators, `Vec`, and the `ndarray` crate
- Explore `rayon` for parallelising the outer loop (each iteration is independent given `best[0..j]`)
- Return `best[]` and `last[]` arrays back to Python as numpy arrays
- Benchmark against: (a) original Python loop, (b) vectorised numpy version
- Keep Python implementation as fallback and for validating Rust output

---

## Phase 4 — Textual TUI

### Task 8: Build Textual TUI `watch` subcommand

- `bayesblocks watch` — long-running Textual dashboard for human monitoring of a live stream
- Add `textual` to `pyproject.toml`
- Display panels:
  - Current event count and active block rate
  - Sparkline of recent event rate
  - Scrolling log of confirmed changepoints with timestamp and rate delta
- Consumes the same stdin/Kafka source as the `stream` subcommand — shared detection logic, different rendering layer
- Detection logic lives in the core library; Textual is purely display

---

## Notes

- The Scargle et al. 2012 paper (arXiv:1207.5578) explicitly addresses real-time/incremental use in Section 4.3 "Real Time Analysis: Triggers". The incremental algorithm is the intended design, not an adaptation.
- The O(N) per-event incremental update works because `tStop` cancels out of the fitness calculation, meaning previously stored `best[]` values remain valid when new events arrive.
- For perpetual streams, the sliding window (`--window`) bounds latency at the cost of not considering very old history as candidate block starts.
- The Rust step is motivated by learning Maturin/PyO3 as much as raw performance. Once `_CalcFitness` is properly numpy-vectorised, the Python version will already be fast; Rust adds further headroom for large N and enables `rayon` parallelism.
