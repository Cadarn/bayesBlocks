# _CalcFitness Benchmark Results

Timings for the `_CalcFitness` hot loop across implementations.
Each result is the **best of 3 runs** on an Apple Silicon M4 MacBook Pro.
Run with: `uv run python benchmarks/bench_calc_fitness.py`

---

## Python — original loop (baseline)

Branch: `chore/benchmarking` · Commit: `chore/project-setup`  
Implementation: Python `for` loop with `np.append` per iteration.  
Complexity: O(N²) time and O(N²) allocations.

| N    | _CalcFitness | find_blocks |
|-----:|-------------:|------------:|
|  100 |       0.7 ms |      0.8 ms |
|  250 |       2.0 ms |      2.1 ms |
|  500 |       4.7 ms |      4.7 ms |
| 1000 |      11.0 ms |     11.0 ms |
| 2000 |      28.8 ms |     30.7 ms |
| 3000 |      59.0 ms |     62.0 ms |

---

## NumPy vectorised

Branch: `perf/vectorise-fitness`  
Implementation: Pre-allocated arrays, Python loop eliminated with numpy broadcasting.  
Complexity: O(N²) compute in C; O(N²) memory, O(1) allocations.

| N    | _CalcFitness | find_blocks | Speedup vs baseline |
|-----:|-------------:|------------:|--------------------:|
|  100 |          TBD |         TBD |                 TBD |
|  250 |          TBD |         TBD |                 TBD |
|  500 |          TBD |         TBD |                 TBD |
| 1000 |          TBD |         TBD |                 TBD |
| 2000 |          TBD |         TBD |                 TBD |
| 3000 |          TBD |         TBD |                 TBD |

---

## Rust via PyO3 / Maturin

Branch: `feat/rust-core`  
Implementation: Rust port of the inner loop with optional `rayon` parallelism.  
Complexity: O(N²) — same algorithm, lower constant factor.

| N    | _CalcFitness | find_blocks | Speedup vs baseline | Speedup vs numpy |
|-----:|-------------:|------------:|--------------------:|-----------------:|
|  100 |          TBD |         TBD |                 TBD |              TBD |
|  250 |          TBD |         TBD |                 TBD |              TBD |
|  500 |          TBD |         TBD |                 TBD |              TBD |
| 1000 |          TBD |         TBD |                 TBD |              TBD |
| 2000 |          TBD |         TBD |                 TBD |              TBD |
| 3000 |          TBD |         TBD |                 TBD |              TBD |

---

## Notes

- All timings are wall-clock, best-of-3, single-threaded unless noted.
- `find_blocks` includes `_ProcessData`, `_CalcPrior`, `_CalcFitness`, `_RecoverCP`, and `_ProcessBlocks`. On all implementations the non-`_CalcFitness` steps are negligible.
- The O(N²) complexity is fundamental to the exact Scargle algorithm. For N > ~10,000 a different algorithm (e.g. PELT) would be required regardless of implementation language.
