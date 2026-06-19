"""Correctness tests for the vectorised _CalcFitness implementation.

The reference implementation below is a direct transcription of the original
Python loop from BayesBlocks.py before vectorisation. Any vectorised or Rust
replacement must produce bit-identical best[] and last[] arrays.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from BayesBlocks import bblock

DATA_DIR = Path(__file__).parent.parent / "benchmarks" / "data"


# ---------------------------------------------------------------------------
# Reference implementation (original loop — do not modify)
# ---------------------------------------------------------------------------

def _calc_fitness_reference(nn_vec, block_length, ncp_prior):
    """Verbatim transcription of the original _CalcFitness loop for mode 1/2.

    Used as the ground-truth to validate any vectorised replacement.
    Returns (best, last) arrays identical to those stored on self after
    the original implementation runs.
    """
    npoints = len(nn_vec)
    best = np.array([])
    last = np.array([], dtype=np.int64)

    for j in range(npoints):
        arg_log = block_length[0:j + 1] - block_length[j + 1]
        arg_log[arg_log == 0.0] = np.inf
        nn_cum_vec = np.cumsum(nn_vec[j::-1])
        nn_cum_vec = nn_cum_vec[j::-1]
        fit_vec = nn_cum_vec * (np.log(nn_cum_vec) - np.log(arg_log))
        fit_vec -= ncp_prior
        fit_vec = np.concatenate((np.array([0]), best)) + fit_vec
        best = np.append(best, np.max(fit_vec))
        last = np.append(last, np.int64(np.argmax(fit_vec)))

    return best, last


def _prepared_bblock(df, data_mode=1, fp_rate=0.003):
    """Return a bblock with _ProcessData and _CalcPrior already run."""
    bb = bblock(df, data_mode=data_mode)
    bb._ProcessData()
    bb._CalcPrior(fp_rate=fp_rate)
    return bb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def prepared_tte(tte_dataframe):
    return _prepared_bblock(tte_dataframe, data_mode=1)


@pytest.fixture
def prepared_binned(binned_dataframe):
    return _prepared_bblock(binned_dataframe, data_mode=2)


@pytest.fixture(params=[100, 500, 1000])
def prepared_tte_at_n(request):
    """Parametrised fixture: fixed benchmark TTE datasets at N = 100, 500, 1000."""
    n = request.param
    path = DATA_DIR / f"tte_n{n}.npy"
    df = pd.DataFrame({"TIME": np.load(path)})
    return _prepared_bblock(df)


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

def test_calc_fitness_best_matches_reference(prepared_tte):
    bb = prepared_tte
    best_ref, _ = _calc_fitness_reference(bb.nn_vec, bb.block_length, bb.ncp_prior)

    bb._CalcFitness()

    np.testing.assert_allclose(
        bb.best, best_ref, rtol=1e-10,
        err_msg="best[] array does not match reference implementation"
    )


def test_calc_fitness_last_matches_reference(prepared_tte):
    bb = prepared_tte
    _, last_ref = _calc_fitness_reference(bb.nn_vec, bb.block_length, bb.ncp_prior)

    bb._CalcFitness()

    np.testing.assert_array_equal(
        bb.last, last_ref,
        err_msg="last[] array does not match reference implementation"
    )


def test_calc_fitness_binned_matches_reference(prepared_binned):
    bb = prepared_binned
    best_ref, last_ref = _calc_fitness_reference(bb.nn_vec, bb.block_length, bb.ncp_prior)

    bb._CalcFitness()

    np.testing.assert_allclose(bb.best, best_ref, rtol=1e-10)
    np.testing.assert_array_equal(bb.last, last_ref)


def test_calc_fitness_matches_reference_at_multiple_sizes(prepared_tte_at_n):
    bb = prepared_tte_at_n
    best_ref, last_ref = _calc_fitness_reference(bb.nn_vec, bb.block_length, bb.ncp_prior)

    bb._CalcFitness()

    np.testing.assert_allclose(bb.best, best_ref, rtol=1e-10)
    np.testing.assert_array_equal(bb.last, last_ref)


def test_find_blocks_changepoints_unchanged_after_refactor(tte_dataframe):
    """End-to-end: changepoints and rates must be identical before and after
    any refactoring of _CalcFitness."""
    bb_ref = bblock(tte_dataframe, data_mode=1)
    bb_ref._ProcessData()
    bb_ref._CalcPrior()
    best_ref, last_ref = _calc_fitness_reference(
        bb_ref.nn_vec, bb_ref.block_length, bb_ref.ncp_prior
    )
    bb_ref.best = best_ref
    bb_ref.last = last_ref
    bb_ref._RecoverCP()
    bb_ref._ProcessBlocks()

    bb = bblock(tte_dataframe, data_mode=1)
    bb.find_blocks()

    np.testing.assert_array_equal(bb.change_points, bb_ref.change_points)
    np.testing.assert_allclose(bb.rate_vec, bb_ref.rate_vec, rtol=1e-10)
