"""Tests for bblock — the single time series Bayesian Blocks class."""
import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from BayesBlocks import bblock


# ---------------------------------------------------------------------------
# _ProcessData
# ---------------------------------------------------------------------------

def test_process_data_tte_sets_time_and_nn_vec(tte_dataframe):
    bb = bblock(tte_dataframe, data_mode=1)
    bb._ProcessData()
    assert len(bb.time) == len(tte_dataframe)
    assert np.all(bb.nn_vec == 1.0)


def test_process_data_exposure_column_is_used(tte_dataframe_with_exposure):
    """EXPOSURE column should be normalised into relExp, not ignored.

    Fails until DataFrame column detection is fixed: hasattr() on a
    DataFrame resolves column names as attributes, but 'col' in df.columns
    is the correct and explicit check.
    """
    bb = bblock(tte_dataframe_with_exposure, data_mode=1)
    bb._ProcessData()
    assert not np.all(bb.relExp == 1.0), (
        "relExp is all-ones — EXPOSURE column was not detected"
    )


def test_process_data_no_exposure_defaults_to_ones(tte_dataframe):
    bb = bblock(tte_dataframe, data_mode=1)
    bb._ProcessData()
    assert np.all(bb.relExp == 1.0)


def test_process_data_binned_reads_counts(binned_dataframe):
    bb = bblock(binned_dataframe, data_mode=2)
    bb._ProcessData()
    assert np.array_equal(bb.nn_vec, binned_dataframe["COUNTS"].values)


def test_process_data_binned_raises_without_counts_or_flux():
    df = pd.DataFrame({"TIME": np.arange(10, dtype=float)})
    bb = bblock(df, data_mode=2)
    with pytest.raises(AttributeError):
        bb._ProcessData()


def test_process_data_mode3_uses_time_column(point_measurement_dataframe):
    """Mode 3 must use the actual TIME column values, not np.arange(npoints).

    Currently broken: _ProcessData assigns self.time = np.arange(self.npoints)
    which discards real time coordinates.
    """
    bb = bblock(point_measurement_dataframe, data_mode=3)
    bb._ProcessData()
    np.testing.assert_array_equal(
        bb.time,
        point_measurement_dataframe["TIME"].values,
        err_msg="Mode 3 must store real TIME values, not np.arange(npoints)",
    )


def test_process_data_rejects_non_monotonic_times():
    df = pd.DataFrame({"TIME": [1.0, 0.5, 2.0]})
    bb = bblock(df, data_mode=1)
    with pytest.raises(AssertionError):
        bb._ProcessData()


# ---------------------------------------------------------------------------
# _CalcPrior
# ---------------------------------------------------------------------------

def test_calc_prior_stricter_fp_rate_gives_larger_prior(tte_dataframe):
    """A lower false-positive rate should produce a larger (more penalising) ncp_prior."""
    bb = bblock(tte_dataframe, data_mode=1)
    bb._ProcessData()

    bb._CalcPrior(fp_rate=0.05)
    prior_loose = bb.ncp_prior

    bb._CalcPrior(fp_rate=0.001)
    prior_strict = bb.ncp_prior

    assert prior_strict > prior_loose


# ---------------------------------------------------------------------------
# find_blocks (end-to-end)
# ---------------------------------------------------------------------------

def test_find_blocks_detects_two_rate_regions(tte_dataframe):
    bb = bblock(tte_dataframe, data_mode=1)
    bb.find_blocks(fp_rate=0.05)
    assert bb.num_blocks >= 2


def test_find_blocks_uniform_data_returns_single_block(tte_uniform_dataframe):
    """Perfectly uniform event spacing should not produce spurious changepoints."""
    bb = bblock(tte_uniform_dataframe, data_mode=1)
    bb.find_blocks(fp_rate=0.01)
    assert bb.num_blocks == 1


def test_find_blocks_result_attributes_present(tte_dataframe):
    bb = bblock(tte_dataframe, data_mode=1)
    bb.find_blocks()
    assert hasattr(bb, "change_points")
    assert hasattr(bb, "rate_vec")
    assert hasattr(bb, "num_blocks")


def test_find_blocks_rate_vec_length_matches_num_blocks(tte_dataframe):
    bb = bblock(tte_dataframe, data_mode=1)
    bb.find_blocks()
    assert len(bb.rate_vec) == bb.num_blocks


def test_find_blocks_binned_detects_step_change(binned_dataframe):
    bb = bblock(binned_dataframe, data_mode=2)
    bb.find_blocks(fp_rate=0.01)
    assert bb.num_blocks >= 2


def test_find_blocks_point_measurements_detects_step_change(point_measurement_dataframe):
    bb = bblock(point_measurement_dataframe, data_mode=3)
    bb.find_blocks(fp_rate=0.05)
    assert bb.num_blocks >= 2
