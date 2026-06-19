"""Tests for bblock_multi — the multi-series Bayesian Blocks class."""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from BayesBlocks import bblock_multi


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def test_multi_accepts_integer_data_mode(two_tte_dataframes):
    multi = bblock_multi(two_tte_dataframes, data_modes=1)
    assert multi.nseries == 2
    assert multi.data_mode_vec == [1, 1]


def test_multi_accepts_list_data_modes(two_tte_dataframes):
    multi = bblock_multi(two_tte_dataframes, data_modes=[1, 1])
    assert multi.nseries == 2


def test_multi_mismatched_modes_list_raises(two_tte_dataframes):
    """Passing a modes list of the wrong length should raise, not silently print."""
    with pytest.raises((AssertionError, ValueError)):
        bblock_multi(two_tte_dataframes, data_modes=[1, 1, 1])


def test_multi_each_series_analysed_on_init(two_tte_dataframes):
    multi = bblock_multi(two_tte_dataframes, data_modes=1)
    assert len(multi.bbData) == 2
    for bb in multi.bbData:
        assert hasattr(bb, "num_blocks")


# ---------------------------------------------------------------------------
# _processTimeMarkers
# ---------------------------------------------------------------------------

def test_process_time_markers_does_not_crash(two_tte_dataframes):
    """_processTimeMarkers currently raises NameError for tt_start_vec,
    tt_stop_vec, ncp_prior_vec, and dataList (all missing self. prefix).
    """
    multi = bblock_multi(two_tte_dataframes, data_modes=1)
    multi._processTimeMarkers()  # should not raise NameError


def test_process_time_markers_populates_tt(two_tte_dataframes):
    multi = bblock_multi(two_tte_dataframes, data_modes=1)
    multi._processTimeMarkers()
    total_points = sum(len(df) for df in two_tte_dataframes)
    assert len(multi.tt) == total_points
