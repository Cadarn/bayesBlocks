"""Shared fixtures and stable test datasets for the BayesBlocks test suite.

All sample data is defined here so individual test modules stay free of
data-construction boilerplate and datasets are reused consistently.
"""
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Stable sample datasets
# ---------------------------------------------------------------------------

# TTE: two clearly separated rate regions (25 events over 5 s, then 50 over 1 s).
# Using a fixed seed so results are deterministic across runs.
_rng = np.random.default_rng(42)
_LOW_RATE_TIMES  = np.sort(_rng.uniform(0.0, 5.0, 25))
_HIGH_RATE_TIMES = np.sort(_rng.uniform(5.0, 6.0, 50))
TTE_TWO_RATE_TIMES = np.concatenate([_LOW_RATE_TIMES, _HIGH_RATE_TIMES])

# TTE with exposure weights
_rng2 = np.random.default_rng(7)
TTE_EXPOSURE_TIMES   = np.sort(_rng2.uniform(0.0, 10.0, 100))
TTE_EXPOSURE_WEIGHTS = _rng2.uniform(0.5, 1.0, 100)

# Perfectly uniform spacing — no rate change present
TTE_UNIFORM_TIMES = np.linspace(0.0, 10.0, 200)

# Binned: flat rate for 50 bins then 5× higher for 50 bins
BINNED_TIMES  = np.arange(100, dtype=float)
BINNED_COUNTS = np.concatenate([np.full(50, 2.0), np.full(50, 10.0)])

# Point measurements: non-uniform time spacing with a rate step at index 5
POINT_TIMES  = np.array([0.0, 0.5, 1.5, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 5.0])
POINT_FLUX   = np.concatenate([np.full(5, 1.0), np.full(5, 5.0)])
POINT_ERRORS = np.full(10, 0.1)

# Second TTE series (for multi-series tests)
_rng3 = np.random.default_rng(99)
TTE_SERIES_2_TIMES = np.sort(_rng3.uniform(0.0, 4.0, 40))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tte_dataframe():
    return pd.DataFrame({"TIME": TTE_TWO_RATE_TIMES})


@pytest.fixture
def tte_dataframe_with_exposure():
    return pd.DataFrame({
        "TIME": TTE_EXPOSURE_TIMES,
        "EXPOSURE": TTE_EXPOSURE_WEIGHTS,
    })


@pytest.fixture
def tte_uniform_dataframe():
    return pd.DataFrame({"TIME": TTE_UNIFORM_TIMES})


@pytest.fixture
def binned_dataframe():
    return pd.DataFrame({"TIME": BINNED_TIMES, "COUNTS": BINNED_COUNTS})


@pytest.fixture
def point_measurement_dataframe():
    return pd.DataFrame({
        "TIME":  POINT_TIMES,
        "FLUX":  POINT_FLUX,
        "ERROR": POINT_ERRORS,
    })


@pytest.fixture
def two_tte_dataframes(tte_dataframe):
    df2 = pd.DataFrame({"TIME": TTE_SERIES_2_TIMES})
    return [tte_dataframe, df2]
