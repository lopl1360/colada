import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from execution.volatility_manager import compute_stop_target


def test_compute_stop_target_returns_expected_values():
    """Computed stop/target should match manual volatility calculation."""
    prices = pd.Series(np.arange(1, 31))
    stop, target = compute_stop_target(prices)
    expected_std = np.std(np.arange(17, 31), ddof=1)
    expected_stop = expected_std / prices.iloc[-1]
    assert stop == pytest.approx(expected_stop)
    assert target == pytest.approx(2 * expected_stop)


def test_compute_stop_target_zero_volatility():
    """Flat price series should yield zero stop and target percentages."""
    prices = pd.Series([10] * 20)
    stop, target = compute_stop_target(prices)
    assert stop == 0
    assert target == 0
