import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from execution.position_sizer import calculate_position_size


def test_calculate_position_size_example():
    """Example: $50K equity, 10% stop loss on $100 price -> 50 shares."""
    assert calculate_position_size(50_000, 0.10, 100) == 50


def test_calculate_position_size_negative_stop_returns_zero():
    """Negative stop distance should result in zero shares."""
    assert calculate_position_size(50_000, -0.05, 100) == 0
