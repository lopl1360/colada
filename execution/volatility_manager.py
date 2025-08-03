"""Utilities for deriving stop-loss and take-profit levels from volatility."""

from __future__ import annotations

import pandas as pd


def compute_stop_target(prices: pd.Series) -> tuple[float, float]:
    """Compute stop-loss and take-profit percentages based on volatility.

    The function calculates the standard deviation of the last 14 price points
    (a simple proxy for volatility) and scales it relative to the latest price
    to determine the stop-loss percentage. The take-profit is set to twice the
    stop-loss value.

    Args:
        prices: Series of recent prices with the most recent price last.

    Returns:
        A tuple of ``(stop_pct, target_pct)`` where each value is expressed as
        a decimal percentage of the latest price.

    Raises:
        ValueError: If fewer than 14 prices are supplied.
    """
    if len(prices) < 14:
        raise ValueError("Need at least 14 prices to compute volatility")

    atr = prices.rolling(14).std().iloc[-1]
    stop_pct = atr / prices.iloc[-1]
    target_pct = 2 * stop_pct
    return stop_pct, target_pct
