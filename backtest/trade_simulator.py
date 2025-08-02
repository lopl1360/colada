"""Simple trade simulator stub."""
from __future__ import annotations

import pandas as pd


class TradeSimulator:
    """Simulate trades based on signals."""

    def __init__(self, initial_cash: float = 10_000.0) -> None:
        self.initial_cash = initial_cash

    def simulate(self, data: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
        """Generate portfolio values from price data and signals."""
        portfolio = pd.DataFrame(index=data.index)
        portfolio["cash"] = self.initial_cash
        portfolio["position"] = signals.fillna(0)
        # Placeholder for actual trade execution logic
        portfolio["portfolio_value"] = portfolio["cash"]
        return portfolio
