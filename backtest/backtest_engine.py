"""Core backtesting engine."""
from __future__ import annotations

import pandas as pd

from .trade_simulator import TradeSimulator
from .metrics import calculate_metrics, plot_performance


class BacktestEngine:
    """Coordinate trade simulations and metric calculations."""

    def __init__(self, data: pd.DataFrame, initial_cash: float = 10_000.0) -> None:
        self.data = data
        self.simulator = TradeSimulator(initial_cash=initial_cash)
        self.results: pd.DataFrame | None = None
        self.metrics: pd.DataFrame | None = None

    def run(self, signals: pd.Series) -> pd.DataFrame:
        """Execute a backtest given trading signals."""
        self.results = self.simulator.simulate(self.data, signals)
        self.metrics = calculate_metrics(self.results)
        return self.results

    def plot(self) -> None:
        """Plot performance metrics if available."""
        if self.metrics is not None:
            plot_performance(self.metrics)
