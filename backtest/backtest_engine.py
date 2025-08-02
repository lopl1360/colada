"""Core backtesting engine."""
from __future__ import annotations

import pandas as pd
from typing import Callable, Sequence

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

    def run_event_driven(
        self,
        predict_fn: Callable[[pd.Series], float],
        feature_cols: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Run an event-driven backtest using a prediction function.

        Parameters
        ----------
        predict_fn:
            Callable that accepts a feature vector and returns a numeric
            prediction.  The sign of this prediction determines the trading
            action (positive -> buy, negative -> sell, otherwise hold).
        feature_cols:
            Optional sequence of column names to use as the feature vector.
            If ``None`` all columns except the standard OHLCV fields are used.

        Returns
        -------
        pandas.DataFrame
            Simulator output containing portfolio information for each bar.
        """

        ohlcv = {"open", "high", "low", "close", "volume"}

        # Determine which columns constitute the feature vector
        if feature_cols is None:
            feature_cols = [c for c in self.data.columns if c not in ohlcv]

        signals: list[int] = []
        for _, row in self.data.iterrows():
            features = row[feature_cols]
            prediction = predict_fn(features)

            # Map numeric prediction to discrete trading signal
            if prediction > 0:
                signal = 1  # buy
            elif prediction < 0:
                signal = -1  # sell
            else:
                signal = 0  # hold

            signals.append(signal)

        signal_series = pd.Series(signals, index=self.data.index, name="signal")

        # Execute the trades via the simulator and calculate metrics
        self.results = self.simulator.simulate(self.data, signal_series)
        self.metrics = calculate_metrics(self.results)
        return self.results

    def plot(self) -> None:
        """Plot performance metrics if available."""
        if self.metrics is not None:
            plot_performance(self.metrics)
