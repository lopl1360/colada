"""Core backtesting engine."""

from __future__ import annotations

import json
from typing import Any, Callable, Sequence
import logging

import pandas as pd

from .trade_simulator import TradeSimulator
from .metrics import (
    _trade_log,
    calculate_metrics,
    plot_performance,
    track_equity,
    equity_curve_dataframe,
)


logger = logging.getLogger(__name__)


class BacktestEngine:
    """Coordinate trade simulations and metric calculations."""

    def __init__(self, data: pd.DataFrame, initial_cash: float = 10_000.0) -> None:
        self.data = data
        self.simulator = TradeSimulator(initial_cash=initial_cash)
        self.results: pd.DataFrame | None = None
        self.metrics: dict[str, float] | None = None

    def run(self, signals: pd.Series) -> pd.DataFrame:
        """Execute a backtest given trading signals."""
        self.results = self.simulator.simulate(self.data, signals)
        for ts, row in self.results.iterrows():
            track_equity(ts, row["portfolio_value"])
        self.metrics = calculate_metrics()
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
        for ts, row in self.results.iterrows():
            track_equity(ts, row["portfolio_value"])
        self.metrics = calculate_metrics()
        return self.results

    def plot(self) -> None:
        """Plot performance metrics if available."""
        if self.metrics is not None:
            plot_performance()


def run_backtest(
    data: pd.DataFrame,
    model: Any,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a backtest and output a summary report.

    Parameters
    ----------
    data:
        Price and feature data for the backtest.
    model:
        Object with a ``predict`` method or a callable returning numeric
        predictions.  The sign of the prediction determines the trading
        signal (positive -> buy, negative -> sell).
    config:
        Optional configuration dictionary.  Recognised keys include
        ``initial_cash`` for the simulator, ``feature_cols`` to select
        model input columns and ``export``/``export_format`` for saving
        results to disk.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the raw results, equity curve, trade log and
        summary metrics.
    """

    if config is None:
        config = {}

    engine = BacktestEngine(data, initial_cash=config.get("initial_cash", 10_000.0))
    feature_cols = config.get("feature_cols")

    def predict_fn(features: pd.Series) -> float:
        if hasattr(model, "predict"):
            prediction = model.predict(features.to_frame().T)
            if isinstance(prediction, (pd.Series, pd.DataFrame, list, tuple)):
                prediction = prediction[0]
        else:
            prediction = model(features)
        return float(prediction)

    results = engine.run_event_driven(predict_fn, feature_cols=feature_cols)

    equity = equity_curve_dataframe()
    trade_log = pd.DataFrame(_trade_log)
    summary = engine.metrics or calculate_metrics()

    logger.info("Equity Curve:\n%s", equity.tail())

    logger.info("\nTrade Log:\n%s", trade_log)

    logger.info("\nSummary Stats:")
    for key, value in summary.items():
        logger.info("%s: %s", key, value)

    export_path = config.get("export")
    if export_path:
        fmt = config.get("export_format", "csv").lower()
        if fmt == "json":
            equity.to_json(
                f"{export_path}_equity.json", orient="records", date_format="iso"
            )
            trade_log.to_json(
                f"{export_path}_trades.json", orient="records", date_format="iso"
            )
            with open(f"{export_path}_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f)
        else:
            equity.to_csv(f"{export_path}_equity.csv")
            trade_log.to_csv(f"{export_path}_trades.csv", index=False)
            pd.DataFrame([summary]).to_csv(f"{export_path}_summary.csv", index=False)

    return {
        "results": results,
        "equity_curve": equity,
        "trade_log": trade_log,
        "summary": summary,
    }
