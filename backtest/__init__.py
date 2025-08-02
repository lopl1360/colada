"""Backtesting module scaffolding."""

from .backtest_engine import BacktestEngine
from .trade_simulator import TradeSimulator
from .metrics import calculate_metrics, plot_performance

__all__ = [
    "BacktestEngine",
    "TradeSimulator",
    "calculate_metrics",
    "plot_performance",
]
