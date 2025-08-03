"""Backtesting module scaffolding."""

from .backtest_engine import BacktestEngine
from .trade_simulator import TradeSimulator
from .metrics import (
    calculate_metrics,
    log_trade,
    plot_performance,
    track_equity,
)

__all__ = [
    "BacktestEngine",
    "TradeSimulator",
    "calculate_metrics",
    "plot_performance",
    "track_equity",
    "log_trade",
]
