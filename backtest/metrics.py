"""Performance metrics and plotting utilities."""

from __future__ import annotations

from math import sqrt
from typing import Any, Dict, List

import pandas as pd

try:  # pragma: no cover - matplotlib is optional
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

# ---------------------------------------------------------------------------
# Data stores
# ---------------------------------------------------------------------------
_equity_curve: List[tuple[pd.Timestamp, float]] = []
_trade_log: List[Dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
def track_equity(timestamp: Any, equity: float) -> None:
    """Record account equity at a given time."""

    ts = pd.to_datetime(timestamp)
    _equity_curve.append((ts, float(equity)))


def log_trade(entry_time: Any, exit_time: Any, pnl: float, reason: str) -> None:
    """Log a completed trade with basic information."""

    trade = {
        "entry_time": pd.to_datetime(entry_time),
        "exit_time": pd.to_datetime(exit_time),
        "pnl": float(pnl),
        "reason": reason,
    }
    _trade_log.append(trade)


# ---------------------------------------------------------------------------
# Metric calculations
# ---------------------------------------------------------------------------
def equity_curve_dataframe() -> pd.DataFrame:
    """Return the recorded equity curve as a DataFrame."""

    if not _equity_curve:
        return pd.DataFrame(columns=["equity"])
    df = pd.DataFrame(_equity_curve, columns=["timestamp", "equity"]).set_index(
        "timestamp"
    )
    return df


def calculate_drawdown() -> float:
    """Calculate the maximum drawdown from the equity curve."""

    df = equity_curve_dataframe()
    if df.empty:
        return 0.0
    running_max = df["equity"].cummax()
    drawdowns = (df["equity"] - running_max) / running_max
    return float(drawdowns.min())


def calculate_metrics() -> Dict[str, float]:
    """Compute summary metrics from the trade log and equity curve."""

    trades = pd.DataFrame(_trade_log)
    equity_df = equity_curve_dataframe()

    returns = equity_df["equity"].pct_change().dropna()
    if not returns.empty and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * sqrt(len(returns))
    else:  # Avoid division by zero
        sharpe = 0.0

    win_trades = trades[trades["pnl"] > 0]
    loss_trades = trades[trades["pnl"] < 0]

    metrics: Dict[str, float] = {
        "num_trades": float(len(trades)),
        "win_rate": float((trades["pnl"] > 0).mean()) if not trades.empty else 0.0,
        "avg_win": float(win_trades["pnl"].mean()) if not win_trades.empty else 0.0,
        "avg_loss": float(loss_trades["pnl"].mean()) if not loss_trades.empty else 0.0,
        "max_drawdown": float(abs(calculate_drawdown())),
        "sharpe_ratio": float(sharpe),
    }

    return metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_performance(metrics: pd.DataFrame | None = None) -> None:  # noqa: D401
    """Plot the equity curve using matplotlib if available."""

    if plt is None:
        return

    df = equity_curve_dataframe()
    if df.empty:
        return

    ax = df["equity"].plot(title="Equity Curve")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    plt.tight_layout()
    plt.show()


__all__ = [
    "track_equity",
    "log_trade",
    "calculate_drawdown",
    "calculate_metrics",
    "plot_performance",
    "equity_curve_dataframe",
]
