"""Performance metrics and plotting utilities."""
from __future__ import annotations

import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def calculate_metrics(portfolio: pd.DataFrame) -> pd.DataFrame:
    """Compute simple performance metrics from a portfolio DataFrame."""
    metrics = pd.DataFrame(index=portfolio.index)
    metrics["cumulative_return"] = (
        portfolio["portfolio_value"].pct_change().fillna(0).add(1).cumprod() - 1
    )
    return metrics


def plot_performance(metrics: pd.DataFrame) -> None:
    """Plot cumulative return using matplotlib if available."""
    if plt is None or metrics.empty:
        return
    ax = metrics["cumulative_return"].plot(title="Cumulative Return")
    ax.set_xlabel("Time")
    ax.set_ylabel("Return")
    plt.tight_layout()
    plt.show()
