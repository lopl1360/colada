"""Utility for submitting bracket orders via the Alpaca API."""

import logging
import os

from alpaca_trade_api.rest import REST

from . import position_manager

logger = logging.getLogger(__name__)

# Acquire credentials from environment variables, falling back to dummy strings to
# allow initialization in environments where real credentials are unavailable
API_KEY = os.getenv("ALPACA_API_KEY", "DUMMY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "DUMMY")
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Create a reusable Alpaca REST client
alpaca = REST(API_KEY, SECRET_KEY, BASE_URL)


def submit_bracket_order(symbol, qty, side, entry_price, stop_pct, target_pct):
    """Submit a market bracket order with stop-loss and take-profit.

    Args:
        symbol (str): The asset symbol to trade.
        qty (int | float): Number of shares to trade.
        side (str): 'buy' or 'sell'.
        entry_price (float): The intended entry price.
        stop_pct (float): Stop-loss percentage (e.g., 0.02 for 2%).
        target_pct (float): Take-profit percentage (e.g., 0.05 for 5%).

    Returns:
        The order object returned by the Alpaca API.

    Raises:
        Exception: Propagates any exception from the Alpaca API.
    """
    if position_manager.get_open_position(symbol):
        logger.info("Existing position for %s; skipping order", symbol)
        return None

    stop_price = entry_price * (1 - stop_pct) if side == "buy" else entry_price * (1 + stop_pct)
    target_price = entry_price * (1 + target_pct) if side == "buy" else entry_price * (1 - target_pct)

    try:
        order = alpaca.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="gtc",
            order_class="bracket",
            stop_loss={"stop_price": stop_price},
            take_profit={"limit_price": target_price},
        )
        logger.info("Bracket order submitted: %s", order)
        return order
    except Exception as exc:  # pragma: no cover - network errors or others
        logger.error("Failed to submit bracket order: %s", exc)
        raise
