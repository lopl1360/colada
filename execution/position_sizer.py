"""Utility for determining trade size based on risk management rules."""


def calculate_position_size(equity, stop_loss_pct, current_price):
    """Return number of shares to buy based on equity and stop loss.

    Args:
        equity: Total account equity in dollars.
        stop_loss_pct: Stop loss distance as a decimal (e.g., 0.1 for 10%).
        current_price: Current price of the asset.

    Returns:
        The integer number of shares to purchase. Returns 0 if calculation
        results in a negative number of shares.
    """
    risk_amount = 0.01 * equity
    stop_distance = stop_loss_pct * current_price
    shares = int(risk_amount / stop_distance)
    return max(shares, 0)
