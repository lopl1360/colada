"""Simple trade simulator stub."""
from __future__ import annotations

import pandas as pd


class TradeSimulator:
    """Simulate trades based on signals.

    The simulator supports a single open position at a time and simple
    bracket orders consisting of an entry price, stop‑loss and take‑profit
    levels.  Only very small pieces of state are tracked – enough for unit
    tests and demonstrations.
    """

    def __init__(self, initial_cash: float = 10_000.0) -> None:
        self.initial_cash = initial_cash
        # Information about the currently open position. ``None`` when flat.
        self.position: dict[str, float] | None = None
        # Flag used to trigger a forced exit at the end of the session.
        self._force_exit = False

    # ------------------------------------------------------------------
    # Position management helpers
    # ------------------------------------------------------------------
    def enter_position(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        shares: float,
    ) -> None:
        """Open a new position using bracket order parameters.

        Parameters
        ----------
        entry_price:
            Executed price of the order.
        stop_loss:
            Price at which the position should be closed for a loss.
        take_profit:
            Price at which profits are realised.
        shares:
            Number of shares in the position.  Negative values represent
            short positions.

        Raises
        ------
        RuntimeError
            If a position is already open.
        """

        if self.position is not None:
            raise RuntimeError("A position is already open")

        self.position = {
            "entry_price": float(entry_price),
            "stop_loss": float(stop_loss),
            "take_profit": float(take_profit),
            "shares": float(shares),
        }

    def check_exit_conditions(self, current_price: float) -> tuple[str | None, float]:
        """Evaluate whether the current price triggers an exit.

        Parameters
        ----------
        current_price:
            Latest traded price of the instrument.

        Returns
        -------
        tuple[str | None, float]
            A pair of ``(exit_type, pnl)`` where ``exit_type`` is one of
            ``"stop_loss"``, ``"take_profit"`` or ``"end_session"``.  When
            no exit is triggered the method returns ``(None, 0.0)``.
        """

        if self.position is None:
            return None, 0.0

        entry = self.position["entry_price"]
        stop_loss = self.position["stop_loss"]
        take_profit = self.position["take_profit"]
        shares = self.position["shares"]

        exit_type: str | None = None
        exit_price: float | None = None

        if shares > 0:  # long
            if current_price <= stop_loss:
                exit_type = "stop_loss"
                exit_price = stop_loss
            elif current_price >= take_profit:
                exit_type = "take_profit"
                exit_price = take_profit
        else:  # short
            if current_price >= stop_loss:
                exit_type = "stop_loss"
                exit_price = stop_loss
            elif current_price <= take_profit:
                exit_type = "take_profit"
                exit_price = take_profit

        if exit_type is None and self._force_exit:
            exit_type = "end_session"
            exit_price = current_price

        if exit_type is None or exit_price is None:
            return None, 0.0

        pnl = (exit_price - entry) * shares
        self.position = None
        self._force_exit = False
        return exit_type, pnl

    # ------------------------------------------------------------------
    # Simulation interface
    # ------------------------------------------------------------------
    def simulate(self, data: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
        """Generate portfolio values from price data and signals."""
        portfolio = pd.DataFrame(index=data.index)
        portfolio["cash"] = self.initial_cash
        portfolio["position"] = signals.fillna(0)
        # Placeholder for actual trade execution logic
        portfolio["portfolio_value"] = portfolio["cash"]
        return portfolio

    def end_session(self, current_price: float) -> tuple[str | None, float]:
        """Force-close any open position at the end of the trading session."""
        self._force_exit = True
        return self.check_exit_conditions(current_price)
