"""Simple trade simulator with rudimentary execution costs."""

from __future__ import annotations

import random

import pandas as pd


class TradeSimulator:
    """Simulate trades based on signals.

    The simulator supports a single open position at a time and simple
    bracket orders consisting of an entry price, stop‑loss and take‑profit
    levels.  Only very small pieces of state are tracked – enough for unit
    tests and demonstrations.
    """

    def __init__(
        self,
        initial_cash: float = 10_000.0,
        commission_per_share: float = 0.005,
        commission_per_trade: float = 1.0,
        spread: float = 0.0,
        spread_pct: float = 0.0,
    ) -> None:
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission_per_share = commission_per_share
        self.commission_per_trade = commission_per_trade
        self.spread = spread
        self.spread_pct = spread_pct
        # Information about the currently open position. ``None`` when flat.
        self.position: dict[str, float] | None = None
        # Flag used to trigger a forced exit at the end of the session.
        self._force_exit = False

    # ------------------------------------------------------------------
    # Execution cost helpers
    # ------------------------------------------------------------------
    def apply_slippage(self, price: float, direction: str) -> float:
        """Return a price adjusted for random slippage.

        Parameters
        ----------
        price:
            The reference price.
        direction:
            ``"buy"`` or ``"sell"`` indicating order side.
        """

        slippage_pct = random.uniform(0.0001, 0.0005)  # 0.01% – 0.05%
        if direction == "buy":
            return price * (1 + slippage_pct)
        return price * (1 - slippage_pct)

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

        direction = "buy" if shares > 0 else "sell"
        executed_price = self.apply_slippage(entry_price, direction)
        spread_amt = (
            self.spread if self.spread_pct == 0 else executed_price * self.spread_pct
        )
        if direction == "buy":
            executed_price += spread_amt / 2
        else:
            executed_price -= spread_amt / 2
        commission = self.commission_per_share * abs(shares) + self.commission_per_trade
        trade_value = executed_price * shares
        self.cash -= trade_value + commission

        self.position = {
            "entry_price": float(executed_price),
            "stop_loss": float(stop_loss),
            "take_profit": float(take_profit),
            "shares": float(shares),
            "entry_commission": commission,
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

        direction = "sell" if shares > 0 else "buy"
        executed_price = self.apply_slippage(exit_price, direction)
        spread_amt = (
            self.spread if self.spread_pct == 0 else executed_price * self.spread_pct
        )
        if direction == "buy":
            executed_price += spread_amt / 2
        else:
            executed_price -= spread_amt / 2
        commission = self.commission_per_share * abs(shares) + self.commission_per_trade
        trade_value = executed_price * (-shares)
        self.cash -= trade_value + commission

        total_commission = self.position.get("entry_commission", 0.0) + commission
        pnl = (executed_price - entry) * shares - total_commission
        self.position = None
        self._force_exit = False
        return exit_type, pnl

    # ------------------------------------------------------------------
    # Simulation interface
    # ------------------------------------------------------------------
    def simulate(self, data: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
        """Generate portfolio values from price data and signals."""
        portfolio = pd.DataFrame(index=data.index)
        portfolio["cash"] = self.cash
        portfolio["position"] = signals.fillna(0)
        # Placeholder for actual trade execution logic
        portfolio["portfolio_value"] = portfolio["cash"]
        return portfolio

    def end_session(self, current_price: float) -> tuple[str | None, float]:
        """Force-close any open position at the end of the trading session."""
        self._force_exit = True
        return self.check_exit_conditions(current_price)
