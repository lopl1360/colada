import os
import sys
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from execution import order_submitter


def test_submit_bracket_order_calls_alpaca():
    with patch.object(
        order_submitter.position_manager, "get_open_position", return_value=None
    ):
        with patch.object(
            order_submitter.alpaca, "submit_order", return_value="ok"
        ) as mock_submit:
            result = order_submitter.submit_bracket_order(
                symbol="AAPL",
                qty=1,
                side="buy",
                entry_price=100,
                stop_pct=0.05,
                target_pct=0.1,
            )
    assert result == "ok"
    stop_expected = 100 * (1 - 0.05)
    target_expected = 100 * (1 + 0.1)
    mock_submit.assert_called_once_with(
        symbol="AAPL",
        qty=1,
        side="buy",
        type="market",
        time_in_force="gtc",
        order_class="bracket",
        stop_loss={"stop_price": stop_expected},
        take_profit={"limit_price": target_expected},
    )


def test_submit_bracket_order_skips_when_position_open():
    with patch.object(
        order_submitter.position_manager,
        "get_open_position",
        return_value={"symbol": "AAPL"},
    ):
        with patch.object(order_submitter.alpaca, "submit_order") as mock_submit:
            result = order_submitter.submit_bracket_order(
                symbol="AAPL",
                qty=1,
                side="buy",
                entry_price=100,
                stop_pct=0.05,
                target_pct=0.1,
            )
    assert result is None
    mock_submit.assert_not_called()
