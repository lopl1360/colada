"""Market timer utilities using Alpaca's clock."""

import logging
from alpaca_trade_api.rest import REST

logger = logging.getLogger(__name__)


def should_exit_positions(api: REST) -> bool:
    """Return ``True`` if the market is closing within five minutes.

    Parameters
    ----------
    api:
        Instance of ``alpaca_trade_api.rest.REST``.
    """

    try:
        clock = api.get_clock()
        seconds_to_close = (clock.next_close - clock.timestamp).total_seconds()
        return seconds_to_close < 300
    except Exception as exc:  # pragma: no cover - relies on external API
        logger.error("Failed to fetch market clock: %s", exc)
        return False
