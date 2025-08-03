"""Market timer utilities using Alpaca's clock."""

from alpaca_trade_api.rest import REST


def should_exit_positions(api: REST) -> bool:
    """Return ``True`` if the market is closing within five minutes.

    Parameters
    ----------
    api:
        Instance of ``alpaca_trade_api.rest.REST``.
    """

    clock = api.get_clock()
    seconds_to_close = (clock.next_close - clock.timestamp).total_seconds()
    return seconds_to_close < 300
