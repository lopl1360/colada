from alpaca_trade_api.rest import REST, TimeFrame
from alpaca_trade_api.stream import Stream
import asyncio
from .config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL

alpaca = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)

def get_latest_price(symbol):
    barset = alpaca.get_bars(symbol, TimeFrame.Minute, limit=1)
    return barset[-1].c if barset else None

def submit_market_order(symbol, qty, side):
    """
    Places a market order to buy or sell a given symbol.
    side: 'buy' or 'sell'
    """
    try:
        order = alpaca.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market',
            time_in_force='gtc'  # good till canceled
        )
        return order
    except Exception as e:
        return {"error": str(e)}

def submit_order(symbol, qty, side, order_type='market', limit_price=None, stop_price=None):
    """
    Submit an order with flexible type.
    :param symbol: e.g. 'AAPL'
    :param qty: int
    :param side: 'buy' or 'sell'
    :param order_type: 'market', 'limit', or 'stop'
    :param limit_price: used for 'limit' orders
    :param stop_price: used for 'stop' orders
    """
    try:
        order = alpaca.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type=order_type,
            time_in_force='gtc',
            limit_price=limit_price,
            stop_price=stop_price
        )
        return order
    except Exception as e:
        return {"error": str(e)}

def get_positions():
    """
    Returns a list of current positions from Alpaca.
    """
    try:
        return alpaca.list_positions()
    except Exception as e:
        return {"error": str(e)}

def get_account_summary():
    """
    Returns account summary from Alpaca (cash, buying power, etc.)
    """
    try:
        account = alpaca.get_account()
        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
            "status": account.status
        }
    except Exception as e:
        return {"error": str(e)}

def get_historical_data(symbol, start, end, timeframe=TimeFrame.Minute):
    """
    Fetch historical bars as a DataFrame.
    """
    try:
        bars = alpaca.get_bars(symbol, timeframe, start=start, end=end).df
        # Only apply the filter if 'symbol' is in the index (multi-symbol request)
        if "symbol" in bars.index.names:
            bars = bars[bars.index.get_level_values("symbol") == symbol]

        return bars
    except Exception as e:
        print(f"[Alpaca Error] {e}")
        return None


def stream_live_data(symbol, data_type="trades"):
    """Subscribe to live Alpaca data for a symbol using WebSockets."""

    async def _run():
        stream = Stream(
            ALPACA_API_KEY,
            ALPACA_SECRET_KEY,
            base_url=ALPACA_BASE_URL,
            data_feed="iex",
        )

        if data_type == "trades":
            stream.subscribe_trades(handler, symbol)
        elif data_type == "quotes":
            stream.subscribe_quotes(handler, symbol)
        else:
            stream.subscribe_bars(handler, symbol)

        await stream._run_forever()

    def handler(data):
        print(data)

    asyncio.run(_run())
