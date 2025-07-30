import requests
import pandas as pd
from datetime import datetime, timedelta
from .config import FINNHUB_API_KEY


def get_finnhub_quote(symbol):
    """Get the latest quote (current price, high, low, etc.)"""
    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def get_finnhub_bars(symbol, resolution=60, count=24):
    """
    Get recent OHLCV bars (default: last 96 x 15min bars = 24 hours).
    Returns a DataFrame with timestamp index and open/high/low/close/volume.
    """
    end = int((datetime.now()  - timedelta(minutes=60)).timestamp())
    start = end - (int(resolution) * 60 * count)
    resolution = "D"
    start = "1738655051"
    end = "1738741451"
    url = (
        f"https://finnhub.io/api/v1/stock/candle"
        f"?symbol={symbol}&resolution={resolution}&from={start}&to={end}&token={FINNHUB_API_KEY}"
    )

    response = requests.get(url)
    data = response.json()

    if data.get("s") != "ok":
        print(f"[Finnhub Error] {data}")
        return None

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(data["t"], unit="s"),
        "open": data["o"],
        "high": data["h"],
        "low": data["l"],
        "close": data["c"],
        "volume": data["v"]
    }).set_index("timestamp")

    return df
