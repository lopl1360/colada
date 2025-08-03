from datetime import datetime, timedelta
import os
from trading_app.alpaca_client import get_historical_data
from trading_app.indicators import TACalculator

SYMBOLS = ["AAPL", "GOOG", "MSFT"]
OUTPUT_DIR = "data"
DAYS = 30


def fetch_data(symbol):
    """Download recent data for ``symbol`` and store a 15-minute file."""
    start = (datetime.now() - timedelta(days=DAYS + 1)).strftime("%Y-%m-%d")
    end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Getting data for {symbol} from: {start} and {end}")
    df = get_historical_data(symbol, start, end)

    if df is None or df.empty:
        print(f"No data for {symbol}")
        return

    # Resample to 15-min
    df_15 = (
        df.resample("15min")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )

    # Add common technical indicators
    df_15 = TACalculator.add_indicators(df_15)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_15.to_csv(f"{OUTPUT_DIR}/{symbol}_15min.csv")
    print(f"[Saved] {symbol} → {OUTPUT_DIR}/{symbol}_15min.csv")


def fetch_all():
    for symbol in SYMBOLS:
        fetch_data(symbol)
