"""Utility for building a sentiment-augmented dataset."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from llm_model.sentiment_analyzer import SentimentAnalyzer
from trading_app.indicators import TACalculator
from utils.feature_engineering import merge_sentiment_features
import logging

logger = logging.getLogger(__name__)


def fetch_ohlcv(symbol: str, days: int = 30) -> pd.DataFrame:
    """Download 1-min OHLCV data using yfinance."""
    df = yf.download(symbol, period=f"{days}d", interval="1m", progress=False)
    if df.empty:
        raise ValueError("No price data returned")
    df.index = pd.to_datetime(df.index)
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    return df[["open", "high", "low", "close", "volume"]]


def fetch_news_sentiment(symbol: str) -> pd.Series:
    """Return a Series indexed by timestamp with sentiment scores."""
    ticker = yf.Ticker(symbol)
    news_items = ticker.news  # type: ignore[attr-defined]
    if not news_items:
        return pd.Series(dtype=float, name="sentiment")

    analyzer = SentimentAnalyzer()
    rows = []
    for item in news_items:
        ts = item.get("providerPublishTime") or item.get("publishTime")
        if ts is None:
            continue
        ts = pd.to_datetime(ts, unit="s")
        text = f"{item.get('title', '')} {item.get('summary', '')}".strip()
        score = analyzer.get_sentiment_score(text)
        rows.append({"timestamp": ts, "sentiment": score})

    if not rows:
        return pd.Series(dtype=float, name="sentiment")

    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    df = df.resample("1min").last().ffill()
    return df["sentiment"]


def build_dataset(
    symbol: str, window: int = 60, output_csv: str | None = None
) -> pd.DataFrame:
    """Create dataset with technical indicators and sentiment scores."""
    price = fetch_ohlcv(symbol)
    price = TACalculator.add_indicators(price)

    sentiment = fetch_news_sentiment(symbol)
    price = merge_sentiment_features(price, sentiment)
    price["sentiment"].fillna(method="ffill", inplace=True)

    price["target"] = price["close"].shift(-1) - price["close"]
    price.dropna(inplace=True)

    if output_csv:
        price.to_csv(output_csv)

    return price


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create sentiment dataset")
    parser.add_argument("symbol", help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--window", type=int, default=60, help="Rolling window size")
    parser.add_argument(
        "--output", default="sentiment_dataset.csv", help="CSV file path"
    )
    args = parser.parse_args()

    df = build_dataset(args.symbol, window=args.window, output_csv=args.output)
    logger.info("Saved %s rows to %s", len(df), args.output)
