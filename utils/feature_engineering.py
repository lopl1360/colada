import pandas as pd
import pandas_ta as ta
from typing import Union


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return `df` with RSI, MACD, SMA and rolling volatility."""
    if "close" not in df.columns:
        raise ValueError("'close' column not found in DataFrame")

    df = df.copy()

    # Relative Strength Index
    df["rsi"] = ta.rsi(df["close"], length=14)

    # Moving Average Convergence Divergence
    macd = ta.macd(df["close"], fast=12, slow=26)
    if isinstance(macd, pd.DataFrame):
        df["macd"] = macd["MACD_12_26_9"]
        df["macd_signal"] = macd["MACDs_12_26_9"]
        df["macd_hist"] = macd["MACDh_12_26_9"]
    else:
        df["macd"] = macd

    # Simple Moving Average
    df["sma_20"] = ta.sma(df["close"], length=20)

    # Rolling volatility using standard deviation of closing prices
    df["volatility"] = df["close"].rolling(window=20).std()

    return df


def merge_sentiment_features(
    df: pd.DataFrame, sentiment_score: Union[pd.Series, pd.DataFrame, float, int]
) -> pd.DataFrame:
    """Merge sentiment information with the provided feature DataFrame."""
    df = df.copy()

    if isinstance(sentiment_score, (pd.Series, pd.DataFrame)):
        df = df.join(sentiment_score)
    else:
        df["sentiment"] = sentiment_score

    return df
