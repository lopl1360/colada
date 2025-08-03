"""Live trading loop using price, technical indicators, and news sentiment.

This module stitches together the price feed, sentiment model and the trained
price predictor.  Each minute we obtain the latest price and a piece of news
text, transform them into model features and feed the most recent ``N``
observations to the LSTM model.  The sign of the prediction decides whether we
would go long, short or simply hold the position.

The function :func:`run_trade_loop` is intentionally written in a way that is
simple to test.  It accepts callables for fetching prices and news so that unit
tests can supply deterministic stubs instead of performing real network calls.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Iterable, Optional

import numpy as np
import pandas as pd

try:  # pragma: no cover - torch may not be installed in some environments
    import torch
except Exception:  # pragma: no cover - handled gracefully in runtime
    torch = None  # type: ignore

from llm_model.sentiment_analyzer import SentimentAnalyzer
from models.lstm_price_predictor import LSTMPricePredictor
from utils.feature_engineering import add_technical_indicators, merge_sentiment_features
from execution.market_timer import should_exit_positions
from trading_app.alpaca_client import alpaca as alpaca_api


FeatureVector = np.ndarray


def _default_price_fetcher(symbol: str) -> float:
    """Return the latest price using the Alpaca client if available."""
    try:  # pragma: no cover - requires external service
        from trading_app.alpaca_client import get_latest_price

        price = get_latest_price(symbol)
        return float(price) if price is not None else float("nan")
    except Exception:
        # In offline environments or if Alpaca is not configured, fall back to NaN
        return float("nan")


def _default_news_fetcher(symbol: str) -> str:
    """Fetch the most recent news headline using yfinance if available."""
    try:  # pragma: no cover - network access may not be available
        import yfinance as yf

        news_items = yf.Ticker(symbol).news  # type: ignore[attr-defined]
        if news_items:
            item = news_items[0]
            return f"{item.get('title', '')} {item.get('summary', '')}".strip()
    except Exception:
        pass
    return ""  # no news available


def _decide_action(prediction: float, threshold: float = 0.0) -> str:
    """Map a numeric model prediction to a trading action."""
    if prediction > threshold:
        return "long"
    if prediction < -threshold:
        return "short"
    return "hold"


def run_trade_loop(
    symbol: str,
    model_path: str = "models/lstm_model.pt",
    seq_len: int = 60,
    price_fetcher: Optional[Callable[[str], float]] = None,
    news_fetcher: Optional[Callable[[str], str]] = None,
    sleep_seconds: float = 60.0,
    max_iterations: Optional[int] = None,
) -> None:
    """Run the main trading loop.

    Parameters
    ----------
    symbol:
        Ticker symbol to trade, e.g. ``"AAPL"``.
    model_path:
        Path to the trained LSTM model weights.  If the file does not exist the
        loop will still run but predictions will be random (model weights are
        uninitialised).
    seq_len:
        Number of historical minutes to feed into the model.
    price_fetcher / news_fetcher:
        Optional callables returning the latest price and news text respectively.
        When ``None`` sensible defaults are used that rely on Alpaca/yfinance.
    sleep_seconds:
        Delay between iterations.  In live trading this would be 60 seconds but
        tests can override it to a small value.
    max_iterations:
        When provided, the loop will exit after this many iterations.  This is
        useful for unit tests to avoid an infinite loop.
    """

    if torch is None:  # pragma: no cover - depends on torch availability
        raise ImportError("PyTorch is required to run the trading loop")

    price_fetcher = price_fetcher or _default_price_fetcher
    news_fetcher = news_fetcher or _default_news_fetcher

    sentiment_model = SentimentAnalyzer()

    price_history = pd.DataFrame(columns=["close"])  # type: ignore[pd-error]
    feature_window: Deque[FeatureVector] = deque(maxlen=seq_len)
    model: Optional[LSTMPricePredictor] = None

    iterations = 0
    while max_iterations is None or iterations < max_iterations:
        iterations += 1

        if should_exit_positions(alpaca_api):
            alpaca_api.close_all_positions()
            break

        price = float(price_fetcher(symbol))
        news_text = news_fetcher(symbol)
        sentiment_score = float(sentiment_model.get_sentiment_score(news_text))

        # Update price history
        price_history.loc[pd.Timestamp.utcnow(), "close"] = price

        # Build feature set with technical indicators and sentiment
        features_df = add_technical_indicators(price_history)
        features_df = merge_sentiment_features(features_df, sentiment_score)
        latest = features_df.iloc[-1][
            [
                "close",
                "rsi",
                "macd",
                "macd_signal",
                "macd_hist",
                "sma_20",
                "volatility",
                "sentiment",
            ]
        ].fillna(0.0)

        feature_vec = latest.to_numpy(dtype=np.float32)
        feature_window.append(feature_vec)

        if len(feature_window) == seq_len:
            # Lazily create the model once the input dimension is known
            if model is None:
                input_dim = feature_vec.shape[0]
                model = LSTMPricePredictor(input_dim)
                if Path(model_path).exists():
                    state = torch.load(model_path, map_location="cpu")
                    model.load_state_dict(state)
                model.eval()

            x = torch.tensor([list(feature_window)], dtype=torch.float32)
            with torch.no_grad():
                prediction = model(x).item()
            action = _decide_action(prediction)
            print(f"{symbol} price={price:.2f} pred={prediction:.4f} -> {action}")
        else:
            remaining = seq_len - len(feature_window)
            print(f"Collecting data... {remaining} minutes remaining")

        time.sleep(sleep_seconds)


__all__ = ["run_trade_loop"]
