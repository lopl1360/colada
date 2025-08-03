"""Tests for feature engineering utilities."""

import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.feature_engineering import (
    add_technical_indicators,
    merge_sentiment_features,
)


def test_add_technical_indicators_adds_expected_columns():
    df = pd.DataFrame({"close": range(1, 101)})
    result = add_technical_indicators(df)

    expected = {"rsi", "macd", "macd_signal", "macd_hist", "sma_20", "volatility"}
    assert expected.issubset(result.columns)
    assert not result["rsi"].iloc[-1:].isna().any()
    assert len(result) == len(df)


def test_add_technical_indicators_requires_close_column():
    df = pd.DataFrame({"open": [1, 2, 3]})
    with pytest.raises(ValueError):
        add_technical_indicators(df)


def test_merge_sentiment_features_with_series():
    df = pd.DataFrame({"feature": [1, 2, 3]})
    sentiment = pd.Series([0.1, 0.2, 0.3], name="sentiment")

    result = merge_sentiment_features(df, sentiment)

    assert "sentiment" in result.columns
    assert result["sentiment"].iloc[0] == pytest.approx(0.1)


def test_merge_sentiment_features_with_scalar():
    df = pd.DataFrame({"feature": [1, 2, 3]})

    result = merge_sentiment_features(df, 0.5)

    assert (result["sentiment"] == 0.5).all()
