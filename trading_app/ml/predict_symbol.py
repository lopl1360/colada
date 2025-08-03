import os
from joblib import load
from trading_app.finnhub_client import get_finnhub_bars
from trading_app.indicators import TACalculator


def load_model(symbol):
    model_path = f"models/model_{symbol}.pkl"
    if not os.path.exists(model_path):
        print(f"Model for {symbol} not found.")
        return None
    return load(model_path)


def prepare_latest_features(symbol):
    df = get_finnhub_bars(symbol)
    if df is None or df.empty:
        print(f"No data for {symbol}")
        return None

    df = TACalculator.add_indicators(df)
    df["return"] = df["close"].pct_change()
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["rsi"] = df["rsi_14"]

    df = df.dropna()
    if df.empty:
        print("Insufficient data to generate features.")
        return None

    latest_row = df.iloc[-1]
    return latest_row[["return", "ma5", "ma20", "rsi"]].values.reshape(1, -1)


def predict_symbol(symbol):
    model = load_model(symbol)
    if not model:
        return

    features = prepare_latest_features(symbol)
    if features is None:
        return

    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]
    print(
        f"Prediction for {symbol}: {'UP' if prediction == 1 else 'DOWN'} (Confidence: {prob:.2f})"
    )
