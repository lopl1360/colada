import os
import pandas as pd
from trading_app.indicators import TACalculator

INPUT_DIR = "data"
OUTPUT_DIR = "features"
SYMBOLS = ["AAPL", "GOOG", "MSFT"]


def prepare_features(symbol):
    path = os.path.join(INPUT_DIR, f"{symbol}_15min.csv")
    if not os.path.exists(path):
        print(f"Missing CSV for {symbol}, skipping.")
        return

    df = pd.read_csv(path, index_col=0, parse_dates=True)

    # Add technical indicators
    df = TACalculator.add_indicators(df)
    df["return"] = df["close"].pct_change()
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["rsi"] = df["rsi_14"]
    df.drop(columns=["rsi_14"], inplace=True)

    # Drop rows with NaN
    df.dropna(inplace=True)

    # Create target: 1 if next close is higher than current close
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # Drop the last row since it has no target
    df = df[:-1]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"features_{symbol}.csv")
    df.to_csv(out_path)
    print(f"[Saved] {symbol} → {out_path}")


def prepare_all():
    for symbol in SYMBOLS:
        prepare_features(symbol)


if __name__ == "__main__":
    prepare_all()
