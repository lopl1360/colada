import os
import logging
import pandas as pd
from trading_app.indicators import TACalculator
from trading_app.symbols import load_symbols

INPUT_DIR = "data"
OUTPUT_DIR = "features"
SYMBOLS = load_symbols()


logger = logging.getLogger(__name__)


def prepare_features(symbol):
    path = os.path.join(INPUT_DIR, f"{symbol}_15min.csv")
    if not os.path.exists(path):
        logger.warning("Missing CSV for %s, skipping.", symbol)
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
    logger.info("[Saved] %s → %s", symbol, out_path)


def prepare_all():
    for symbol in SYMBOLS:
        prepare_features(symbol)


if __name__ == "__main__":
    prepare_all()
