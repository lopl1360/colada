import os
import logging

from models.lstm_price_predictor import train_lstm
from trading_app.symbols import load_symbols

INPUT_DIR = "features_1min"
SYMBOLS = load_symbols()


logger = logging.getLogger(__name__)


def train_all():
    for symbol in SYMBOLS:
        path = os.path.join(INPUT_DIR, f"{symbol}.csv")
        if not os.path.exists(path):
            logger.warning("Missing data for %s, skipping.", symbol)
            continue
        logger.info("Training LSTM for %s from %s", symbol, path)
        train_lstm(
            path,
            seq_len=60,
            batch_size=32,
            num_epochs=5,
            target_col="target",
            classification=False,
            model_path=f"models/lstm_{symbol}.pt",
        )


if __name__ == "__main__":
    train_all()
