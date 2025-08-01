import os

from models.lstm_price_predictor import train_lstm

INPUT_DIR = "features_1min"
SYMBOLS = ["AAPL", "GOOG", "MSFT"]


def train_all():
    for symbol in SYMBOLS:
        path = os.path.join(INPUT_DIR, f"{symbol}.csv")
        if not os.path.exists(path):
            print(f"Missing data for {symbol}, skipping.")
            continue
        print(f"Training LSTM for {symbol} from {path}")
        train_lstm(path, seq_len=60, batch_size=32, num_epochs=5,
                   target_col="target", classification=False,
                   model_path=f"models/lstm_{symbol}.pt")


if __name__ == "__main__":
    train_all()
