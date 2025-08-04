import os
import logging
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from joblib import dump

INPUT_DIR = "features"
OUTPUT_DIR = "models"
SYMBOLS = ["AAPL", "GOOG", "MSFT"]


logger = logging.getLogger(__name__)


def train_model_for_symbol(symbol):
    file_path = os.path.join(INPUT_DIR, f"features_{symbol}.csv")
    if not os.path.exists(file_path):
        logger.warning("Missing features for %s, skipping.", symbol)
        return

    df = pd.read_csv(file_path)

    X = df.drop(columns=["target", "timestamp"], errors="ignore")
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    model = xgb.XGBClassifier(eval_metric="logloss")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    logger.info("--- %s ---", symbol)
    logger.info("Accuracy: %s", accuracy_score(y_test, y_pred))
    logger.info("Precision: %s", precision_score(y_test, y_pred))
    logger.info("Recall: %s", recall_score(y_test, y_pred))
    logger.info("F1 Score: %s", f1_score(y_test, y_pred))
    logger.info("")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dump(model, os.path.join(OUTPUT_DIR, f"model_{symbol}.pkl"))
    logger.info("Saved model to %s/model_%s.pkl", OUTPUT_DIR, symbol)


def train_all():
    for symbol in SYMBOLS:
        train_model_for_symbol(symbol)


if __name__ == "__main__":
    train_all()
