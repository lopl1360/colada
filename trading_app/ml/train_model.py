import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from joblib import dump

INPUT_DIR = "features"
OUTPUT_DIR = "models"
SYMBOLS = ["AAPL", "GOOG", "MSFT"]

def train_model_for_symbol(symbol):
    file_path = os.path.join(INPUT_DIR, f"features_{symbol}.csv")
    if not os.path.exists(file_path):
        print(f"Missing features for {symbol}, skipping.")
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

    print(f"--- {symbol} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dump(model, os.path.join(OUTPUT_DIR, f"model_{symbol}.pkl"))
    print(f"Saved model to {OUTPUT_DIR}/model_{symbol}.pkl")

def train_all():
    for symbol in SYMBOLS:
        train_model_for_symbol(symbol)

if __name__ == "__main__":
    train_all()
