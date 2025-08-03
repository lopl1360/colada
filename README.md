# Colada Trading App

Colada is a Python-based trading toolkit that integrates real-time market data, order execution, and machine learning models. It provides a command-line interface for manual trading, tools for building datasets, and utilities for training and running predictive models.

## Features
- **Alpaca and Finnhub integration** for fetching market data and executing orders.
- **Command-line interface** for placing market, limit, and stop orders, checking positions, streaming live trades, and monitoring account balances.
- **Data pipeline** to collect historical prices, engineer features, and build sentiment-augmented datasets.
- **Machine learning workflows** for training XGBoost and LSTM models and generating real-time predictions.
- **Task scheduler** to automate recurring jobs such as data collection.

## Project Layout
- `cli.py` – entry point exposing trading and data commands.
- `trading_app/` – core modules for API clients, data handling, indicators, models, and scheduling.
- `llm_model/` – FinBERT-based sentiment analysis utilities.
- `trading_bot/` – example LSTM trading loop using sentiment and technical features.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure environment variables for external services:
   - `ALPACA_API_KEY` / `ALPACA_SECRET_KEY`
   - `FINNHUB_API_KEY`
   - `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_DB`
3. Optionally start the MySQL database and bot via Docker:
    ```bash
    make build
    make up
    ```

### DigitalOcean droplet preparation

On a fresh droplet you can install Python 3.11, create the virtual environment and install project dependencies with:

```bash
make prepare
```

To run commands inside this environment use:

```bash
make venv CMD="python cli.py alpaca AAPL"
```

Replace the command passed to `CMD` to run any script; use `bash` to drop into an interactive shell.

## Usage
Invoke commands through the CLI:

- Get the latest price from Alpaca:
  ```bash
  python cli.py alpaca AAPL
  ```
- Place a market buy order:
  ```bash
  python cli.py buy AAPL 10
  ```
- Collect and prepare training data:
  ```bash
  python cli.py collect_data
  python cli.py prepare_dataset
  ```
- Train a model and predict a symbol:
  ```bash
  python cli.py train_model
  python cli.py predict_symbol AAPL
  ```
- Run scheduled tasks:
  ```bash
  python cli.py run_scheduler
  ```

## Data and Machine Learning
- `trading_app/data/collect_data.py` fetches historical bars, resamples to 15‑minute intervals, and augments with technical indicators.
- `trading_app/data/prepare_dataset.py` engineers features and generates classification targets.
- `trading_app/data/create_sentiment_dataset.py` merges FinBERT sentiment scores with price data for sentiment-aware models.
- `trading_app/ml/train_model.py` trains XGBoost classifiers and saves them to `models/`.
- `trading_app/ml/train_lstm.py` trains sequence models for use in `trading_bot/run_trade_loop.py`.

## Backtesting
Evaluate a strategy offline with the `backtest` module:

```python
import pandas as pd
from backtest import run_backtest

# price data indexed by timestamp with OHLCV columns
data = pd.read_csv("data.csv", index_col="timestamp", parse_dates=True)

def simple_strategy(row: pd.Series) -> int:
    return 1 if row["close"] > row["open"] else -1

report = run_backtest(data, simple_strategy,
                      config={"initial_cash": 10_000, "export": "results"})
```

`run_backtest` prints an equity curve, trade log and summary statistics. The
returned `report` dictionary also exposes:

- `results` – simulator output for each bar
- `equity_curve` – account value over time
- `trade_log` – each trade's entry/exit and PnL
- `summary` – metrics such as `num_trades`, `win_rate`, `avg_win`, `avg_loss`,
  `max_drawdown` and `sharpe_ratio`

Exporting is optional; when `export` is provided CSV/JSON files with the
equity curve, trades and metrics are written with the given prefix.

## Testing
Run the test suite with:
```bash
pytest
```

## License
This project is provided as-is without warranty. Refer to the repository for license details.
