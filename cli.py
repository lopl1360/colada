import click
from trading_app.alpaca_client import get_latest_price
from trading_app.finnhub_client import get_finnhub_quote
from trading_app import init_app
from trading_app.helpers import save_order_if_valid
from trading_app.alpaca_client import stream_live_data

init_app()


@click.group()
def cli():
    pass


@cli.command()
@click.argument("symbol")
def alpaca(symbol):
    """Get latest price from Alpaca."""
    try:
        price = get_latest_price(symbol)
        click.echo(f"{symbol} latest price from Alpaca: {price}")
    except Exception as e:
        click.echo(f"Error: {e}")


@cli.command()
@click.argument("symbol")
def finnhub(symbol):
    """Get latest quote from Finnhub."""
    try:
        quote = get_finnhub_quote(symbol)
        click.echo(f"{symbol} quote from Finnhub: {quote}")
    except Exception as e:
        click.echo(f"Error: {e}")


@cli.command()
@click.argument("symbol")
@click.argument("qty", type=int)
def buy(symbol, qty):
    """Place a market buy order."""
    from trading_app.alpaca_client import submit_market_order

    result = submit_market_order(symbol, qty, "buy")

    if isinstance(result, dict) and "error" in result:
        click.echo(f"Error: {result['error']}")
        return

    click.echo(result)
    save_order_if_valid(result, "buy")


@cli.command()
@click.argument("symbol")
@click.argument("qty", type=int)
def sell(symbol, qty):
    """Place a market sell order."""
    from trading_app.alpaca_client import submit_market_order

    result = submit_market_order(symbol, qty, "sell")

    if isinstance(result, dict) and "error" in result:
        click.echo(f"Error: {result['error']}")
        return

    click.echo(result)
    save_order_if_valid(result, "sell")


@cli.command()
@click.argument("symbol")
@click.argument("qty", type=int)
@click.argument("limit_price", type=float)
def limit_buy(symbol, qty, limit_price):
    """Place a limit buy order."""
    from trading_app.alpaca_client import submit_order

    result = submit_order(symbol, qty, "buy", "limit", limit_price=limit_price)

    if isinstance(result, dict) and "error" in result:
        click.echo(f"Error: {result['error']}")
        return

    click.echo(result)
    save_order_if_valid(result, "buy", fallback_price=limit_price)


@cli.command()
@click.argument("symbol")
@click.argument("qty", type=int)
@click.argument("stop_price", type=float)
def stop_sell(symbol, qty, stop_price):
    """Place a stop-loss sell order."""
    from trading_app.alpaca_client import submit_order

    result = submit_order(symbol, qty, "sell", "stop", stop_price=stop_price)

    if isinstance(result, dict) and "error" in result:
        click.echo(f"Error: {result['error']}")
        return

    click.echo(result)
    save_order_if_valid(result, "sell", fallback_price=stop_price)


@cli.command()
def positions():
    """List current Alpaca positions."""
    from trading_app.alpaca_client import get_positions

    result = get_positions()

    if isinstance(result, dict) and "error" in result:
        click.echo(f"Error: {result['error']}")
    elif not result:
        click.echo("No open positions.")
    else:
        for pos in result:
            click.echo(
                f"{pos.symbol}: {pos.qty} shares at avg price ${pos.avg_entry_price}"
            )


@cli.command()
def balance():
    """Show account balance and equity."""
    from trading_app.alpaca_client import get_account_summary

    summary = get_account_summary()

    if "error" in summary:
        click.echo(f"Error: {summary['error']}")
    else:
        click.echo(f"Equity:          ${summary['equity']}")
        click.echo(f"Cash:            ${summary['cash']}")
        click.echo(f"Buying Power:    ${summary['buying_power']}")
        click.echo(f"Portfolio Value: ${summary['portfolio_value']}")
        click.echo(f"Account Status:  {summary['status']}")


@cli.command()
def market_status():
    """Show whether the market is currently open."""
    from trading_app.alpaca_client import is_market_open

    status = "open" if is_market_open() else "closed"
    click.echo(f"Market is currently {status}.")


@cli.command()
def collect_data():
    """Download and save historical 15min data for training."""
    from trading_app.data.collect_data import fetch_all

    fetch_all()


@cli.command()
def prepare_dataset():
    """Prepare ML training data with features and labels."""
    from trading_app.data.prepare_dataset import prepare_all

    prepare_all()


@cli.command()
@click.argument("symbol")
@click.option("--window", default=60, help="Rolling window size")
@click.option("--output", default="sentiment_dataset.csv", help="Output CSV path")
def sentiment_dataset(symbol, window, output):
    """Create sentiment-augmented dataset for `symbol`."""
    from trading_app.data.create_sentiment_dataset import build_dataset

    df = build_dataset(symbol.upper(), window=window, output_csv=output)
    click.echo(f"Saved {len(df)} rows to {output}")


@cli.command()
def train_model():
    """Train XGBoost models for all available symbols."""
    from trading_app.ml.train_model import train_all

    train_all()


@cli.command()
def train_lstm():
    """Train LSTM models for all available symbols."""
    from trading_app.ml.train_lstm import train_all

    train_all()


@cli.command()
@click.argument("symbol")
def predict_symbol(symbol):
    """Predict if the given symbol will go UP or DOWN using the latest model."""
    from trading_app.ml.predict_symbol import predict_symbol as predict

    predict(symbol.upper())


@cli.command()
@click.argument("symbol")
def stream(symbol):
    """Print live trade updates for the given symbol."""
    stream_live_data(symbol)


@cli.command()
def run_scheduler():
    """Run scheduled tasks defined in trading_app.tasks."""
    from trading_app.scheduler import run

    click.echo("Starting scheduler... Press Ctrl+C to exit.")
    run()


if __name__ == "__main__":
    cli()
