from trading_app.scheduler import daily
from trading_app.data.collect_data import fetch_all


@daily("14:00")
def collect_history_data():
    """Fetch historical data for configured symbols."""
    fetch_all()
