import logging
import os
from alpaca_trade_api.rest import REST, APIError

API_KEY = os.getenv("ALPACA_API_KEY", "DUMMY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "DUMMY")
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

api = REST(API_KEY, SECRET_KEY, BASE_URL)

logger = logging.getLogger(__name__)


def get_open_position(symbol):
    try:
        return api.get_position(symbol)
    except APIError as exc:
        logger.error("Error retrieving position for %s: %s", symbol, exc)
        return None
