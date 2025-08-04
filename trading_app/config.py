import os
from dotenv import load_dotenv

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "DUMMY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "DUMMY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
