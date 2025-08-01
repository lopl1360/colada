import pandas as pd
import pandas_ta as ta


class TACalculator:
    """Utility class for adding common technical indicators."""

    @staticmethod
    def add_indicators(bars: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
        """Return a copy of ``bars`` with basic technical indicators added."""
        if close_col not in bars.columns:
            raise ValueError(f"'{close_col}' column not found in DataFrame")

        df = bars.copy()
        df["rsi_14"] = ta.rsi(df[close_col], length=14)

        macd = ta.macd(df[close_col], fast=12, slow=26)
        if isinstance(macd, pd.DataFrame):
            df["macd"] = macd["MACD_12_26_9"]
            df["macd_signal"] = macd["MACDs_12_26_9"]
            df["macd_hist"] = macd["MACDh_12_26_9"]

        df["sma_20"] = df[close_col].rolling(20).mean()
        df["volatility"] = df[close_col].rolling(20).std()

        return df
