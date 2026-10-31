"""Download daily price history from Yahoo Finance."""

import pandas as pd
import yfinance as yf

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def load_prices(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Return daily OHLCV data for `ticker`, oldest row first.

    Raises ValueError if Yahoo Finance returns nothing (bad ticker, no network).
    """
    df = yf.Ticker(ticker.strip().upper()).history(period=period, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No price data found for '{ticker}'.")
    df = df[PRICE_COLUMNS].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "Date"
    return df.dropna()
