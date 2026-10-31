"""Turn raw OHLCV prices into model features.

The model predicts the next day's percentage return rather than the raw price,
because returns are roughly stationary while prices trend.
"""

import numpy as np
import pandas as pd

TARGET = "target_return"


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Return a frame of features (and the next-day target) indexed by date.

    The final row has a NaN target because tomorrow is unknown; callers use it
    to forecast.
    """
    close = prices["Close"]
    ret = close.pct_change()
    f = pd.DataFrame(index=prices.index)

    for lag in (1, 2, 3, 5, 10):
        f[f"return_lag_{lag}"] = ret.shift(lag - 1)
    for window in (5, 10, 20, 50):
        f[f"close_vs_sma_{window}"] = close / close.rolling(window).mean() - 1
    f["volatility_10"] = ret.rolling(10).std()
    f["volatility_20"] = ret.rolling(20).std()
    f["rsi_14"] = _rsi(close) / 100
    f["high_low_range"] = (prices["High"] - prices["Low"]) / close
    f["volume_change"] = prices["Volume"].pct_change().replace([np.inf, -np.inf], 0)
    f["volume_vs_avg_20"] = prices["Volume"] / prices["Volume"].rolling(20).mean() - 1
    f["day_of_week"] = prices.index.day_of_week

    f[TARGET] = ret.shift(-1)
    f = f.replace([np.inf, -np.inf], np.nan)
    feature_cols = [c for c in f.columns if c != TARGET]
    return f.dropna(subset=feature_cols)


def feature_columns(features: pd.DataFrame) -> list[str]:
    return [c for c in features.columns if c != TARGET]
