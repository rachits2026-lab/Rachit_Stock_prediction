import numpy as np
import pandas as pd
import pytest

from stock_ml.features import TARGET, build_features, feature_columns
from stock_ml.model import MODELS, backtest, feature_importance, forecast


@pytest.fixture
def prices():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=600)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.015, len(dates)))
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.003, len(dates))),
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, len(dates)).astype(float),
        },
        index=dates,
    )


def test_target_is_next_day_return(prices):
    f = build_features(prices)
    expected = prices["Close"].pct_change().shift(-1).loc[f.index]
    pd.testing.assert_series_equal(f[TARGET], expected, check_names=False)
    assert np.isnan(f[TARGET].iloc[-1])
    assert not f[feature_columns(f)].isna().any().any()


def test_features_do_not_use_future_data(prices):
    full = build_features(prices)
    cut = build_features(prices.iloc[:400])
    cols = feature_columns(full)
    pd.testing.assert_frame_equal(full.loc[cut.index, cols], cut[cols])


@pytest.mark.parametrize("name", list(MODELS))
def test_backtest_and_forecast(prices, name):
    result = backtest(prices, name, test_fraction=0.2)
    assert 0 <= result.metrics["Direction accuracy"] <= 1
    assert result.predictions.notna().all().all()
    assert result.predictions.index.min() > prices.index[int(len(prices) * 0.7)]

    future = forecast(prices, result.model, result.features, days=5)
    assert len(future) == 5
    assert future.index.min() > prices.index[-1]
    assert (future["predicted_close"] > 0).all()

    imp = feature_importance(result.model, result.features)
    assert set(imp.index) == set(result.features)
