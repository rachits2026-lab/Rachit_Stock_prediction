import numpy as np
import pandas as pd

from stock_ml.ai import score_locally
from stock_ml.risk import make_labels, predict_risk
from tests.test_model import prices  # noqa: F401  (pytest fixture)


def test_labels_look_only_at_the_next_days():
    dates = pd.bdate_range("2024-01-01", periods=6)
    close = pd.Series([100, 100, 90, 100, 100, 100.0], index=dates)
    df = pd.DataFrame({"Close": close, "High": close, "Low": close})
    labels = make_labels(df, horizon=2, threshold=0.05)
    assert labels["fall"].tolist()[:4] == [1.0, 1.0, 0.0, 0.0]
    assert labels["grow"].iloc[2] == 1.0  # 90 -> 100 is +11%
    assert labels.iloc[-2:].isna().all().all()


def test_predict_risk_returns_probabilities(prices):  # noqa: F811
    r = predict_risk(prices)
    for p in (r.fall_probability, r.grow_probability, r.fall_base_rate, r.grow_base_rate):
        assert 0 <= p <= 1


def test_local_sentiment_direction():
    bad = score_locally([{"title": "Shares plunge after fraud probe and downgrade"}], [])
    good = score_locally([{"title": "Stock soars to record on strong growth"}], [{"text": "", "sentiment": "Bullish"}])
    assert bad.fall_probability > 0.5 > good.fall_probability
    assert score_locally([], []) is None
