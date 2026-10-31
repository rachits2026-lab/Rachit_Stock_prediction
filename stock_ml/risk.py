"""Estimate the probability that a stock falls (or grows) sharply soon.

"Fall" means the price drops at least THRESHOLD below today's close at some
point in the next HORIZON trading days; "grow" means it rises at least
THRESHOLD above it. Two classifiers learn these from the stock's own history.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from .features import build_features, feature_columns

HORIZON = 10
THRESHOLD = 0.05


@dataclass
class RiskResult:
    fall_probability: float
    grow_probability: float
    fall_base_rate: float  # how often a fall happened historically
    grow_base_rate: float
    fall_auc: float | None  # test-set ROC AUC; 0.5 = no better than chance
    grow_auc: float | None
    horizon_days: int = HORIZON
    threshold: float = THRESHOLD


def make_labels(prices: pd.DataFrame, horizon: int = HORIZON, threshold: float = THRESHOLD) -> pd.DataFrame:
    """Label each day with whether a sharp fall / rise happens in the next `horizon` days.

    The last `horizon` rows have no complete future and are left as NaN.
    """
    close = prices["Close"]
    future_low = prices["Low"][::-1].rolling(horizon, min_periods=horizon).min()[::-1].shift(-1)
    future_high = prices["High"][::-1].rolling(horizon, min_periods=horizon).max()[::-1].shift(-1)
    return pd.DataFrame(
        {
            "fall": (future_low / close - 1 <= -threshold).astype(float).where(future_low.notna()),
            "grow": (future_high / close - 1 >= threshold).astype(float).where(future_high.notna()),
        },
        index=prices.index,
    )


def _classifier():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )


def _fit_one(X: pd.DataFrame, y: pd.Series, latest: pd.DataFrame, test_fraction: float):
    split = int(len(X) * (1 - test_fraction))
    auc = None
    if y.iloc[:split].nunique() == 2 and y.iloc[split:].nunique() == 2:
        model = _classifier().fit(X.iloc[:split], y.iloc[:split])
        auc = float(roc_auc_score(y.iloc[split:], model.predict_proba(X.iloc[split:])[:, 1]))
    if y.nunique() < 2:
        return float(y.iloc[0]), auc
    model = _classifier().fit(X, y)
    return float(model.predict_proba(latest)[0, 1]), auc


def predict_risk(prices: pd.DataFrame, test_fraction: float = 0.2) -> RiskResult:
    features = build_features(prices)
    cols = feature_columns(features)
    labels = make_labels(prices).loc[features.index]
    labelled = labels.dropna().index
    if len(labelled) < 250:
        raise ValueError("Not enough price history to estimate crash risk (need about a year).")

    X = features.loc[labelled, cols]
    latest = features[cols].iloc[[-1]]
    fall_p, fall_auc = _fit_one(X, labels.loc[labelled, "fall"], latest, test_fraction)
    grow_p, grow_auc = _fit_one(X, labels.loc[labelled, "grow"], latest, test_fraction)
    return RiskResult(
        fall_probability=fall_p,
        grow_probability=grow_p,
        fall_base_rate=float(np.mean(labels.loc[labelled, "fall"])),
        grow_base_rate=float(np.mean(labels.loc[labelled, "grow"])),
        fall_auc=fall_auc,
        grow_auc=grow_auc,
    )
