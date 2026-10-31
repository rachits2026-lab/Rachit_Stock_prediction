"""Train, backtest and forecast with scikit-learn regressors."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import TARGET, build_features, feature_columns

MODELS = {
    "Random Forest": lambda: RandomForestRegressor(
        n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": lambda: GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=42
    ),
    "Ridge Regression": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
}


@dataclass
class BacktestResult:
    model: object
    features: list[str]
    predictions: pd.DataFrame  # columns: actual_close, predicted_close, actual_return, predicted_return
    metrics: dict[str, float]


def backtest(prices: pd.DataFrame, model_name: str, test_fraction: float = 0.2) -> BacktestResult:
    """Train on the earliest data and evaluate on the most recent `test_fraction`.

    The split is chronological so the model never sees the future.
    """
    data = build_features(prices).dropna(subset=[TARGET])
    cols = feature_columns(data)
    split = int(len(data) * (1 - test_fraction))
    if split < 100 or len(data) - split < 10:
        raise ValueError("Not enough history to train and test. Pick a longer period.")
    train, test = data.iloc[:split], data.iloc[split:]

    model = MODELS[model_name]()
    model.fit(train[cols], train[TARGET])
    pred_ret = model.predict(test[cols])

    close = prices["Close"]
    today_close = close.loc[test.index]
    next_close = close.shift(-1).loc[test.index]
    next_dates = close.index[close.index.get_indexer(test.index) + 1]
    predictions = pd.DataFrame(
        {
            "actual_close": next_close.values,
            "predicted_close": today_close.values * (1 + pred_ret),
            "actual_return": test[TARGET].values,
            "predicted_return": pred_ret,
        },
        index=next_dates,
    )

    actual = test[TARGET].values
    metrics = {
        "MAE (price)": mean_absolute_error(predictions["actual_close"], predictions["predicted_close"]),
        "Naive MAE (price)": mean_absolute_error(predictions["actual_close"], today_close.values),
        "Direction accuracy": float(np.mean(np.sign(pred_ret) == np.sign(actual))),
        "Always-up accuracy": float(np.mean(actual > 0)),
        "Train rows": len(train),
        "Test rows": len(test),
    }

    # Refit on all labelled data so forecasts use the newest information.
    model = MODELS[model_name]()
    model.fit(data[cols], data[TARGET])
    return BacktestResult(model, cols, predictions, metrics)


def forecast(prices: pd.DataFrame, model, cols: list[str], days: int) -> pd.DataFrame:
    """Predict the next `days` closing prices by feeding each prediction back in.

    Uncertainty compounds with every step, so long horizons are rough guesses.
    """
    history = prices.copy()
    rows = []
    for _ in range(days):
        latest = build_features(history).iloc[[-1]]
        pred_ret = float(model.predict(latest[cols])[0])
        last = history.iloc[-1]
        next_close = last["Close"] * (1 + pred_ret)
        next_date = history.index[-1] + pd.offsets.BDay(1)
        history.loc[next_date] = {
            "Open": last["Close"],
            "High": max(last["Close"], next_close),
            "Low": min(last["Close"], next_close),
            "Close": next_close,
            "Volume": history["Volume"].iloc[-20:].mean(),
        }
        rows.append({"Date": next_date, "predicted_close": next_close, "predicted_return": pred_ret})
    return pd.DataFrame(rows).set_index("Date")


def feature_importance(model, cols: list[str]) -> pd.Series:
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(est, "feature_importances_"):
        values = est.feature_importances_
    else:
        values = np.abs(est.coef_)
    return pd.Series(values, index=cols).sort_values(ascending=False)
