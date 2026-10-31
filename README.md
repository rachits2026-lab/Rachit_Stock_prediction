# Rachit Stock Prediction

A machine-learning app that predicts stock prices from daily price history, with a Streamlit dashboard.

> Educational project only — not financial advice.

## How it works

1. **Data** (`stock_ml/data.py`): downloads daily prices from Yahoo Finance with `yfinance`.
2. **Features** (`stock_ml/features.py`): recent returns, moving-average gaps, volatility, RSI, volume changes.
   The target is the **next day's % return**.
3. **Model** (`stock_ml/model.py`): Random Forest, Gradient Boosting or Ridge Regression.
   It trains on older data and tests on the most recent slice it has never seen, then compares
   against two simple baselines ("tomorrow = today" and "always up").
4. **Forecast**: predicts several days ahead by feeding each prediction back in. Errors grow with each day.
5. **Dashboard** (`app.py`): pick a ticker, model and horizon; view forecast, backtest and feature importance.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then open http://localhost:8501.

## Tests

```bash
pip install pytest
python -m pytest
```
