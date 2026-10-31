"""Combine price models, news and AI sentiment into one company report."""

import threading
import time

import yfinance as yf

from stock_ml.ai import analyze_sentiment
from stock_ml.data import load_prices
from stock_ml.model import backtest, forecast
from stock_ml.news import fetch_news, fetch_social
from stock_ml.risk import predict_risk

CACHE_SECONDS = 30 * 60
ML_WEIGHT = 0.6  # share of the final probability from the price model; the rest is news/AI
FORECAST_DAYS = 10

_cache: dict[str, tuple[float, dict]] = {}
_lock = threading.Lock()


def _company_info(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:  # yfinance raises many network/parse error types
        info = {}
    return {
        "name": info.get("longName") or info.get("shortName") or ticker,
        "sector": info.get("sector") or "",
        "industry": info.get("industry") or "",
        "currency": info.get("currency") or "USD",
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "analyst_target": info.get("targetMeanPrice"),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
        "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
        "summary": (info.get("longBusinessSummary") or "")[:600],
    }


def _verdict(fall: float, grow: float) -> str:
    if fall >= 0.6 and fall > grow:
        return "High risk of falling"
    if grow >= 0.6 and grow > fall:
        return "Likely to grow"
    if fall > grow + 0.1:
        return "Leaning down"
    if grow > fall + 0.1:
        return "Leaning up"
    return "Uncertain"


def analyze(ticker: str, use_cache: bool = True) -> dict:
    ticker = ticker.strip().upper()
    with _lock:
        hit = _cache.get(ticker)
    if use_cache and hit and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1]

    prices = load_prices(ticker, "10y")
    info = _company_info(ticker)
    risk = predict_risk(prices)
    price_model = backtest(prices, "Random Forest")
    future = forecast(prices, price_model.model, price_model.features, FORECAST_DAYS)

    news = fetch_news(ticker)
    posts = fetch_social(ticker)
    ai = analyze_sentiment(ticker, info["name"], news, posts)

    fall, grow = risk.fall_probability, risk.grow_probability
    if ai:
        ai_fall = sum(a["fall_probability"] for a in ai) / len(ai)
        ai_grow = sum(a["grow_probability"] for a in ai) / len(ai)
        fall = ML_WEIGHT * fall + (1 - ML_WEIGHT) * ai_fall
        grow = ML_WEIGHT * grow + (1 - ML_WEIGHT) * ai_grow

    recent = prices["Close"].iloc[-252:]
    report = {
        "ticker": ticker,
        **info,
        "price": float(prices["Close"].iloc[-1]),
        "price_date": prices.index[-1].strftime("%Y-%m-%d"),
        "day_change": float(prices["Close"].iloc[-1] / prices["Close"].iloc[-2] - 1),
        "fall_probability": fall,
        "grow_probability": grow,
        "verdict": _verdict(fall, grow),
        "model": {
            "fall_probability": risk.fall_probability,
            "grow_probability": risk.grow_probability,
            "fall_base_rate": risk.fall_base_rate,
            "grow_base_rate": risk.grow_base_rate,
            "fall_auc": risk.fall_auc,
            "grow_auc": risk.grow_auc,
            "horizon_days": risk.horizon_days,
            "threshold": risk.threshold,
        },
        "ai": ai,
        "news": news,
        "social": posts,
        "history": {
            "dates": [d.strftime("%Y-%m-%d") for d in recent.index],
            "close": [round(float(v), 2) for v in recent],
        },
        "forecast": {
            "dates": [d.strftime("%Y-%m-%d") for d in future.index],
            "close": [round(float(v), 2) for v in future["predicted_close"]],
            "direction_accuracy": price_model.metrics["Direction accuracy"],
        },
    }
    with _lock:
        _cache[ticker] = (time.time(), report)
    return report
