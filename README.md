# Rachit Stock Prediction

A website that estimates how likely any publicly traded company is to **fall** or **grow** soon.
It combines machine learning on price history with the latest news, social media, and AI (Gemini and Claude).

> Educational project. Predictions are estimates and are often wrong. This is not financial advice,
> and the app never holds or invests money.

## Features

- **Login and sign up.** Demo account: `example@gmail.com` / `123`
- **Teddy bear mascot** on the home page
- **Any company in the world.** About 100 well-known companies are built in, and search covers everything on Yahoo Finance
- **Crash and growth probabilities.** The chance the price drops or rises 5% or more in the next 10 trading days
- **Company value.** Share price, market cap, P/E ratio, analyst target, 52-week range
- **News and social media.** Yahoo Finance headlines and StockTwits posts, scored by Gemini, Claude and a built-in word scorer
- **10-day price forecast** chart
- **iMessage alerts.** Watch a stock and get a text through the Messages app on your Mac when its crash risk crosses your threshold
- **FastAPI backend** with interactive API docs at `/docs`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Optional: AI keys

Without keys the app uses its built-in word scorer. To add AI scoring, set either or both before starting:

```bash
export GEMINI_API_KEY="your-gemini-key"        # https://aistudio.google.com/apikey
export ANTHROPIC_API_KEY="your-anthropic-key"  # https://console.anthropic.com
```

Never commit keys to git. This repo is public.

## Run

```bash
uvicorn api.main:app --reload
```

Open http://localhost:8000 and log in.

In VS Code, you can also open **Run and Debug** and pick **Run website (FastAPI)**.

The original analyst dashboard is still available with `streamlit run app.py`.

## How the prediction works

| Part | File | What it does |
|---|---|---|
| Prices | `stock_ml/data.py` | 10 years of daily prices from Yahoo Finance |
| Features | `stock_ml/features.py` | Returns, moving averages, volatility, RSI, volume |
| Crash/growth model | `stock_ml/risk.py` | Random forest classifiers trained on the stock's own history |
| Price forecast | `stock_ml/model.py` | Predicts the next 10 closing prices |
| News & social | `stock_ml/news.py` | Yahoo Finance RSS headlines, StockTwits posts |
| AI scoring | `stock_ml/ai.py` | Gemini, Claude, built-in word scorer |
| Alerts | `stock_ml/alerts.py` | Sends iMessages through macOS Messages |
| Web API | `api/main.py` | FastAPI routes, login sessions, background alert checks |
| Pages | `web/` | Login, sign up, home, company, alerts |

Final probability = 60% price model + 40% average of the news/AI scores.
Each company page shows how the model scored on past data it never saw (AUC: 0.5 = guessing, 1 = perfect).

## iMessage alerts

1. Open **Alerts**, enter your phone number or Apple ID email, choose a threshold, and turn alerts on.
2. Click **Send test message**. The first time, macOS asks to let Terminal/Python control Messages. Click OK.
3. Tap **Watch** on any company. The server checks every hour (`ALERT_CHECK_MINUTES`) and texts at most once a day per stock.

Alerts only work while the server is running on a Mac signed in to iMessage.

## Tests

```bash
python -m pytest
```
