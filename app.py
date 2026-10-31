"""Streamlit dashboard for stock price prediction.

Run with:  streamlit run app.py
"""

import plotly.graph_objects as go
import streamlit as st

from stock_ml.data import load_prices
from stock_ml.model import MODELS, backtest, feature_importance, forecast

st.set_page_config(page_title="Stock Prediction", page_icon="📈", layout="wide")


@st.cache_data(ttl=3600, show_spinner="Downloading prices…")
def cached_prices(ticker: str, period: str):
    return load_prices(ticker, period)


@st.cache_resource(show_spinner="Training model…")
def cached_backtest(ticker: str, period: str, model_name: str, test_fraction: float):
    return backtest(cached_prices(ticker, period), model_name, test_fraction)


with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker symbol", "AAPL").strip().upper()
    period = st.selectbox("History", ["2y", "5y", "10y", "max"], index=1)
    model_name = st.selectbox("Model", list(MODELS))
    test_pct = st.slider("Test set (% of history)", 10, 40, 20, step=5)
    horizon = st.slider("Forecast days", 1, 30, 10)

st.title(f"📈 {ticker} price prediction")
st.caption(
    "Educational project only — not financial advice. "
    "Stock prices are noisy and past patterns often fail to repeat."
)

try:
    prices = cached_prices(ticker, period)
    result = cached_backtest(ticker, period, model_name, test_pct / 100)
except ValueError as err:
    st.error(str(err))
    st.stop()

future = forecast(prices, result.model, result.features, horizon)
last_close = prices["Close"].iloc[-1]
final_pred = future["predicted_close"].iloc[-1]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Last close", f"${last_close:,.2f}", prices.index[-1].strftime("%b %d, %Y"))
c2.metric(
    f"Forecast in {horizon} day(s)",
    f"${final_pred:,.2f}",
    f"{(final_pred / last_close - 1) * 100:+.2f}%",
)
m = result.metrics
c3.metric(
    "Direction accuracy (test)",
    f"{m['Direction accuracy']:.1%}",
    f"{(m['Direction accuracy'] - m['Always-up accuracy']) * 100:+.1f} pts vs always-up",
)
c4.metric(
    "Price MAE (test)",
    f"${m['MAE (price)']:.2f}",
    f"{m['MAE (price)'] - m['Naive MAE (price)']:+.2f} vs naive",
    delta_color="inverse",
)

tab_forecast, tab_backtest, tab_features, tab_data = st.tabs(
    ["Forecast", "Backtest", "Feature importance", "Raw data"]
)

with tab_forecast:
    recent = prices["Close"].iloc[-120:]
    fig = go.Figure()
    fig.add_scatter(x=recent.index, y=recent, name="Close", line=dict(color="#4C78A8"))
    fig.add_scatter(
        x=[recent.index[-1], *future.index],
        y=[recent.iloc[-1], *future["predicted_close"]],
        name="Forecast",
        line=dict(color="#F58518", dash="dash"),
    )
    fig.update_layout(height=450, yaxis_title="Price ($)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        future.assign(predicted_return=future["predicted_return"] * 100).rename(
            columns={"predicted_close": "Predicted close ($)", "predicted_return": "Daily change (%)"}
        ).style.format("{:.2f}"),
        use_container_width=True,
    )

with tab_backtest:
    st.write(
        f"The model trained on the first {m['Train rows']} days and was tested on the "
        f"last {m['Test rows']} days it had never seen. Each point predicts the next "
        "day's close using only information available the day before."
    )
    p = result.predictions
    fig = go.Figure()
    fig.add_scatter(x=p.index, y=p["actual_close"], name="Actual", line=dict(color="#4C78A8"))
    fig.add_scatter(x=p.index, y=p["predicted_close"], name="Predicted", line=dict(color="#F58518"))
    fig.update_layout(height=450, yaxis_title="Price ($)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "**Naive** means predicting tomorrow's close equals today's close. "
        "**Always-up** means always guessing the price rises. A useful model should beat both."
    )

with tab_features:
    imp = feature_importance(result.model, result.features).iloc[::-1]
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color="#4C78A8"))
    fig.update_layout(height=500, xaxis_title="Importance")
    st.plotly_chart(fig, use_container_width=True)

with tab_data:
    st.dataframe(prices.iloc[::-1], use_container_width=True)
