import os
import tempfile

os.environ["APP_DB"] = os.path.join(tempfile.mkdtemp(), "test.db")
os.environ["ALERT_CHECK_MINUTES"] = "0"
os.environ["SECRET_KEY"] = "test-secret"

import pytest
from fastapi.testclient import TestClient

from api import main
from stock_ml import alerts


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


def login(client, email="example@gmail.com", password="123"):
    return client.post("/api/auth/login", json={"email": email, "password": password})


def test_pages_redirect_to_login_until_signed_in(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/login"
    assert client.get("/api/me").status_code == 401
    assert login(client).status_code == 200
    assert client.get("/", follow_redirects=False).status_code == 200
    assert client.get("/login", follow_redirects=False).headers["location"] == "/"


def test_demo_login_and_wrong_password(client):
    assert login(client, password="wrong").status_code == 401
    assert login(client, email="EXAMPLE@gmail.com ").json()["email"] == "example@gmail.com"


def test_signup_then_logout(client):
    r = client.post("/api/auth/signup", json={"email": "new@test.com", "password": "pw"})
    assert r.status_code == 200
    assert client.get("/api/me").json()["email"] == "new@test.com"
    assert client.post("/api/auth/signup", json={"email": "new@test.com", "password": "x"}).status_code == 409
    client.post("/api/auth/logout")
    assert client.get("/api/me").status_code == 401


def test_watchlist_and_settings(client):
    login(client)
    assert "TSLA" in client.post("/api/watchlist/tsla").json()
    assert "TSLA" not in client.delete("/api/watchlist/TSLA").json()
    assert client.put("/api/settings", json={"imessage_to": "", "alerts_enabled": True}).status_code == 400
    assert client.put("/api/settings", json={"imessage_to": "+15550001111", "alerts_enabled": True, "alert_threshold": 0.7}).status_code == 200
    me = client.get("/api/me").json()
    assert me["alerts_enabled"] and me["alert_threshold"] == 0.7


def test_featured_companies(client):
    login(client)
    tickers = {c["ticker"] for c in client.get("/api/companies").json()}
    assert {"AAPL", "TSLA", "TSM"} <= tickers


def test_predict_unknown_ticker_is_404(client, monkeypatch):
    def fail(ticker):
        raise ValueError("No price data found")
    monkeypatch.setattr(main, "analyze", fail)
    login(client)
    assert client.get("/api/predict/NOPE").status_code == 404


def test_alert_text_mentions_probability():
    text = alerts.crash_alert_text("AAPL", "Apple", 0.72, 10, 0.05)
    assert "72%" in text and "AAPL" in text and "taking money out" in text
