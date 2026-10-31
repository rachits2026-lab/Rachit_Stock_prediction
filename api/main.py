"""FastAPI app: login, company search, predictions, watchlist and iMessage alerts.

Run with:  uvicorn api.main:app --reload
"""

import asyncio
import datetime as dt
import logging
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.sessions import SessionMiddleware

from stock_ml import alerts, companies

from . import db
from .analysis import analyze

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
ALERT_CHECK_MINUTES = float(os.getenv("ALERT_CHECK_MINUTES", "60"))
log = logging.getLogger("uvicorn.error")


def _secret_key() -> str:
    if key := os.getenv("SECRET_KEY"):
        return key
    path = db.DB_PATH.parent / "secret.key"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_hex(32))
    return path.read_text().strip()


async def alert_loop() -> None:
    """Every ALERT_CHECK_MINUTES, check each watched stock and text its owner at most once a day."""
    while True:
        today = dt.date.today().isoformat()
        for row in await asyncio.to_thread(db.alert_targets):
            if row["last_alert_date"] == today:
                continue
            try:
                report = await asyncio.to_thread(analyze, row["ticker"])
            except Exception as err:  # one bad ticker must not stop the loop
                log.warning("Alert check failed for %s: %s", row["ticker"], err)
                continue
            if report["fall_probability"] >= row["alert_threshold"]:
                text = alerts.crash_alert_text(
                    report["ticker"], report["name"], report["fall_probability"],
                    report["model"]["horizon_days"], report["model"]["threshold"],
                )
                ok, error = await asyncio.to_thread(alerts.send_imessage, row["imessage_to"], text)
                if ok:
                    db.mark_alerted(row["user_id"], row["ticker"], today)
                else:
                    log.warning("iMessage alert failed: %s", error)
        await asyncio.sleep(ALERT_CHECK_MINUTES * 60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    task = asyncio.create_task(alert_loop()) if ALERT_CHECK_MINUTES > 0 else None
    yield
    if task:
        task.cancel()


app = FastAPI(title="Rachit Stock Prediction", lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=_secret_key(), same_site="lax", max_age=7 * 24 * 3600)
app.mount("/static", StaticFiles(directory=WEB / "static"), name="static")


# ---------- auth ----------

class Credentials(BaseModel):
    email: str = Field(min_length=3, max_length=200)
    password: str = Field(min_length=1, max_length=200)


def current_user(request: Request):
    user_id = request.session.get("user_id")
    user = db.get_user(user_id) if user_id else None
    if not user:
        raise HTTPException(401, "Please log in.")
    return user


@app.post("/api/auth/login")
def login(creds: Credentials, request: Request):
    user = db.verify_password(creds.email, creds.password)
    if not user:
        raise HTTPException(401, "Wrong email or password.")
    request.session["user_id"] = user["id"]
    return {"email": user["email"]}


@app.post("/api/auth/signup")
def signup(creds: Credentials, request: Request):
    if "@" not in creds.email:
        raise HTTPException(400, "Please enter a valid email address.")
    if db.get_user_by_email(creds.email):
        raise HTTPException(409, "An account with this email already exists.")
    request.session["user_id"] = db.create_user(creds.email, creds.password)
    return {"email": creds.email.strip().lower()}


@app.post("/api/auth/logout")
def logout(request: Request):
    request.session.clear()
    return {"ok": True}


@app.get("/api/me")
def me(user=Depends(current_user)):
    return {
        "email": user["email"],
        "imessage_to": user["imessage_to"],
        "alerts_enabled": bool(user["alerts_enabled"]),
        "alert_threshold": user["alert_threshold"],
    }


# ---------- companies & predictions ----------

@app.get("/api/companies")
def list_companies(q: str = "", user=Depends(current_user)):
    return companies.search(q, limit=20) if q else companies.featured()


@app.get("/api/predict/{ticker}")
def predict(ticker: str, user=Depends(current_user)):
    try:
        return analyze(ticker)
    except ValueError as err:
        raise HTTPException(404, str(err))


# ---------- watchlist ----------

@app.get("/api/watchlist")
def get_watchlist(user=Depends(current_user)):
    return db.watchlist(user["id"])


@app.post("/api/watchlist/{ticker}")
def add_watch(ticker: str, user=Depends(current_user)):
    db.add_to_watchlist(user["id"], ticker.strip().upper())
    return db.watchlist(user["id"])


@app.delete("/api/watchlist/{ticker}")
def remove_watch(ticker: str, user=Depends(current_user)):
    db.remove_from_watchlist(user["id"], ticker.strip().upper())
    return db.watchlist(user["id"])


# ---------- alert settings ----------

class AlertSettings(BaseModel):
    imessage_to: str = Field(default="", max_length=100)
    alerts_enabled: bool = False
    alert_threshold: float = Field(default=0.6, ge=0.1, le=0.95)


@app.put("/api/settings")
def save_settings(settings: AlertSettings, user=Depends(current_user)):
    if settings.alerts_enabled and not settings.imessage_to.strip():
        raise HTTPException(400, "Add your phone number or Apple ID email to turn on alerts.")
    db.update_settings(user["id"], settings.imessage_to, settings.alerts_enabled, settings.alert_threshold)
    return {"ok": True}


@app.post("/api/alerts/test")
def test_alert(user=Depends(current_user)):
    if not user["imessage_to"]:
        raise HTTPException(400, "Save your phone number or Apple ID email first.")
    ok, error = alerts.send_imessage(user["imessage_to"], "✅ Test alert from Rachit Stock Prediction. Alerts are working!")
    if not ok:
        raise HTTPException(502, error)
    return {"ok": True}


# ---------- pages ----------

def _page(name: str, request: Request, needs_login: bool = True):
    logged_in = bool(request.session.get("user_id"))
    if needs_login and not logged_in:
        return RedirectResponse("/login")
    if not needs_login and logged_in:
        return RedirectResponse("/")
    return FileResponse(WEB / name)


@app.get("/", include_in_schema=False)
def home_page(request: Request):
    return _page("home.html", request)


@app.get("/company/{ticker}", include_in_schema=False)
def company_page(ticker: str, request: Request):
    return _page("company.html", request)


@app.get("/settings", include_in_schema=False)
def settings_page(request: Request):
    return _page("settings.html", request)


@app.get("/login", include_in_schema=False)
def login_page(request: Request):
    return _page("login.html", request, needs_login=False)


@app.get("/signup", include_in_schema=False)
def signup_page(request: Request):
    return _page("signup.html", request, needs_login=False)
