"""SQLite storage for users, their settings and watchlists."""

import hashlib
import hmac
import os
import secrets
import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(os.getenv("APP_DB", Path(__file__).resolve().parent.parent / "data" / "app.db"))

# Demo account created on first start so the app can be tried right away.
DEMO_EMAIL = "example@gmail.com"
DEMO_PASSWORD = "123"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    imessage_to TEXT NOT NULL DEFAULT '',
    alerts_enabled INTEGER NOT NULL DEFAULT 0,
    alert_threshold REAL NOT NULL DEFAULT 0.6
);
CREATE TABLE IF NOT EXISTS watchlist (
    user_id INTEGER NOT NULL REFERENCES users(id),
    ticker TEXT NOT NULL,
    last_alert_date TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (user_id, ticker)
);
"""


@contextmanager
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000).hex()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)
    if not get_user_by_email(DEMO_EMAIL):
        create_user(DEMO_EMAIL, DEMO_PASSWORD)


def create_user(email: str, password: str) -> int:
    salt = secrets.token_hex(16)
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO users (email, password_hash, salt) VALUES (?, ?, ?)",
            (email.strip().lower(), _hash(password, salt), salt),
        )
        return cur.lastrowid


def get_user_by_email(email: str) -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()


def get_user(user_id: int) -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def verify_password(email: str, password: str) -> sqlite3.Row | None:
    user = get_user_by_email(email)
    if user and hmac.compare_digest(user["password_hash"], _hash(password, user["salt"])):
        return user
    return None


def update_settings(user_id: int, imessage_to: str, alerts_enabled: bool, alert_threshold: float) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE users SET imessage_to = ?, alerts_enabled = ?, alert_threshold = ? WHERE id = ?",
            (imessage_to.strip(), int(alerts_enabled), alert_threshold, user_id),
        )


def watchlist(user_id: int) -> list[str]:
    with connect() as conn:
        rows = conn.execute("SELECT ticker FROM watchlist WHERE user_id = ? ORDER BY ticker", (user_id,))
        return [r["ticker"] for r in rows]


def add_to_watchlist(user_id: int, ticker: str) -> None:
    with connect() as conn:
        conn.execute("INSERT OR IGNORE INTO watchlist (user_id, ticker) VALUES (?, ?)", (user_id, ticker))


def remove_from_watchlist(user_id: int, ticker: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM watchlist WHERE user_id = ? AND ticker = ?", (user_id, ticker))


def alert_targets() -> list[sqlite3.Row]:
    """Watchlist entries whose owner has iMessage alerts switched on."""
    with connect() as conn:
        return conn.execute(
            """SELECT u.id AS user_id, u.imessage_to, u.alert_threshold, w.ticker, w.last_alert_date
               FROM watchlist w JOIN users u ON u.id = w.user_id
               WHERE u.alerts_enabled = 1 AND u.imessage_to != ''"""
        ).fetchall()


def mark_alerted(user_id: int, ticker: str, date: str) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE watchlist SET last_alert_date = ? WHERE user_id = ? AND ticker = ?", (date, user_id, ticker)
        )
