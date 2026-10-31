"""Collect recent news headlines and social media posts about a stock."""

import html
import ssl
import xml.etree.ElementTree as ET

import httpx
import truststore

HEADERS = {"User-Agent": "Mozilla/5.0 (RachitStockPrediction)"}
TIMEOUT = 8.0
# Use the operating system's certificate store; some feeds fail with Python's bundled one.
SSL_CONTEXT = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


def fetch_news(ticker: str, limit: int = 15) -> list[dict]:
    """Latest headlines from the Yahoo Finance RSS feed. Returns [] on failure."""
    url = "https://feeds.finance.yahoo.com/rss/2.0/headline"
    try:
        resp = httpx.get(url, params={"s": ticker, "region": "US", "lang": "en-US"}, headers=HEADERS, timeout=TIMEOUT, verify=SSL_CONTEXT)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except (httpx.HTTPError, ET.ParseError):
        return []
    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        if title:
            items.append(
                {
                    "title": html.unescape(title),
                    "link": item.findtext("link") or "",
                    "published": item.findtext("pubDate") or "",
                    "source": "Yahoo Finance",
                }
            )
    return items[:limit]


def fetch_social(ticker: str, limit: int = 20) -> list[dict]:
    """Latest StockTwits posts, with the poster's own Bullish/Bearish tag if given."""
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
    try:
        resp = httpx.get(url, headers=HEADERS, timeout=TIMEOUT, verify=SSL_CONTEXT)
        resp.raise_for_status()
        messages = resp.json().get("messages", [])
    except (httpx.HTTPError, ValueError):
        return []
    posts = []
    for m in messages[:limit]:
        sentiment = ((m.get("entities") or {}).get("sentiment") or {}).get("basic")
        posts.append(
            {
                "text": html.unescape(m.get("body", ""))[:280],
                "sentiment": sentiment,  # "Bullish", "Bearish" or None
                "published": m.get("created_at", ""),
                "source": "StockTwits",
            }
        )
    return posts
