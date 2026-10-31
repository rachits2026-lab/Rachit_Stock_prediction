"""Company catalog: a bundled list of well-known companies plus live Yahoo Finance search,
so any publicly traded company in the world can be looked up."""

import csv
from functools import lru_cache
from pathlib import Path

import yfinance as yf

CATALOG = Path(__file__).resolve().parent.parent / "data" / "companies.csv"
SEARCHABLE_TYPES = {"EQUITY", "ETF"}


@lru_cache
def featured() -> list[dict]:
    with CATALOG.open() as f:
        return list(csv.DictReader(f))


def search(query: str, limit: int = 15) -> list[dict]:
    q = query.strip().lower()
    if not q:
        return featured()[:limit]
    results = [c for c in featured() if q in c["ticker"].lower() or q in c["name"].lower()]
    seen = {c["ticker"] for c in results}
    try:
        quotes = yf.Search(query, max_results=limit).quotes
    except Exception:  # yfinance raises many network/parse error types
        quotes = []
    for quote in quotes:
        symbol = quote.get("symbol")
        if symbol and symbol not in seen and quote.get("quoteType") in SEARCHABLE_TYPES:
            seen.add(symbol)
            results.append(
                {
                    "ticker": symbol,
                    "name": quote.get("longname") or quote.get("shortname") or symbol,
                    "sector": quote.get("sectorDisp") or quote.get("quoteType", "").title(),
                    "country": quote.get("exchDisp") or quote.get("exchange", ""),
                }
            )
    return results[:limit]
