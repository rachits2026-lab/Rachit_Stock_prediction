"""Score news and social media with AI models.

Each provider returns the probability the stock falls or grows based on what
people are writing. Gemini and Claude run only when their API keys are set;
a simple built-in word scorer always runs so the app works without keys.
"""

import json
import os
import re

import anthropic
import httpx
from pydantic import BaseModel, Field

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")

PROMPT = """You are a financial news analyst. Read the recent news headlines and social media
posts about {name} ({ticker}) below. Estimate, from this text alone, the probability that the
stock FALLS sharply and the probability that it GROWS strongly over the next two weeks.
Probabilities are between 0 and 1. Give a one or two sentence summary of the main reasons.

News headlines:
{news}

Social media posts:
{posts}
"""


class SentimentScore(BaseModel):
    fall_probability: float = Field(ge=0, le=1)
    grow_probability: float = Field(ge=0, le=1)
    summary: str


def _prompt(ticker: str, name: str, news: list[dict], posts: list[dict]) -> str:
    news_text = "\n".join(f"- {n['title']}" for n in news) or "(none)"
    post_text = "\n".join(f"- [{p['sentiment'] or 'untagged'}] {p['text']}" for p in posts) or "(none)"
    return PROMPT.format(name=name, ticker=ticker, news=news_text, posts=post_text)


def score_with_gemini(prompt: str) -> SentimentScore | None:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    body = {
        "contents": [{"parts": [{"text": prompt + "\nRespond with JSON with keys fall_probability, grow_probability, summary."}]}],
        "generationConfig": {"responseMimeType": "application/json"},
    }
    try:
        resp = httpx.post(url, json=body, headers={"x-goog-api-key": key}, timeout=30)
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        return SentimentScore.model_validate(json.loads(text))
    except (httpx.HTTPError, KeyError, IndexError, ValueError):
        return None


def score_with_claude(prompt: str) -> SentimentScore | None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        response = anthropic.Anthropic().messages.parse(
            model=CLAUDE_MODEL,
            max_tokens=2000,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": prompt}],
            output_format=SentimentScore,
        )
    except anthropic.APIError:
        return None
    if response.stop_reason == "refusal":
        return None
    return response.parsed_output


NEGATIVE = {
    "crash", "plunge", "plunges", "fall", "falls", "drop", "drops", "slump", "sell", "selloff",
    "lawsuit", "probe", "fraud", "miss", "misses", "downgrade", "downgraded", "cut", "cuts",
    "layoffs", "bankruptcy", "recall", "weak", "loss", "losses", "decline", "bearish", "short",
    "warning", "investigation", "tumble", "tumbles", "sink", "sinks", "fears", "risk", "puts",
}
POSITIVE = {
    "surge", "surges", "soar", "soars", "jump", "jumps", "rally", "beat", "beats", "upgrade",
    "upgraded", "record", "growth", "strong", "buy", "bullish", "gain", "gains", "profit",
    "rise", "rises", "outperform", "raises", "boost", "partnership", "approval", "calls", "moon",
}


def score_locally(news: list[dict], posts: list[dict]) -> SentimentScore | None:
    """Count positive/negative words, plus StockTwits users' own Bullish/Bearish tags."""
    pos = neg = 0
    for text in [n["title"] for n in news] + [p["text"] for p in posts]:
        words = set(re.findall(r"[a-z]+", text.lower()))
        pos += len(words & POSITIVE)
        neg += len(words & NEGATIVE)
    pos += 2 * sum(p["sentiment"] == "Bullish" for p in posts)
    neg += 2 * sum(p["sentiment"] == "Bearish" for p in posts)
    if pos + neg == 0:
        return None
    # Shrink toward 50/50 when there is little evidence.
    fall = (neg + 2) / (pos + neg + 4)
    return SentimentScore(
        fall_probability=round(fall, 3),
        grow_probability=round(1 - fall, 3),
        summary=f"Found {pos} positive and {neg} negative signals in {len(news)} headlines and {len(posts)} posts.",
    )


def analyze_sentiment(ticker: str, name: str, news: list[dict], posts: list[dict]) -> list[dict]:
    """Run every available provider. Returns a list of {provider, fall_probability, grow_probability, summary}."""
    results = []
    if news or posts:
        prompt = _prompt(ticker, name, news, posts)
        for provider, fn in (("Gemini", lambda: score_with_gemini(prompt)), ("Claude", lambda: score_with_claude(prompt)), ("Built-in word scorer", lambda: score_locally(news, posts))):
            score = fn()
            if score:
                results.append({"provider": provider, **score.model_dump()})
    return results
