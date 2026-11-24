"""News and geopolitical events.

- **Company news**: yfinance (Yahoo Finance's own news list for a ticker).
- **Any topic, company or country**: Google News RSS search (free, no key).
- **Geopolitical events**: GDELT 2.0 DOC API (free, no key). GDELT monitors news in 100+ languages
  worldwide; we use article lists and the daily volume/tone timeline for a query.
"""
from __future__ import annotations

import json
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import pandas as pd

from .base import DataSourceError
from .macro import _get

NEWS_TTL_S = 15 * 60  # headlines go stale fast


@dataclass
class Article:
    title: str
    url: str
    published: datetime | None
    publisher: str
    source: str  # which feed found it
    country: str = ""


def _clean(t: str) -> str:
    return " ".join((t or "").split())


# ---------------------------------------------------------------- Google News RSS

def parse_rss(xml_bytes: bytes, source: str) -> list[Article]:
    root = ET.fromstring(xml_bytes)
    out = []
    for item in root.iter("item"):
        title = _clean(item.findtext("title"))
        pub_el = item.find("source")
        publisher = _clean(pub_el.text) if pub_el is not None and pub_el.text else ""
        # Google appends " - Publisher" to titles; strip it when we know the publisher.
        if publisher and title.endswith(f" - {publisher}"):
            title = title[: -len(publisher) - 3]
        try:
            published = parsedate_to_datetime(item.findtext("pubDate")) if item.findtext("pubDate") else None
        except (TypeError, ValueError):
            published = None
        out.append(Article(title, _clean(item.findtext("link")), published, publisher, source))
    return out


def google_news(query: str, lang: str = "en", region: str = "US") -> list[Article]:
    q = urllib.parse.urlencode({"q": query, "hl": f"{lang}-{region}", "gl": region, "ceid": f"{region}:{lang}"})
    return parse_rss(_get(f"https://news.google.com/rss/search?{q}", NEWS_TTL_S), "Google News")


# ---------------------------------------------------------------- yfinance company news

def parse_yf_news(items: list) -> list[Article]:
    out = []
    for it in items or []:
        c = it.get("content", it)  # newer yfinance nests under "content"
        title = c.get("title")
        if not title:
            continue
        url = (c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url") or c.get("link") or c.get("url", "")
        when = c.get("pubDate") or c.get("displayTime") or c.get("providerPublishTime")
        if isinstance(when, (int, float)):
            published = datetime.fromtimestamp(when, timezone.utc)
        else:
            published = pd.to_datetime(when, utc=True, errors="coerce")
            published = None if pd.isna(published) else published.to_pydatetime()
        publisher = (c.get("provider") or {}).get("displayName") or c.get("publisher", "")
        out.append(Article(_clean(title), url, published, publisher, "Yahoo Finance"))
    return out


def company_news(symbol: str) -> list[Article]:
    import yfinance as yf

    try:
        items = yf.Ticker(symbol).news
    except Exception as exc:
        raise DataSourceError(f"Yahoo Finance news unavailable for {symbol}: {exc}") from exc
    return parse_yf_news(items)


# ---------------------------------------------------------------- GDELT

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def parse_gdelt_articles(payload: dict) -> list[Article]:
    out = []
    for a in payload.get("articles", []):
        try:
            when = datetime.strptime(a.get("seendate", ""), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            when = None
        out.append(Article(_clean(a.get("title")), a.get("url", ""), when, a.get("domain", ""), "GDELT", a.get("sourcecountry", "")))
    return out


def gdelt_articles(query: str, timespan: str = "3d", max_records: int = 150) -> list[Article]:
    q = urllib.parse.urlencode({"query": query, "mode": "artlist", "format": "json", "maxrecords": max_records,
                                "timespan": timespan, "sort": "datedesc"})
    raw = _get(f"{GDELT_URL}?{q}", NEWS_TTL_S)
    try:
        return parse_gdelt_articles(json.loads(raw))
    except json.JSONDecodeError as exc:  # GDELT answers bad queries with plain text
        raise DataSourceError(f"GDELT: {raw[:200].decode(errors='replace')}") from exc


def parse_gdelt_timeline(payload: dict) -> pd.DataFrame:
    frames = {}
    for series in payload.get("timeline", []):
        name = series.get("series", "value")
        pts = {pd.to_datetime(p["date"], format="%Y%m%dT%H%M%SZ"): p["value"] for p in series.get("data", [])}
        frames[name] = pd.Series(pts, dtype=float)
    return pd.DataFrame(frames).sort_index()


def gdelt_timeline(query: str, mode: str = "timelinetone", timespan: str = "3m") -> pd.DataFrame:
    """mode: 'timelinetone' (average tone, -ve = negative coverage) or 'timelinevol' (share of all coverage)."""
    q = urllib.parse.urlencode({"query": query, "mode": mode, "format": "json", "timespan": timespan})
    raw = _get(f"{GDELT_URL}?{q}", NEWS_TTL_S)
    try:
        return parse_gdelt_timeline(json.loads(raw))
    except json.JSONDecodeError as exc:
        raise DataSourceError(f"GDELT: {raw[:200].decode(errors='replace')}") from exc


def plain_query(gdelt_query: str) -> str:
    """Turn GDELT syntax into a plain search: drop filters like sourcecountry:x and brackets/quotes."""
    import re

    q = re.sub(r"\b\w+:\S+", " ", gdelt_query)
    return " ".join(re.sub(r"[()\"]", " ", q).split())


def to_frame(articles: list[Article]) -> pd.DataFrame:
    df = pd.DataFrame([a.__dict__ for a in articles])
    if df.empty:
        return df
    df["published"] = pd.to_datetime(df["published"], utc=True, errors="coerce")
    return df.drop_duplicates("title").sort_values("published", ascending=False)
