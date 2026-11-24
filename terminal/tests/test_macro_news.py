"""Parsers for every macro/news source, tested on sample responses (no network)."""
import json

import pandas as pd
import pytest

from gterm.data import macro, news
from gterm.data.base import DataSourceError


def test_fred_csv_handles_missing_dots_and_both_header_styles():
    for header in ("DATE", "observation_date"):
        s = macro.parse_fred_csv(f"{header},DGS10\n2026-11-20,4.10\n2026-11-21,.\n2026-11-24,4.15\n")
        assert list(s.values) == [4.10, 4.15]
        assert s.index[-1] == pd.Timestamp("2026-11-24")


def test_fred_json():
    s = macro.parse_fred_json({"observations": [{"date": "2026-10-01", "value": "3.5"}, {"date": "2026-11-01", "value": "."}]})
    assert list(s.values) == [3.5]


WB = [{"page": 1, "pages": 1, "total": 3},
      [{"indicator": {"id": "NY.GDP.MKTP.KD.ZG", "value": "GDP growth (annual %)"}, "date": "2024", "value": 0.6},
       {"indicator": {"id": "NY.GDP.MKTP.KD.ZG", "value": "GDP growth (annual %)"}, "date": "2023", "value": 0.7},
       {"indicator": {"id": "NY.GDP.MKTP.KD.ZG", "value": "GDP growth (annual %)"}, "date": "2025", "value": None}]]


def test_worldbank_parse_sorts_and_drops_nulls():
    s, title = macro.parse_worldbank(WB)
    assert title == "GDP growth (annual %)"
    assert list(s.index.year) == [2023, 2024] and list(s.values) == [0.7, 0.6]


def test_worldbank_error_message():
    with pytest.raises(DataSourceError, match="Invalid value"):
        macro.parse_worldbank([{"message": [{"id": "120", "key": "Invalid value", "value": "Invalid value"}]}])


def test_imf_parse():
    payload = {"values": {"NGDP_RPCH": {"ZAF": {"2024": 0.6, "2025": 1.1, "2030": 1.8}}}}
    s = macro.parse_imf(payload, "NGDP_RPCH", "ZAF")
    assert s.loc["2030-12-31"] == 1.8
    with pytest.raises(DataSourceError):
        macro.parse_imf(payload, "NGDP_RPCH", "USA")


def test_ecb_csv_daily_and_monthly():
    d = macro.parse_ecb_csv("KEY,FREQ,TIME_PERIOD,OBS_VALUE\nX,B,2026-06-11,2.0\nX,B,2026-06-12,2.25\n")
    assert d.iloc[-1] == 2.25 and d.index[-1] == pd.Timestamp("2026-06-12")
    m = macro.parse_ecb_csv("KEY,FREQ,TIME_PERIOD,OBS_VALUE\nX,M,2026-09,2.1\nX,M,2026-10,2.0\n")
    assert m.index[-1] == pd.Timestamp("2026-10-31")


def test_sarb_parse_is_case_insensitive():
    df = macro.parse_sarb_rates([{"Name": "Repo rate", "Value": "7.00", "Date": "2026-11-20"},
                                 {"name": "Prime lending rate", "value": 10.5, "date": "2026-11-20"},
                                 {"SomethingElse": 1}])
    assert list(df["Indicator"]) == ["Repo rate", "Prime lending rate"]
    assert df["Value"].tolist() == [7.0, 10.5]


def test_yield_curve_uses_latest_point_per_tenor(monkeypatch):
    def fake_fred(sid, *a, **k):
        idx = pd.to_datetime(["2025-11-24", "2026-11-24"])
        val = {"DGS2": [4.0, 3.5], "DGS10": [4.2, 4.1]}.get(sid, [4.0, 4.0])
        return macro.MacroSeries(sid, sid, "%", pd.Series(val, index=idx), "FRED", "u", macro._now())

    monkeypatch.setattr(macro, "fred", fake_fred)
    curve, src, d = macro.yield_curve("US Treasury")
    assert curve[2.0] == 3.5 and curve[10.0] == 4.1 and d == pd.Timestamp("2026-11-24")
    old, _, d_old = macro.yield_curve("US Treasury", pd.Timestamp("2026-01-01"))
    assert old[2.0] == 4.0 and d_old == pd.Timestamp("2025-11-24")


RSS = b"""<?xml version="1.0"?><rss><channel>
<item><title>SAP beats forecasts - Reuters</title><link>https://example.com/a</link>
<pubDate>Mon, 24 Nov 2026 08:00:00 GMT</pubDate><source url="https://reuters.com">Reuters</source></item>
<item><title>No date story</title><link>https://example.com/b</link></item>
</channel></rss>"""


def test_rss_parse_strips_publisher_suffix():
    arts = news.parse_rss(RSS, "Google News")
    assert arts[0].title == "SAP beats forecasts" and arts[0].publisher == "Reuters"
    assert arts[0].published.year == 2026 and arts[1].published is None


def test_yf_news_new_and_old_formats():
    new = [{"id": "1", "content": {"title": "A", "pubDate": "2026-11-24T08:00:00Z", "provider": {"displayName": "Yahoo"},
                                   "canonicalUrl": {"url": "https://y/a"}}}]
    old = [{"title": "B", "link": "https://y/b", "providerPublishTime": 1795000000, "publisher": "Bloomberg"}]
    a, b = news.parse_yf_news(new)[0], news.parse_yf_news(old)[0]
    assert (a.title, a.url, a.publisher) == ("A", "https://y/a", "Yahoo")
    assert (b.title, b.url, b.publisher) == ("B", "https://y/b", "Bloomberg") and b.published is not None


def test_gdelt_articles_and_timeline():
    arts = news.parse_gdelt_articles({"articles": [{"url": "u", "title": " Talks  stall ", "seendate": "20261124T081500Z",
                                                    "domain": "bbc.co.uk", "sourcecountry": "United Kingdom"}]})
    assert arts[0].title == "Talks stall" and arts[0].country == "United Kingdom" and arts[0].published.hour == 8
    tl = news.parse_gdelt_timeline({"timeline": [{"series": "Average Tone", "data": [
        {"date": "20261122T000000Z", "value": -1.5}, {"date": "20261123T000000Z", "value": 0.5}]}]})
    assert list(tl["Average Tone"]) == [-1.5, 0.5]


def test_to_frame_dedupes_and_sorts():
    arts = news.parse_rss(RSS, "Google News") + news.parse_rss(RSS, "Google News")
    df = news.to_frame(arts)
    assert len(df) == 2 and df.iloc[0]["title"] == "SAP beats forecasts"
