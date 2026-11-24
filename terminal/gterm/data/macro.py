"""Macro data adapters: FRED, World Bank, IMF, ECB and SARB. All free.

Each source has a small `parse_*` function (pure, tested on sample responses) and a
fetch function that downloads, parses and caches. Every result is a `MacroSeries`
carrying its source, the exact URL and the retrieval time.

| Source | What | Key? |
|---|---|---|
| FRED (St Louis Fed) | US series: rates, yields, inflation, plus some OECD series for other countries | optional (keyless CSV used otherwise) |
| World Bank | annual GDP growth, inflation, unemployment, trade for every country | no |
| IMF DataMapper | World Economic Outlook series, including forecasts | no |
| ECB Data Portal | euro-area policy rates, HICP inflation, euro yield curve | no |
| SARB | South African policy and market rates (current values) | no |
"""
from __future__ import annotations

import io
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from ..core import cache, config
from .base import DataSourceError

TTL_S = 12 * 3600
USER_AGENT = "GlobalEquityTerminal/1.0 (personal research tool)"


@dataclass
class MacroSeries:
    id: str
    title: str
    units: str
    series: pd.Series  # DatetimeIndex -> float
    source: str
    url: str
    as_of: datetime
    forecast_from: pd.Timestamp | None = None  # first date that is a forecast (IMF), if any

    @property
    def last(self) -> tuple[pd.Timestamp, float]:
        s = self.series.dropna()
        return s.index[-1], float(s.iloc[-1])


def _get(url: str) -> bytes:
    key = f"macro|{url}"
    if (hit := cache.get(key, TTL_S)) is not None:
        return hit
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            body = r.read()
    except Exception as exc:
        host = urllib.parse.urlparse(url).hostname
        hint = " The network blocked this host: add it to the allowed domains." if "403" in str(exc) or "Tunnel" in str(exc) else ""
        raise DataSourceError(f"{host} request failed: {exc}.{hint}") from exc
    cache.put(key, body)
    return body


def _now():
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------- FRED

def parse_fred_csv(text: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    date_col = df.columns[0]  # "DATE" or "observation_date"
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")  # FRED uses "." for missing
    s.index = pd.to_datetime(df[date_col])
    return s.dropna()


def parse_fred_json(payload: dict) -> pd.Series:
    obs = payload.get("observations", [])
    s = pd.Series({pd.Timestamp(o["date"]): o["value"] for o in obs})
    return pd.to_numeric(s, errors="coerce").dropna()


def fred(series_id: str, title: str = "", units: str = "") -> MacroSeries:
    key = config.get_key("FRED_API_KEY")
    if key:
        url = ("https://api.stlouisfed.org/fred/series/observations?"
               + urllib.parse.urlencode({"series_id": series_id, "api_key": key, "file_type": "json"}))
        s = parse_fred_json(json.loads(_get(url)))
        shown_url = url.replace(key, "***")
    else:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={urllib.parse.quote(series_id)}"
        s = parse_fred_csv(_get(url).decode())
        shown_url = url
    if s.empty:
        raise DataSourceError(f"FRED returned no data for {series_id}")
    return MacroSeries(series_id, title or series_id, units, s, "FRED", shown_url, _now())


# ---------------------------------------------------------------- World Bank

def parse_worldbank(payload: list) -> tuple[pd.Series, str]:
    if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
        msg = payload[0].get("message", [{}])[0].get("value", "no data") if isinstance(payload, list) and payload else "no data"
        raise DataSourceError(f"World Bank: {msg}")
    rows = payload[1]
    s = pd.Series({pd.Timestamp(f"{r['date']}-12-31"): r["value"] for r in rows if r.get("value") is not None}, dtype=float)
    return s.sort_index(), rows[0]["indicator"]["value"]


def worldbank(country_iso3: str, indicator: str) -> MacroSeries:
    url = f"https://api.worldbank.org/v2/country/{country_iso3}/indicator/{indicator}?format=json&per_page=200"
    s, title = parse_worldbank(json.loads(_get(url)))
    return MacroSeries(indicator, title, "", s, "World Bank", url, _now())


# ---------------------------------------------------------------- IMF DataMapper (WEO, includes forecasts)

def parse_imf(payload: dict, indicator: str, country_iso3: str) -> pd.Series:
    vals = payload.get("values", {}).get(indicator, {}).get(country_iso3, {})
    if not vals:
        raise DataSourceError(f"IMF has no {indicator} for {country_iso3}")
    return pd.Series({pd.Timestamp(f"{y}-12-31"): float(v) for y, v in vals.items() if v is not None}).sort_index()


def imf(indicator: str, country_iso3: str, title: str = "", units: str = "") -> MacroSeries:
    url = f"https://www.imf.org/external/datamapper/api/v1/{indicator}/{country_iso3}"
    s = parse_imf(json.loads(_get(url)), indicator, country_iso3)
    this_year = pd.Timestamp(f"{_now().year - 1}-12-31")  # WEO: last full year is mostly actuals; later years forecast
    return MacroSeries(indicator, title or indicator, units, s, "IMF World Economic Outlook", url, _now(),
                       forecast_from=this_year + pd.offsets.YearEnd(1))


# ---------------------------------------------------------------- ECB Data Portal

def _ecb_date(p: str) -> pd.Timestamp:
    """ECB periods: '2024-06-12' (daily), '2024-05' (monthly), '2024-Q1' (quarterly), '2024' (annual).
    Periods are dated at their last day."""
    if len(p) == 10:
        return pd.Timestamp(p)
    if "Q" in p:
        return pd.Period(p.replace("-", ""), "Q").end_time.normalize()
    if len(p) == 7:
        return pd.Period(p, "M").end_time.normalize()
    return pd.Period(p, "Y").end_time.normalize()


def parse_ecb_csv(text: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    if "TIME_PERIOD" not in df.columns:
        raise DataSourceError("ECB: unexpected response format")
    idx = pd.DatetimeIndex([_ecb_date(p) for p in df["TIME_PERIOD"].astype(str)])
    return pd.Series(pd.to_numeric(df["OBS_VALUE"], errors="coerce").values, index=idx).dropna().sort_index()


def ecb(flow_key: str, title: str = "", units: str = "", last_n: int | None = None) -> MacroSeries:
    flow, key = flow_key.split(".", 1)
    q = "format=csvdata" + (f"&lastNObservations={last_n}" if last_n else "")
    url = f"https://data-api.ecb.europa.eu/service/data/{flow}/{key}?{q}"
    return MacroSeries(flow_key, title or flow_key, units, parse_ecb_csv(_get(url).decode()), "ECB Data Portal", url, _now())


# ---------------------------------------------------------------- SARB (current rates)

def parse_sarb_rates(payload) -> pd.DataFrame:
    """SARB's web-indicator endpoint returns a list of rate records. Field names have varied, so
    match them case-insensitively."""
    if not isinstance(payload, list):
        raise DataSourceError("SARB: unexpected response format")
    rows = []
    for r in payload:
        low = {k.lower(): v for k, v in r.items()}
        name = low.get("name") or low.get("description") or low.get("title")
        value = low.get("value")
        date = low.get("date") or low.get("period")
        if name is None or value is None:
            continue
        rows.append({"Indicator": str(name).strip(), "Value": pd.to_numeric(value, errors="coerce"),
                     "Date": pd.to_datetime(date, errors="coerce")})
    if not rows:
        raise DataSourceError("SARB: no rates in response")
    return pd.DataFrame(rows)


def sarb_rates() -> tuple[pd.DataFrame, str]:
    url = "https://custom.resbank.co.za/SarbWebApi/WebIndicators/HomePageRates"
    return parse_sarb_rates(json.loads(_get(url))), url


# ---------------------------------------------------------------- catalogues

COUNTRIES = {  # name -> ISO3
    "United States": "USA", "Euro area": "EMU", "Germany": "DEU", "France": "FRA", "Italy": "ITA", "Spain": "ESP",
    "Netherlands": "NLD", "United Kingdom": "GBR", "Switzerland": "CHE", "Sweden": "SWE", "Japan": "JPN",
    "China": "CHN", "India": "IND", "Brazil": "BRA", "South Africa": "ZAF",
}
IMF_EURO_AREA = "EURO"  # the IMF code for the euro area differs from the World Bank's

INDICATORS = {  # label -> (World Bank code, IMF WEO code, units)
    "Real GDP growth": ("NY.GDP.MKTP.KD.ZG", "NGDP_RPCH", "% a year"),
    "Inflation (CPI)": ("FP.CPI.TOTL.ZG", "PCPIPCH", "% a year"),
    "Unemployment": ("SL.UEM.TOTL.ZS", "LUR", "% of labour force"),
    "Current account balance": ("BN.CAB.XOKA.GD.ZS", "BCA_NGDPD", "% of GDP"),
    "Trade (exports + imports)": ("NE.TRD.GNFS.ZS", None, "% of GDP"),
    "Government debt": ("GC.DOD.TOTL.GD.ZS", "GGXWDG_NGDP", "% of GDP"),
}

POLICY_RATES = {  # label -> (fetcher, args)
    "US Fed funds (effective)": ("fred", "DFF"),
    "ECB deposit facility": ("ecb", "FM.B.U2.EUR.4F.KR.DFR.LEV"),
    "ECB main refinancing": ("ecb", "FM.B.U2.EUR.4F.KR.MRR_FR.LEV"),
    "UK short-term rate (OECD via FRED)": ("fred", "IRSTCI01GBM156N"),
    "South Africa (short-term rate, OECD via FRED)": ("fred", "IR3TIB01ZAM156N"),
}

US_CURVE = {"1M": "DGS1MO", "3M": "DGS3MO", "6M": "DGS6MO", "1Y": "DGS1", "2Y": "DGS2", "3Y": "DGS3", "5Y": "DGS5",
            "7Y": "DGS7", "10Y": "DGS10", "20Y": "DGS20", "30Y": "DGS30"}
EURO_CURVE = {m: f"YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_{m}" for m in ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]}
TENOR_YEARS = {"1M": 1 / 12, "3M": 0.25, "6M": 0.5, "1Y": 1, "2Y": 2, "3Y": 3, "5Y": 5, "7Y": 7, "10Y": 10, "20Y": 20, "30Y": 30}


def policy_rate(label: str) -> MacroSeries:
    kind, sid = POLICY_RATES[label]
    return fred(sid, label, "%") if kind == "fred" else ecb(sid, label, "%")


def country_indicator(label: str, country: str, source: str = "World Bank") -> MacroSeries:
    wb_code, imf_code, units = INDICATORS[label]
    iso = COUNTRIES[country]
    if source == "IMF" and imf_code:
        ms = imf(imf_code, IMF_EURO_AREA if iso == "EMU" else iso, label, units)
    else:
        ms = worldbank(iso, wb_code)
        ms.units = units
    ms.title = f"{label}: {country}"
    return ms


def yield_curve(which: str, on: pd.Timestamp | None = None) -> tuple[pd.Series, str, pd.Timestamp]:
    """Curve (tenor in years -> yield %) on the latest date, or on/just before `on`."""
    tenors = US_CURVE if which == "US Treasury" else EURO_CURVE
    pts, src, dates = {}, "", []
    for tenor, sid in tenors.items():
        ms = fred(sid) if which == "US Treasury" else ecb(sid)
        s = ms.series if on is None else ms.series[ms.series.index <= on]
        if s.empty:
            continue
        pts[TENOR_YEARS[tenor]] = float(s.iloc[-1])
        dates.append(s.index[-1])
        src = ms.source
    if not pts:
        raise DataSourceError(f"No {which} curve data")
    return pd.Series(pts).sort_index(), src, max(dates)
