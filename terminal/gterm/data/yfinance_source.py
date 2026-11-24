"""yfinance adapter: free prices and quotes for equities, ETFs, indices, FX and crypto.

Symbol conventions (Yahoo): AAPL, SAP.DE, ASML.AS, VOD.L, ^GSPC, EURUSD=X, BTC-USD.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from ..core.fx import normalise_currency
from ..core.statements import latest, normalise
from .base import PRICE_COLUMNS, DataSource, DataSourceError, Fundamentals, KeyStats, PriceHistory, Quote


class YFinanceSource(DataSource):
    name = "yfinance"
    requires_key = False

    def __init__(self):
        try:
            import yfinance  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise DataSourceError("yfinance is not installed: pip install yfinance") from exc

    def _ticker(self, symbol: str):
        import yfinance as yf

        return yf.Ticker(symbol)

    def get_prices(self, symbol: str, period: str = "1y", interval: str = "1d") -> PriceHistory:
        t = self._ticker(symbol)
        try:
            df = t.history(period=period, interval=interval, auto_adjust=False)
        except Exception as exc:
            raise DataSourceError(f"yfinance could not fetch {symbol}: {exc}") from exc
        if df is None or df.empty:
            raise DataSourceError(f"yfinance returned no prices for {symbol}. Check the symbol.")
        df = df[[c for c in PRICE_COLUMNS if c in df.columns]].copy()
        # Use exchange-local calendar dates as a naive index.
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        df.index = idx.normalize() if interval in ("1d", "1wk", "1mo") else idx
        df.index.name = "Date"
        df = df.dropna(subset=["Close"])
        currency = self._currency(t)
        return PriceHistory(symbol, df, currency, self.name, datetime.now(timezone.utc))

    def _currency(self, t) -> str:
        # Keep Yahoo's case: "GBp" (pence) and "ZAc" (cents) differ from "GBP"/"ZAR" by 100x.
        try:
            ccy = t.fast_info.get("currency") or "USD"
        except Exception:
            return "USD"
        from ..core.fx import MINOR_UNITS

        return ccy if ccy in MINOR_UNITS else ccy.upper()

    def get_quote(self, symbol: str) -> Quote:
        t = self._ticker(symbol)
        hist = self.get_prices(symbol, period="5d", interval="1d")
        frame = hist.frame
        price = float(frame["Close"].iloc[-1])
        prev = float(frame["Close"].iloc[-2]) if len(frame) > 1 else price
        name, exchange, qtype = symbol, "", "EQUITY"
        info = self._info(t)  # .info is flaky; history-based values are a fine fallback
        if info:
            name = " ".join((info.get("longName") or info.get("shortName") or symbol).split())
            exchange = info.get("fullExchangeName") or info.get("exchange") or ""
            qtype = info.get("quoteType") or qtype
            # Prefer the live market price when the market is open (never from a stale profile).
            if not info.get("_stale"):
                if info.get("regularMarketPrice"):
                    price = float(info["regularMarketPrice"])
                if info.get("regularMarketPreviousClose"):
                    prev = float(info["regularMarketPreviousClose"])
        return Quote(
            symbol=symbol,
            name=name,
            price=price,
            previous_close=prev,
            currency=hist.currency,
            exchange=exchange,
            asset_type=qtype,
            price_date=hist.last_date,
            source=self.name,
            as_of=hist.as_of,
        )

    PROFILE_TTL_S = 7 * 86400

    def _info(self, t) -> dict:
        """Yahoo's company-profile endpoint is rate-limited ("Invalid Crumb"/401). Keep the last good
        response for 7 days and fall back to it, marked `_stale`, so names/sectors don't vanish."""
        from ..core import cache

        key = f"yfinfo|{t.ticker}"
        try:
            info = t.info or {}
        except Exception:
            info = {}
        if info.get("quoteType"):
            cache.put(key, info)
            return info
        stale = cache.get(key, self.PROFILE_TTL_S)
        return {**stale, "_stale": True} if stale else {}

    def get_fundamentals(self, symbol: str) -> Fundamentals:
        t = self._ticker(symbol)
        try:
            inc, bal, cf = t.income_stmt, t.balance_sheet, t.cashflow
        except Exception as exc:
            raise DataSourceError(f"yfinance could not fetch statements for {symbol}: {exc}") from exc
        if inc is None or inc.empty:
            raise DataSourceError(f"yfinance has no financial statements for {symbol} (indices, FX and ETFs have none).")
        stmts, notes = normalise(inc, bal, cf)
        info = self._info(t)
        price_ccy = self._currency(t)
        price = info.get("regularMarketPrice") or info.get("currentPrice")
        if not price:
            price = self.get_prices(symbol, "5d").last_close
        shares = info.get("sharesOutstanding") or latest(stmts["balance"], "shares_outstanding") or latest(
            stmts["income"], "diluted_shares"
        )
        return Fundamentals(
            symbol=symbol,
            name=" ".join((info.get("longName") or info.get("shortName") or symbol).split()),
            sector=info.get("sector") or "Unknown",
            industry=info.get("industry") or "Unknown",
            country=info.get("country") or "Unknown",
            price=float(price),
            price_currency=price_ccy,
            financial_currency=info.get("financialCurrency") or normalise_currency(price_ccy)[0],
            shares=float(shares) if shares else float("nan"),
            beta=info.get("beta"),
            statements=stmts,
            adjustments=notes,
            source=self.name,
            as_of=datetime.now(timezone.utc),
        )

    def get_key_stats(self, symbol: str) -> KeyStats:
        info = self._info(self._ticker(symbol))
        if not info or not (info.get("marketCap") or info.get("trailingPE")):
            raise DataSourceError(f"yfinance has no key stats for {symbol}")

        def pos(k):
            v = info.get(k)
            return float(v) if isinstance(v, (int, float)) and v > 0 else None

        return KeyStats(
            symbol=symbol,
            name=" ".join((info.get("longName") or info.get("shortName") or symbol).split()),
            sector=info.get("sector") or "",
            country=info.get("country") or "",
            currency=info.get("currency") or "",
            market_cap=pos("marketCap"),
            pe=pos("trailingPE"),
            ev_ebitda=pos("enterpriseToEbitda"),
            pb=pos("priceToBook"),
            ev_sales=pos("enterpriseToRevenue"),
            enterprise_value=pos("enterpriseValue"),
            source=self.name,
            as_of=datetime.now(timezone.utc),
        )

    def get_options_chain(self, symbol: str, max_expiries: int = 12) -> tuple[pd.DataFrame, float, str]:
        """All calls and puts for the next `max_expiries` expiries. Returns (chain, spot, currency)."""
        t = self._ticker(symbol)
        try:
            expiries = list(t.options)
            if len(expiries) > max_expiries:  # spread the choice across the whole range, not just the next few weeks
                import numpy as np

                idx = sorted(set(np.linspace(0, len(expiries) - 1, max_expiries).round().astype(int)))
                expiries = [expiries[i] for i in idx]
        except Exception as exc:
            raise DataSourceError(f"yfinance has no options for {symbol}: {exc}") from exc
        if not expiries:
            raise DataSourceError(f"No listed options for {symbol} on Yahoo (mostly US-listed names have them).")
        frames = []
        for e in expiries:
            try:
                oc = t.option_chain(e)
            except Exception:
                continue
            for kind, df in (("call", oc.calls), ("put", oc.puts)):
                frames.append(pd.DataFrame({
                    # US equity options stop trading at 16:00 New York time (about 21:00 UTC) on expiry day.
                    "expiry": pd.Timestamp(e) + pd.Timedelta(hours=21), "strike": df["strike"], "type": kind,
                    "bid": df["bid"], "ask": df["ask"], "last": df["lastPrice"], "volume": df["volume"],
                    "open_interest": df["openInterest"], "yahoo_iv": df["impliedVolatility"]}))
        if not frames:
            raise DataSourceError(f"Could not load any option chains for {symbol}")
        spot = self.get_prices(symbol, "5d").last_close
        return pd.concat(frames, ignore_index=True), spot, self._currency(t)

    def search(self, query: str) -> list[dict]:
        try:
            import yfinance as yf

            res = yf.Search(query, max_results=10)
            return [
                {
                    "symbol": q.get("symbol"),
                    "name": q.get("longname") or q.get("shortname") or "",
                    "exchange": q.get("exchDisp") or q.get("exchange") or "",
                    "type": q.get("quoteType") or "",
                }
                for q in res.quotes
                if q.get("symbol")
            ]
        except Exception:
            return []
