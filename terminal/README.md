# Global Equity Terminal

A personal financial research terminal for global equity evaluation, written from scratch
in Python and Streamlit. No Fincept Terminal code is used.

**Status: Phase 1 (Foundation) complete.** See [BUILD_LOG.md](BUILD_LOG.md) for how it was built, step by step.

## Run it

```bash
cd terminal
pip install -r requirements.txt
streamlit run Home.py        # opens http://localhost:8501
python -m pytest -q          # 18 tests
```

You can also use GitHub Codespaces: open the repo in a Codespace and the app installs and starts by itself (`.devcontainer/`).
On a Chromebook, enable Linux (Settings → Developers) and run the commands above in the Linux terminal.

## Data sources

| Setting | What happens |
|---|---|
| `auto` (default) | Uses **yfinance** if Yahoo Finance is reachable, else the **demo** source |
| `yfinance` | Free real data: equities, ETFs, indices, FX, crypto worldwide |
| `demo (synthetic)` | Offline, made-up but realistic prices. The sidebar shows a warning whenever it is active |

Symbols follow Yahoo conventions: `AAPL`, `SAP.DE` (Xetra), `ASML.AS` (Amsterdam), `MC.PA` (Paris),
`SHEL.L` (London, priced in pence), `NESN.SW` (Swiss), `^GSPC` (index), `EURUSD=X` (FX), `BTC-USD`.

## Layout

```
terminal/
  Home.py                    home page + market snapshot
  pages/                     one file per page (Streamlit multi-page)
    1_Quotes_and_Charts.py
    2_Watchlists.py
    3_Settings.py
  gterm/
    data/base.py             the DataSource interface every source implements
    data/yfinance_source.py  yfinance adapter
    data/demo_source.py      offline synthetic adapter
    data/registry.py         picks the source and wraps it in the cache
    core/cache.py            SQLite cache (data/cache.sqlite)
    core/config.py           settings.json + .env API keys
    core/fx.py               historical-rate currency conversion
    core/indicators.py       SMA, EMA, RSI, MACD, Bollinger, volatility, drawdown
    core/watchlists.py       watchlists and alerts (data/workspace.sqlite)
    ui.py                    source/as-of captions, sidebar
  tests/                     pytest suite
```

## The models, in plain English

**Data adapter.** Pages never talk to yfinance directly. They ask the registry for prices or a quote.
The registry picks a source that implements the same few methods (`get_prices`, `get_quote`, later
`get_fundamentals`, `get_macro`). Adding EODHD for emerging markets later means writing one new
adapter file, with no changes to any page or model.

**Source and as-of date.** Every result carries the source name, the date of the last price and when it
was fetched. These show under every number. Cached data keeps its *original* fetch time and is labelled "(cached)".

**Cache.** Responses are saved in a local SQLite file: daily prices for 6 hours, quotes for 5 minutes.
This keeps the app fast and avoids hitting Yahoo's rate limits.

**Currency conversion.** To show a euro stock in dollars, each day's price is multiplied by *that
day's* EUR/USD close, not today's rate. If a day has no FX close (for example a holiday), the most recent
earlier rate is used, never a future one. London prices quoted in pence (GBp) are divided by 100 first.

**Indicators.**
- *SMA(n)*: the average of the last n closes. A price above its 200-day SMA is a common sign of an uptrend.
- *EMA(n)*: like the SMA, but recent days count more (weight 2/(n+1)), so it reacts faster.
- *RSI(14)*: compares average up-moves to average down-moves over 14 periods (Wilder's smoothing) on a 0–100 scale. Above 70 is often read as overbought, below 30 as oversold.
- *MACD*: the 12-day EMA minus the 26-day EMA (momentum). The signal line is a 9-day EMA of that, and the histogram is the gap between the two.
- *Bollinger bands*: the 20-day SMA ± 2 standard deviations. The bands widen when the price is volatile.
- *Annualised volatility*: the standard deviation of daily log returns × √252.
- *Max drawdown*: the worst fall from a previous peak during the period.

## Phase plan

1. **Foundation** ✅: shell, data layer, cache, Settings, quotes, charts, watchlists
2. Equity research core: DCF, multiples, quality score, country/currency overlay (yfinance + SEC EDGAR)
3. Portfolio and macro
4. Quant lab
5. Agentic research (needs Anthropic API key)
6. Everything else
