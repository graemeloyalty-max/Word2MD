# Global Equity Terminal

A personal financial research terminal for global equity evaluation, written from scratch
in Python and Streamlit. No Fincept Terminal code is used.

**Status: Phase 4 (Quant lab) complete.** See BUILD_LOG.md for how it was built. See [BUILD_LOG.md](BUILD_LOG.md) for how it was built, step by step.

## Run it

```bash
cd terminal
pip install -r requirements.txt
streamlit run Home.py        # opens http://localhost:8501
python -m pytest -q          # 104 tests
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
  Home.py                    customisable dashboard
  pages/                     one file per page (Streamlit multi-page)
    1_Quotes_and_Charts.py
    2_Watchlists.py
    3_Settings.py
    4_Equity_Research.py
    5_Portfolio.py
    6_Macro.py
    7_News_and_Events.py
    8_Quant_Lab.py
    9_Volatility_Surface.py
    10_AI_Research.py
  gterm/
    data/base.py             the DataSource interface every source implements
    data/yfinance_source.py  yfinance adapter
    data/demo_source.py      offline synthetic adapter
    data/registry.py         picks the source and wraps it in the cache
    data/edgar_source.py     SEC EDGAR 10-K history for US filers
    data/macro.py            FRED, World Bank, IMF, ECB, SARB
    data/news.py             Yahoo news, Google News RSS, GDELT
    analysis/portfolio.py    returns, risk, correlation, attribution
    analysis/options.py      Black-Scholes, Greeks, implied vol, vol surface
    data/deribit.py          crypto option chains
    quant/backtest.py        backtesting engine and parameter sweeps
    quant/strategies.py      buy & hold, equal weight, momentum, trend, inverse vol, factor top group
    quant/factors.py         factor scores, point-in-time fundamentals, IC, quantile returns
    quant/metrics.py         CAGR, Sharpe, Sortino, drawdown, Calmar
    quant/universe.py        loads a universe in one currency
    ai/providers.py          Gemini (Google AI Studio) and Groq adapters
    ai/tools.py              the terminal's functions as AI tools (cited)
    ai/agent.py              ask → run tools → answer loop
    core/portfolio_loader.py prices + FX + profiles for a portfolio
    analysis/dcf.py          DCF, sensitivity grid, reverse DCF
    analysis/multiples.py    peer groups and implied prices
    analysis/quality.py      quality score
    analysis/country.py      country risk premiums, risk-free defaults
    core/statements.py       normalises statements across sources and accounting standards
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

## Phase 2 models, in plain English

**Statement normalisation** (`core/statements.py`). Every company's statements are mapped onto the same
standard lines. Where IFRS lets a company classify something differently from US GAAP, it is moved so both look alike:
interest paid or received is put in operating cash flow; capitalised development costs (purchases of intangibles) count as capex;
lease liabilities are part of debt. The adjustments made for each company are listed on its Statements tab.

**DCF** (`analysis/dcf.py`). Start from the latest free cash flow (operating cash flow − capex). Grow it at your
chosen rate for 5 years, then fade the growth in a straight line down to the terminal rate. After the projection, assume
it grows at the terminal rate for ever (Gordon growth). Discount all of it at the WACC. Then subtract debt and minorities,
add cash, and divide by shares.
- Cost of equity = risk-free + beta × equity risk premium + λ × country risk premium.
- WACC weights equity at market value and debt at book value, and gives debt its tax shield.
- The **sensitivity grid** shows the value for WACC ±1% and terminal growth ±1%.
- The **reverse DCF** finds the growth rate the market price implies: if that number looks unrealistic, so does the price.
- Not used for banks and insurers, whose debt is an operating input, not financing.

**Peer multiples** (`analysis/multiples.py`). Compare P/E, EV/EBITDA, P/B and EV/Sales with a default group of
same-sector peers in the same region (US or Europe); edit the group freely. Each multiple implies a share price at the
peer *median*. Medians ignore one-off extremes. EV-based prices hold net debt constant. The headline figure is the
median of the four implied prices.

**Quality score** (`analysis/quality.py`). Four measures, each scored 0–100 on a straight line, then averaged:
ROIC (0% → 0, 20% → 100), margin variation (50% → 0, 5% → 100), net debt/EBITDA (4× → 0, net cash → 100) and
cash conversion, FCF/net income (30% → 0, 100% → 100).

**Country & currency overlay** (`analysis/country.py`). A country risk premium for the company's home country is added
to its cost of equity, scaled by λ (its exposure to that country). The defaults are approximate, rating-based figures and are
editable. The share's 1/3/5-year return is shown in its own currency and in your base currency, and the gap
between them is the currency effect.

**Inputs you must check before relying on a valuation.** Only the USD risk-free rate is live (^TNX). The EUR, GBP,
CHF and ZAR risk-free rates, the equity risk premium and the country risk premiums are labelled placeholders on the page.

## Phase 3, in plain English

**Portfolio** (`analysis/portfolio.py`). Your current holdings are priced over the period and converted to your
base currency (ZAR, USD or EUR) at each day's FX rate, so currency moves are part of the return. Shares on
different exchanges trade on different days; prices are carried forward over another market's holidays, never backward.
- *Volatility*: daily return standard deviation × √252.
- *Beta*: how much the portfolio moves when the benchmark moves 1%, measured in the same currency.
- *Max drawdown*: the worst fall from a peak.
- *Correlation*: −1 to +1; near +1 means two holdings add little diversification.
- *Attribution*: each holding's value change ÷ the starting portfolio value. These add up exactly to the total return,
  so they can be summed by sector or region.
- The period return uses today's quantities held constant ("how has what I own behaved?"), not your actual trading history.

**Macro** (`data/macro.py`). Annual country data from the World Bank (actuals) or the IMF World Economic Outlook
(including forecasts, drawn dashed); policy rates from FRED and the ECB; US Treasury and euro AAA yield curves
(today vs. a chosen earlier date); SARB's current rates; and any FRED, World Bank or ECB series by code.
FRED works without a key via its public CSV; add `FRED_API_KEY` to use the JSON API instead.

**News & events** (`data/news.py`). Company news from Yahoo Finance plus a Google News search; any topic or country through
Google News; geopolitical events from GDELT. The GDELT map counts articles by the publisher's country, and the timeline shows
the average tone of coverage (below 0 = negative).

**Dashboard** (`Home.py`). Choose and order widgets: market snapshot, portfolio, watchlist, key rates, headlines, build progress.

## Phase 4, in plain English

**Backtester** (`quant/backtest.py`). Give it decisions ("on this date, hold these weights"). It trades each decision
one day *after* it was made, so it never uses a price it could not have traded at. Weights drift with prices between
rebalances. Every trade pays a cost (default 10 bps of the value traded). Uninvested money earns 0%.

**Strategies** (`quant/strategies.py`): buy & hold; equal weight rebalanced monthly; **momentum rotation** (hold the
top N by 12-month return, skipping the last month); **trend following** (invested when the 50-day average is above
the 200-day); **inverse volatility** (calmer assets get more money); **factor top group** (hold the best-scoring 20%).

**Factor research** (`quant/factors.py`). Each month, rank the universe by a factor (momentum, low volatility, value =
earnings yield, quality = ROE) and check whether the ranking predicted next month's returns:
- the **information coefficient** (rank correlation; meaningful above about 0.03–0.05 with a t-stat above 2);
- **quantile returns** (do top-ranked stocks beat bottom-ranked ones?).

**Point-in-time rule**: annual results only count as known 90 days (editable) after the fiscal year end.

**Parameter sweep**: runs a strategy across a grid of settings and shows a heatmap. Prefer areas where neighbouring
settings also work, not the single best cell, which may just be luck (overfitting).

**Strategy arena**: several strategies side by side, with CAGR, volatility, Sharpe, Sortino, max drawdown, Calmar,
hit rate, turnover and costs.

**Volatility surface** (`analysis/options.py`). Implied volatility is computed from option mid prices (Black-Scholes,
solved by bisection) for out-of-the-money options, then shown as a 3D surface, a smile per expiry and an
at-the-money term structure. US options come from Yahoo; BTC/ETH from Deribit, using its mark IV when prices are unusable.

**Limits you must keep in mind:**
- **Survivorship bias**: the universe is today's companies, with no delisted firms, so results look better than reality.
- Fundamentals cover about 4 years, so value and quality tests are short.
- Market value uses today's share count throughout.

## AI research (Phase 5, part 1), in plain English

The AI never browses the web or touches the app directly. It can only *ask* for one of the terminal's tools (quote, fundamentals,
DCF, peers, quality, macro, policy rates, news); the app runs it and sends the result back, including its source and date. The
model must cite those. Providers are pluggable like data sources: **Google AI Studio (Gemini)** and **Groq** use free keys
(`GEMINI_API_KEY`, `GROQ_API_KEY` in Settings). Defaults are the models verified with the tools: `gemini-3.5-flash` and
`openai/gpt-oss-120b`. Free tiers have rate limits; busy errors are retried automatically. Treat the model's *reasoning* with
care: in testing, one model quoted correct numbers but drew the wrong conclusion from them.

## Phase plan

1. **Foundation** ✅: shell, data layer, cache, Settings, quotes, charts, watchlists
2. **Equity research core** ✅: DCF, multiples, quality score, country/currency overlay (yfinance + SEC EDGAR)
3. **Portfolio, macro, news and dashboard** ✅
4. **Quant lab** ✅
5. Agentic research (needs Anthropic API key)
6. Everything else
