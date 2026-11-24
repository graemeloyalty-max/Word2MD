# Build log

A record of every essential step taken to build the terminal, newest phase last.

---

## Phase 1: Foundation (24 Nov 2026)

**Goal:** app shell, a data adapter layer with yfinance, local cache, Settings page, global quotes,
charts and watchlists. **Cost:** nothing beyond Claude Pro.

### Step 0: Environment check
- The repo (`Word2MD`) held an unrelated document converter. To leave it untouched, the terminal lives in its own folder, `terminal/`.
- Installed Python packages: streamlit, yfinance, plotly, pandas, numpy, python-dotenv, pytest.
- **Finding:** the cloud sandbox Claude Code was building in could not reach Yahoo Finance (its network blocks it).
  A request to allow the Yahoo hosts was declined, so the build went ahead offline.
- **Decision:** add a second data source, `demo (synthetic)`, which produces deterministic made-up prices.
  The app can then be built and tested with no internet. It also proves the adapter design works: two sources
  plug in behind one interface. On your own machine, `auto` mode picks yfinance.

### Step 1: Data adapter layer (`gterm/data/`)
- `base.py` defines the contract: a `DataSource` with `get_prices()` and `get_quote()`, returning
  `PriceHistory` / `Quote` objects that always carry `source` and `as_of`.
- `yfinance_source.py`: the real adapter. It normalises timestamps to dates, reads the trading
  currency, and falls back gracefully when Yahoo's `.info` is flaky.
- `demo_source.py`: the offline adapter, with 15 realistic symbols (US, Europe, indices, FX, crypto) and any other symbol on demand.
- `registry.py`: the only thing pages import. It picks the active source (auto/yfinance/demo) and wraps every call in the cache.

### Step 2: Core services (`gterm/core/`)
- `cache.py`: SQLite cache. Prices are kept 6 h and quotes 5 min. Cached data keeps its original retrieval time.
- `config.py`: preferences in `data/settings.json`, API keys in `.env` (git-ignored). Lists every key future
  modules will need, so paid modules can say "add your key in Settings" instead of crashing.
- `fx.py`: converts a price series at each day's historical FX rate, with no look-ahead. Handles pence (GBp).
- `indicators.py`: SMA, EMA, RSI (Wilder), MACD, Bollinger, annualised volatility, max drawdown.
- `watchlists.py`: watchlists and price alerts in `data/workspace.sqlite`, seeded with three starter lists.

### Step 3: Interface (Streamlit multi-page app)
- **Home**: market snapshot (S&P 500, EURO STOXX 50, EUR/USD, USD/ZAR, BTC) and a progress table.
- **Quotes & Charts**: any symbol; candlestick or line chart; SMA/EMA/Bollinger overlays; volume, RSI and MACD panels;
  return, volatility and drawdown metrics; a toggle to view prices in your base currency; symbol search; CSV download; add to watchlist.
- **Watchlists**: quote table with 3-month sparklines, source and time per row; add/remove symbols; price alerts;
  rebased comparison chart; optional 60 s auto-refresh.
- **Settings**: base currency (USD/EUR/ZAR), data source, API keys, cache stats and clearing.
- The sidebar always shows which data source is live, with a warning banner when demo data is in use.

### Step 4: Tests (18 passing)
- Indicators checked against hand calculations and known extremes (an always-rising series gives RSI 100).
- FX: uses each date's rate, never a future one; pence → pounds; pence → USD.
- Data layer: the adapter contract, determinism, weekly resampling, the cache flag preserving the retrieval time.
- Watchlist CRUD, alerts, and API keys written to `.env`.
- One bug caught: `df.hist` is a pandas built-in method, so the MACD histogram column must be read as `df["hist"]`. Fixed in the test and the chart page.

### Step 5: End-to-end check
- Launched the real app headless and loaded every page in a browser (Playwright): no exceptions on any page.
- Verified SAP.DE switched to USD shows "converted EUR->USD at daily EURUSD=X close" under the chart.

### Step 6: Packaging
- `requirements.txt`, `.gitignore` (keeps `.env` and `data/` out of git), `.env.example`.
- `.devcontainer/` so GitHub Codespaces installs and launches the app automatically.
- README with plain-English explanations of every model (ground rule 5).

**Live data:** verified in Step 7.

### Step 7: Live-data verification (after network access was granted)
- You added the six Yahoo Finance domains to the cloud environment's allowed list. `auto` mode then switched to yfinance by itself.
- Checked real quotes for AAPL, SAP.DE, ASML.AS, SHEL.L, NESN.SW, ^GSPC, USD/ZAR and BTC-USD; symbol search works ("siemens" → SIE.DE, ...).
- **Bug found by real data:** Yahoo prices London stocks in pence (`GBp`) and JSE stocks in cents (`ZAc`). The adapter
  upper-cased the currency code, turning `GBp` into `GBP`, so Shell's 3,552p was read as £3,552 and converted to $4,660 instead of $46.60.
  The fix keeps Yahoo's case for minor units. A regression test was added (19 tests now). Shell now converts to $46.60 and Naspers (NPN.JO) to $36.44.
- UI: the currency moved into the "Last (GBp)" label so long prices aren't cut off.
- All pages loaded in a browser on live data with no errors.
- Note: the watchlist comparison chart shows NVDA −14% three sessions ago, then a rebound. That is the data Yahoo returned, not a charting bug.
