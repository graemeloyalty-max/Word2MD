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

---

## Phase 2: Equity research core (24 Nov 2026)

**Goal:** an equity research page for any US or European ticker with five parts: normalised statements, an
editable DCF, peer multiples, a quality score and a country/currency overlay. SEC EDGAR provides longer US history.
**Cost:** nothing new.

### Step 1: Explore the data
- yfinance returns 4–5 fiscal years of income statement, balance sheet and cash flow for both US and European companies.
- The line names differ by accounting standard. SAP (IFRS) reports `Interest Paid Cfo`, `Interest Received Cfi` and
  `Purchase Of Intangibles`; Apple (US GAAP) has none of these.
- **SEC EDGAR is blocked** by the sandbox network (data.sec.gov, www.sec.gov), as Yahoo was.

### Step 2: Statement normaliser (`gterm/core/statements.py`)
- Maps every source's labels to ~30 standard lines (revenue, EBIT, FCF, net debt, ...), oldest year → newest.
- IFRS → US GAAP-style adjustments: interest paid/received moved into operating cash flow, capitalised development
  (intangibles) counted as capex, lease liabilities in debt. Each adjustment is recorded and shown on the page.

### Step 3: Data layer
- `Fundamentals` and `KeyStats` added to the adapter interface; implemented for yfinance and demo (demo statements use
  IFRS-style labels for European names so the normaliser is exercised offline).
- `edgar_source.py`: reads EDGAR company facts and keeps only full-year 10-K values. A restated figure replaces
  the original, quarters filed inside 10-Ks are dropped, and only years *older* than yfinance's are added.
- SEC requires a User-Agent with your name and email. It's a Settings field and is never hard-coded.
- Fundamentals and key stats are cached for 24 h.

### Step 4: Models (`gterm/analysis/`)
- `dcf.py`: FCFF DCF with a growth fade, CAPM + country-risk cost of equity, WACC, an equity bridge, a 5×5 sensitivity grid, and a reverse DCF (bisection).
- `multiples.py`: default sector × region peer lists (11 sectors, US/Europe) and implied prices at peer medians.
- `quality.py`: ROIC, margin variation, leverage and cash conversion, each scored 0–100.
- `country.py`: approximate CRPs and risk-free placeholders, all labelled as such on screen.

### Step 5: Page (`pages/4_Equity_Research.py`)
- A summary strip (price, DCF value, peer value, quality) above five tabs. Every assumption is an input with its default's source written underneath.
- Three currencies are handled at once: statement currency (Shell reports in USD), trading currency (GBp) and your base currency.

### Step 6: Bugs and edge cases found with live data
- **Shell's beta from yfinance is −0.25.** The input rejected it and the page crashed. It now accepts the value, but defaults to 1.0 when the reported beta is below 0.3, and says why.
- **JPMorgan's DCF came out at −$1,481 per share** (banks' "free cash flow" is meaningless). Banks/insurers now show
  "n/a" for DCF and quality, with an explanation, and a negative-FCF DCF never reaches the summary.
- The base-currency caption said "same currency" when it had converted GBp→USD. Fixed.
- Peer names had stray padding from Yahoo. Fixed.
- Layout: chart legends overlapped axis titles, and a metric label was truncated. Fixed.

### Step 7: Tests (40 passing, 21 new)
- DCF checked against closed-form answers: a zero-growth perpetuity = FCF/r; constant growth = Gordon formula; the equity bridge;
  WACC weights and tax shield; λ-scaled CRP; the growth fade; a reverse DCF round trip; sensitivity moves in the right direction.
- Normaliser: IFRS adjustments applied, US GAAP left alone. Quality on hand-built statements gives exactly 100.
- Multiples: implied prices for equity and EV multiples; the median ignores an outlier.
- EDGAR parsing on a fixture: quarters dropped, restatements win, 10-Qs ignored, only older years merged.
- FX: pence → USD at the latest rate.

### Step 8: End-to-end on live data
- SAP.DE, AAPL, SHEL.L and JPM: all five tabs load with no exceptions.
- Results at default assumptions (24 Nov 2026): SAP DCF €277 vs €190 price, peers €149, quality 87;
  Apple DCF $176 vs $353, peers $175, quality 99; Shell DCF 2,516p vs 3,552p, peers 4,012p; JPM DCF n/a (bank).
- **Note on SAP:** the high DCF value comes mostly from a low WACC (5.8%), which in turn comes from the 2.7% EUR
  risk-free *placeholder* and a 0.70 beta. Check the risk-free rate before trusting it.

**Not yet verified:** SEC EDGAR live (network blocked; parsing tested on a fixture). The toggle shows a clear message instead of failing.

### Step 9: Repository fix found while committing
- The `.gitignore` rule `data/` was meant for the local cache folder, but it also matched the code folder `gterm/data/`.
  **The data adapter layer had been missing from the Phase 1 commits**, so a fresh clone would not have run.
- Fixed by changing the rule to `/data/`. Verified by cloning the pushed branch into a clean folder and running all 40 tests there.

---

## Phase 3: Portfolio, macro, news and dashboard (24 Nov 2026)

**Goal:** portfolio tools, macro data (FRED, World Bank, IMF, ECB, SARB), news, GDELT events and a dashboard. **Cost:** nothing new.

### Step 1: Network check
- Every macro and news host was blocked (FRED, World Bank, IMF, ECB, SARB, GDELT, Google News). You were asked to allow them.
- Tracing a failure showed Yahoo *news* comes from a separate host, `nexus-finance-prod.media.yahoo.com`, which was added to the list.
- **Decision:** build the portfolio tools first (yfinance only, already allowed), and write each macro/news adapter as a pure
  parser plus a fetcher, so the parsers could be tested on sample responses while the hosts were blocked.

### Step 2: Portfolio tools
- `analysis/portfolio.py`: calendar alignment (forward-fill only), position values, volatility, beta, drawdown, Sharpe,
  correlation, contributions, and attribution by any grouping.
- `core/portfolio_loader.py`: prices each holding, converts it to the base currency at daily FX, and attaches sector/country/region.
- Holdings are stored in `data/workspace.sqlite` with an optional cost price. The page has an editable holdings grid, metrics against
  a benchmark (default URTH, the MSCI World ETF), growth and drawdown charts, allocation pies, attribution by sector and region,
  and a correlation heatmap.
- **Problem found live:** Yahoo's company-profile endpoint started refusing requests ("Invalid Crumb", HTTP 401, which is its rate
  limiting), so names and sectors disappeared. Fixes: (1) the last good profile is kept for 7 days and used when Yahoo refuses
  (never for prices); (2) the country is inferred from the exchange suffix (.SW → Switzerland, .JO → South Africa, ...).
- Verified on live data: sample portfolio in ZAR, +13.8% over 1 year; contributions sum exactly to the total.

### Step 3: Macro adapters and page
- `data/macro.py`: FRED (keyless CSV or JSON API with a key), World Bank, IMF DataMapper (with forecasts), ECB Data Portal
  (handles its daily/monthly/quarterly/annual date formats), SARB current rates. 12 h cache.
- `pages/6_Macro.py`: country comparison, policy rates, US and euro yield curves (today vs. earlier, with the 10y−2y spread),
  SARB, and a series explorer. Each fetch fails on its own with a message naming the blocked host; the page keeps working.

### Step 4: News and GDELT
- `data/news.py`: Yahoo Finance company news (old and new response formats), Google News RSS, GDELT article list and tone/volume timeline.
- `pages/7_News_and_Events.py`: company, country/topic and geopolitical tabs, with a coverage map and a tone chart.

### Step 5: Dashboard
- `Home.py` is now a dashboard of widgets you choose and order (saved in settings): market snapshot, portfolio, watchlist,
  key rates (US bill/5y/10y live from Yahoo; Fed and ECB from FRED/ECB), headlines, build progress.
- A stale-module error appeared after adding the new setting: Streamlit reloads page files, but not the `gterm` package. Restarting the
  app fixed it. **If you pull new code while the app is running, restart it.**

### Step 6: Tests (61 passing, 21 new)
- Portfolio: holiday alignment never backfills; contributions sum to the return; attribution weights; beta of a 2× levered copy = 2;
  drawdown; annualised return; region fallbacks; the holdings store; base-currency loading.
- Parsers on sample responses for FRED (both CSV header styles, "." for missing), World Bank (nulls, error message), IMF, ECB (daily
  and monthly), SARB (field-name case), RSS (publisher suffix stripped), Yahoo news (both formats), GDELT (articles, timeline);
  the yield curve picks the right date.
