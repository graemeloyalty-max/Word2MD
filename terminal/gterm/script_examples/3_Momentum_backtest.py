# Backtest momentum rotation on your own universe, with costs, versus equal-weight buy & hold.
universe = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "SAP.DE", "ASML.AS", "MC.PA", "NESN.SW", "SHEL.L"]
px_ = pd.DataFrame({s: to_ccy(prices(s, "5y")["Close"], quote(s).currency, "USD") for s in universe}).ffill().dropna()
mom = backtest.run(px_, strategies.momentum(px_, lookback=6, top_n=3), cost_bps=10, name="Momentum 6m, top 3")
bh = backtest.run(px_, strategies.buy_and_hold(px_), cost_bps=10, name="Buy & hold")
show(pd.DataFrame({r.name: r.summary() for r in (mom, bh)}).T.round(3), "Summary")
show(go.Figure([go.Scatter(x=r.equity.index, y=r.equity, name=r.name) for r in (mom, bh)]).update_layout(title="Growth of 1 (USD)"))
