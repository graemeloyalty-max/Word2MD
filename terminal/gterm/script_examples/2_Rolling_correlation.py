# Rolling 60-day correlation between two assets, both converted to rand.
a, b = "SAP.DE", "^GSPC"
pa_ = prices(a, "5y")["Close"]
pb_ = prices(b, "5y")["Close"]
ra = to_ccy(pa_, quote(a).currency, "ZAR").pct_change()
rb = to_ccy(pb_, quote(b).currency, "ZAR").pct_change()
corr = ra.rolling(60).corr(rb).dropna()
show(go.Figure(go.Scatter(x=corr.index, y=corr.values, name="60d correlation")).update_layout(
    title=f"Rolling 60-day correlation: {a} vs {b} (in ZAR)", yaxis_range=[-1, 1]))
print(f"Latest: {corr.iloc[-1]:.2f}, average: {corr.mean():.2f}")
