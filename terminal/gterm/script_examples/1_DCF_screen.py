# DCF screen: value a list of companies with the terminal's default assumptions and rank by upside.
from gterm.ai.tools import run_dcf

tickers = ["AAPL", "MSFT", "SAP.DE", "ASML.AS", "NESN.SW"]
rows = []
for t in tickers:
    r = run_dcf(t)
    if "error" in r:
        print(t, "skipped:", r["error"])
        continue
    rows.append({"Ticker": t, "Price": r["price"], "Ccy": r["price_currency"], "DCF value": r["value_per_share"],
                 "Upside": r["upside"], "WACC": r["wacc"],
                 "Growth yrs 1-5": r["assumptions"]["growth_years_1_5"]})
df = pd.DataFrame(rows).sort_values("Upside", ascending=False)
show(df, "DCF upside at default assumptions")
show(px.bar(df, x="Ticker", y="Upside", title="DCF upside vs price"), "Upside chart")
print("Source:", registry.active_source_name())
