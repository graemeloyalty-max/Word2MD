import pandas as pd
import pytest

from gterm.analysis import ownership as ow
from gterm.data import registry
from gterm.data.base import DataSourceError


def test_classify_transactions():
    assert ow.classify("Sale at price 352.77 per share.") == "sale"
    assert ow.classify("Purchase at price 10 per share.") == "buy"
    assert ow.classify("Stock Gift at price 0.00 per share.") == "other"
    assert ow.classify("", "Stock Award(Grant)") == "other"


def test_insider_activity_nets_buys_and_sales():
    tx = pd.DataFrame({"Insider": ["A", "A", "B", "B"], "Position": ["CEO", "CEO", "CFO", "CFO"],
                       "Text": ["Purchase at price 10", "Sale at price 12", "Stock Gift", "Sale at price 11"],
                       "Shares": [100, 40, 500, 10], "Value": [1000, 480, 0, 110],
                       "Start Date": pd.to_datetime(["2026-10-01", "2026-10-02", "2026-10-03", "2025-01-01"])})
    a = ow.insider_activity(tx).set_index("Insider")
    assert a.loc["A", "net_shares"] == 60 and a.loc["A", "net_value"] == 520
    assert a.loc["B", "net_shares"] == -10  # the gift is neither bought nor sold
    recent = ow.insider_activity(tx, since=pd.Timestamp("2026-01-01")).set_index("Insider")
    assert recent.loc["B", "sold_shares"] == 0
    assert ow.insider_activity(pd.DataFrame()).empty


def test_networks_on_demo_data():
    o = registry.ownership("AAPL")
    nodes, edges = ow.company_network("AAPL", o.institutions, o.funds, ow.insider_activity(o.insider_transactions))
    assert nodes.iloc[0]["id"] == "AAPL" and len(edges) == len(nodes) - 1
    assert set(nodes["kind"]) == {"company", "institution", "fund", "insider"}
    hold = {s: registry.ownership(s).institutions for s in ["AAPL", "MSFT", "SAP.DE"]}
    nodes2, edges2, shared = ow.common_ownership(hold)
    assert set(nodes2[nodes2["kind"] == "company"]["id"]) == {"AAPL", "MSFT", "SAP.DE"}
    assert len(shared) and (shared["Companies"] >= 2).all()
    for h in shared["Holder"]:  # a shared holder has one edge per company it owns
        assert sum(1 for a, b in edges2 if b == f"h:{h}") == shared.set_index("Holder").loc[h, "Companies"]
    with pytest.raises(DataSourceError):
        registry.ownership("^GSPC")


def test_shared_holders_do_not_overlap():
    hold = {s: registry.ownership(s).institutions for s in ["AAPL", "MSFT", "NVDA"]}
    nodes, _, shared = ow.common_ownership(hold)
    pts = nodes[nodes["kind"] == "shared"][["x", "y"]].round(6)
    assert not pts.duplicated().any()
    assert shared[[c for c in ["AAPL", "MSFT", "NVDA"]]].dtypes.eq(float).all()
