import pandas as pd
import pytest

from gterm import reports, workflows as wf
from gterm.ai.providers import PROVIDERS, Provider, Turn


def g(nodes, links):
    return wf.build_graph([{**n, "x": 0, "y": 0} for n in nodes], links)


def test_parse_and_topological_order():
    graph = g([{"name": "tickers"}, {"name": "dcf"}, {"name": "quality"}, {"name": "merge"}],
              [(1, 2, "input_1"), (1, 3, "input_1"), (2, 4, "input_1"), (3, 4, "input_2")])
    nodes = wf.parse(graph)
    assert nodes["4"]["inputs"] == {"input_1": ["2"], "input_2": ["3"]}
    order = wf.topo_order(nodes)
    assert order.index("1") < order.index("2") < order.index("4") and order.index("3") < order.index("4")


def test_cycle_is_refused():
    graph = g([{"name": "filter"}, {"name": "sort"}], [(1, 2, "input_1"), (2, 1, "input_1")])
    with pytest.raises(wf.WorkflowError, match="loop"):
        wf.run(graph)


def test_dcf_quality_merge_filter_sort_on_demo_data():
    graph = g([{"name": "tickers", "data": {"symbols": "SAP.DE, AAPL, JPM"}}, {"name": "dcf"}, {"name": "quality"},
               {"name": "merge"}, {"name": "filter", "data": {"condition": "quality > 0"}},
               {"name": "sort", "data": {"column": "quality", "order": "descending"}}],
              [(1, 2, "input_1"), (1, 3, "input_1"), (2, 4, "input_1"), (3, 4, "input_2"), (4, 5, "input_1"), (5, 6, "input_1")])
    res = wf.run(graph)
    assert not any(r.error for r in res.results.values()), {k: r.error for k, r in res.results.items()}
    dcf = res.results["2"].output
    assert set(dcf["symbol"]) == {"SAP.DE", "AAPL", "JPM"}
    assert "error" in dcf.columns and dcf.set_index("symbol").loc["JPM", "error"]  # bank: DCF refused, row kept with the reason
    out = res.results["6"].output
    assert list(out["quality"]) == sorted(out["quality"], reverse=True)
    assert all(p["result"].get("source") or p["result"].get("error") for p in res.provenance)


def test_errors_skip_downstream_and_bad_inputs_are_explained():
    graph = g([{"name": "tickers", "data": {"symbols": ""}}, {"name": "dcf"}, {"name": "sort", "data": {"column": "nope"}}],
              [(1, 2, "input_1"), (2, 3, "input_1")])
    res = wf.run(graph)
    assert "at least one ticker" in res.results["1"].error
    assert res.results["2"].skipped and res.results["3"].skipped
    lonely = wf.run(g([{"name": "dcf"}], []))
    assert "Nothing is connected" in lonely.results["1"].error
    bad_filter = wf.run(g([{"name": "tickers", "data": {"symbols": "SAP.DE"}}, {"name": "quotes"},
                           {"name": "filter", "data": {"condition": "nonexistent > 1"}}], [(1, 2, "input_1"), (2, 3, "input_1")]))
    assert "Could not apply" in bad_filter.results["3"].error


class FakeAI(Provider):
    def chat(self, model, system, messages, tools_):
        assert "| symbol |" in messages[0]["text"] and "Rank them" in messages[0]["text"]
        return Turn("- SAP looks best [demo, 2026].")


def test_ai_block_and_report_with_sources(monkeypatch):
    monkeypatch.setitem(PROVIDERS, "Fake", FakeAI)
    graph = g([{"name": "tickers", "data": {"symbols": "SAP.DE, AAPL"}}, {"name": "quotes"},
               {"name": "ai", "data": {"provider": "Fake", "model": "m", "instructions": "Rank them."}},
               {"name": "report", "data": {"title": "Test report"}}],
              [(1, 2, "input_1"), (2, 3, "input_1"), (3, 4, "input_1")])
    res = wf.run(graph)
    assert res.results["3"].output.startswith("- SAP looks best") and len(res.report_paths) == 1
    rep = reports.load(res.report_paths[0])
    assert rep["title"] == "Test report" and len(rep["steps"]) == 2
    assert "| 1 | quotes #2 | get_quote | symbol=SAP.DE |" in reports.to_markdown(rep)


def test_examples_are_valid_and_library_round_trip():
    for name, graph in wf.EXAMPLES.items():
        wf.topo_order(wf.parse(graph))
    p = wf.save_workflow("my flow!", wf.EXAMPLES["Example: macro dashboard (no AI)"])
    assert p.name == "my_flow.json" and "my_flow" in wf.list_workflows()
    wf.delete_workflow("my_flow")
    assert "my_flow" not in wf.list_workflows()


def test_md_table():
    t = wf.md_table(pd.DataFrame({"a": [1.23456, None], "b": ["x|y", "z"]}))
    assert t.splitlines()[0] == "| a | b |" and "1.235" in t and "x/y" in t
