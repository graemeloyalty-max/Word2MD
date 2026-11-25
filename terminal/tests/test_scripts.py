import json

import pandas as pd

from gterm import scripts


def test_script_runs_with_helpers_and_show():
    code = ('df = prices("SAP.DE", "3mo")\nshow(df[["Close"]].tail(3), "tail")\n'
            'show(go.Figure(go.Scatter(y=[1, 2, 3])))\nshow({"a": 1})\nprint("source", registry.active_source_name())')
    r = scripts.run(code, timeout=60)
    assert r.ok, r.stderr
    assert "source demo" in r.stdout  # tests run on the offline demo source
    kinds = [o["kind"] for o in r.outputs]
    assert kinds == ["table", "chart", "json"] and r.outputs[0]["title"] == "tail"
    assert len(scripts.table(r.outputs[0])) == 3
    assert json.loads(r.outputs[2]["data"]) == {"a": 1}


def test_errors_point_at_user_code():
    r = scripts.run("x = 1\nraise ValueError('bad input')\n", timeout=30)
    assert not r.ok and r.returncode == 1
    assert 'your_script.py", line 2' in r.stderr and "ValueError: bad input" in r.stderr
    assert "runner.py" not in r.stderr


def test_timeout_stops_runaway_script():
    r = scripts.run("import time\nprint('start', flush=True)\ntime.sleep(30)\n", timeout=3)
    assert r.timed_out and not r.ok and "longer than 3 seconds" in r.stderr


def test_library_save_list_delete_and_examples_protected():
    lib = scripts.list_scripts()
    examples = [k for k in lib if k.startswith("Example:")]
    assert len(examples) == 3
    p = scripts.save_script("my test/script!", "print(1)")
    assert p.name == "my_test_script.py" and "my_test_script" in scripts.list_scripts()
    scripts.delete_script(p)
    assert "my_test_script" not in scripts.list_scripts()
    scripts.delete_script(lib[examples[0]])  # must be a no-op
    assert lib[examples[0]].exists()


def test_examples_compile():
    for name, path in scripts.list_scripts().items():
        compile(path.read_text(), name, "exec")
