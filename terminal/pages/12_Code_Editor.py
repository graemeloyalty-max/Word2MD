import json

import plotly.io as pio
import streamlit as st

from gterm import scripts, ui

ui.page_setup("Code Editor", "🧑‍💻")
st.markdown("<style>textarea {font-family: ui-monospace, Menlo, Consolas, monospace !important; font-size: 0.85rem !important;}</style>",
            unsafe_allow_html=True)

lib = scripts.list_scripts()
c1, c2 = st.columns([3, 1])
choice = c1.selectbox("Script", ["(new script)"] + list(lib))
timeout = c2.number_input("Time limit (s)", 5, 600, 60, 5)
if choice == "(new script)":
    initial = '# Your script. Try:\ndf = prices("SAP.DE", "6mo")\nshow(df.tail(), "Last rows")\nprint(quote("SAP.DE"))\n'
else:
    initial = lib[choice].read_text()

code = st.text_area("Python", initial, height=380, key=f"code_{choice}", label_visibility="collapsed")
b1, b2, b3, b4 = st.columns([1, 2, 1, 1])
run = b1.button("▶ Run", type="primary", use_container_width=True)
name = b2.text_input("Save as", "" if choice.startswith(("(", "Example")) else choice, placeholder="script name",
                     label_visibility="collapsed")
if b3.button("💾 Save", use_container_width=True):
    if name.strip():
        scripts.save_script(name, code)
        st.toast(f"Saved {scripts.safe_name(name)}")
        st.rerun()
    else:
        st.warning("Enter a name to save.")
if b4.button("🗑 Delete", use_container_width=True, disabled=choice.startswith(("(", "Example"))):
    scripts.delete_script(lib[choice])
    st.rerun()

with st.expander("What's available in scripts"):
    st.markdown(scripts.__doc__)

if run:
    with st.spinner(f"Running (limit {timeout}s)…"):
        res = scripts.run(code, int(timeout))
    if res.ok:
        st.success(f"Finished in {res.seconds:.1f}s")
    elif res.timed_out:
        st.error(res.stderr)
    else:
        st.error(f"The script failed (exit code {res.returncode}). See the error below.")
    for i, o in enumerate(res.outputs):
        if o.get("title"):
            st.markdown(f"**{o['title']}**")
        if o["kind"] == "table":
            st.dataframe(scripts.table(o), use_container_width=True)
        elif o["kind"] == "chart":
            st.plotly_chart(pio.from_json(o["data"]), use_container_width=True, key=f"chart_{i}")
        elif o["kind"] == "json":
            st.json(json.loads(o["data"]))
        else:
            st.code(o["data"])
    if res.stdout.strip():
        st.markdown("**Printed output**")
        st.code(res.stdout, language="text")
    if res.stderr.strip() and not res.timed_out:
        with st.expander("Errors and warnings", expanded=not res.ok):
            st.code(res.stderr, language="text")
