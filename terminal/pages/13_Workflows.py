import pandas as pd
import streamlit as st

from gterm import ui, workflows
from gterm.components.flow import flow_editor

ui.page_setup("Workflows", "🧩")
st.markdown("Chain data, analysis and AI blocks on a canvas, then press **Run**. Drag blocks from the left, connect a block's "
            "right-hand dot to the next block's left-hand dot, and edit settings inside each block. Blue = data, green = analysis, "
            "purple = steps, orange = AI, red = output.")

lib = workflows.list_workflows()
c1, c2 = st.columns([3, 1])
names = ["(new workflow)"] + list(lib)
choice = c1.selectbox("Workflow", names, index=1)
graph_id = f"{choice}:{st.session_state.get('wf_reload', 0)}"
if c2.button("↺ Reload from saved", use_container_width=True):
    st.session_state["wf_reload"] = st.session_state.get("wf_reload", 0) + 1
    st.rerun()

current = flow_editor(lib.get(choice), graph_id, key="flow")
graph = current if current else lib.get(choice)
n_blocks = len((graph or {}).get("drawflow", {}).get("Home", {}).get("data", {})) if graph else 0
st.caption(f"{n_blocks} block(s) on the canvas")

b1, b2, b3, b4 = st.columns([1, 2, 1, 1])
run = b1.button("▶ Run", type="primary", use_container_width=True, disabled=n_blocks == 0)
save_as = b2.text_input("Save as", "" if choice.startswith(("(", "Example")) else choice, placeholder="workflow name",
                        label_visibility="collapsed")
if b3.button("💾 Save", use_container_width=True, disabled=not graph):
    if save_as.strip():
        workflows.save_workflow(save_as, graph)
        st.toast("Saved")
        st.rerun()
    else:
        st.warning("Enter a name to save.")
if b4.button("🗑 Delete", use_container_width=True, disabled=choice.startswith(("(", "Example"))):
    workflows.delete_workflow(choice)
    st.rerun()

if run:
    status = st.status("Running workflow…", expanded=True)
    try:
        res = workflows.run(graph, on_node=lambda r: status.write(
            f"{'⚠️' if r.error else '⏭️' if r.skipped else '✅'} {r.name} #{r.id}" + (f": {r.error or r.note}" if (r.error or r.note) else "")))
    except workflows.WorkflowError as exc:
        status.update(label="Could not run", state="error")
        st.error(str(exc))
    else:
        bad = [r for r in res.results.values() if r.error]
        status.update(label=f"Done: {len(res.order)} block(s), {len(bad)} error(s)", state="error" if bad else "complete", expanded=False)
        st.session_state["wf_last"] = res
res = st.session_state.get("wf_last")
if res:
    st.subheader("Results by block")
    for nid in res.order:
        r = res.results[nid]
        icon = "⚠️" if r.error else "⏭️" if r.skipped else "✅"
        with st.expander(f"{icon} {r.name} #{nid}", expanded=r.name in ("ai", "report") or bool(r.error)):
            if r.error:
                st.error(r.error)
            elif r.skipped:
                st.caption(r.note)
            elif isinstance(r.output, pd.DataFrame):
                st.dataframe(r.output, use_container_width=True, hide_index=True)
            elif isinstance(r.output, list):
                st.write(", ".join(r.output))
            else:
                st.markdown(str(r.output))
    if res.provenance:
        st.caption(f"{len(res.provenance)} data calls; sources: "
                   + ", ".join(sorted({str(p['result'].get('source')) for p in res.provenance if isinstance(p['result'], dict) and p['result'].get('source')})))
with st.expander("How workflows run"):
    st.markdown(workflows.__doc__)
