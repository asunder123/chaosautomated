import streamlit as st
from pathlib import Path
import zipfile
from core.aws_claude import configure_boto3, ClaudeBedrock
from core.terraform import ensure_terraform
from core.nodes import (
    node_codegen,
    node_init,
    node_validate,
    node_plan,
    node_apply,
    node_verify,
    node_output,
    deduplicate_providers_in_dir
)

# --- Utility: Create ZIP of Terraform files ---
def create_zip(workdir: Path, zip_path: Path):
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in workdir.rglob("*.tf"):
            zipf.write(file, arcname=file.name)

# --- Corrected build_graph() ---
def build_graph(nodes: dict):
    """
    Build a simple linear graph for LangGraph execution.
    Ensures the workflow starts at 'codegen' and proceeds in order of nodes dict.
    """
    if not nodes or "codegen" not in nodes:
        raise ValueError("Graph must include 'codegen' as the starting node.")

    steps_order = list(nodes.keys())
    edges = [(steps_order[i], steps_order[i + 1]) for i in range(len(steps_order) - 1)]

    config = {
        "nodes": steps_order,
        "edges": edges,
        "start": "codegen",
        "end": steps_order[-1],
    }

    class GraphApp:
        def __init__(self, nodes, edges):
            self.nodes = nodes
            self.edges = edges

        def stream(self, state, config):
            for step in steps_order:
                state = nodes[step](state)  # ✅ FIXED: node function
                yield {"current_step": step, "state": state}

    return GraphApp(nodes, edges), config

# --- UI Setup ---
st.set_page_config(page_title="Terraform Orchestrator", layout="wide")
st.title("Terraform Orchestrator — LangGraph + Claude + Self-Healing")

region = st.text_input("AWS Region", "us-east-1")
access = st.text_input("Access Key ID", type="password")
secret = st.text_input("Secret Access Key", type="password")
token = st.text_input("Session Token (optional)", type="password")
prompt = st.text_area("Describe infrastructure", height=160)
run_btn = st.button("Run Orchestrator")

workdir = Path.cwd() / "tf_langgraph_selfheal"
workdir.mkdir(parents=True, exist_ok=True)

progress_bar = st.progress(0)
status_box = st.empty()
log_box = st.expander("Execution Logs", expanded=True)

# --- Workflow Execution ---
def run_workflow():
    try:
        sess, _, _ = configure_boto3(region, access, secret, token or None)
        claude = ClaudeBedrock(sess, model="anthropic.claude-3-haiku-20240307-v1:0", region=region)
        ensure_terraform(Path.home() / ".tfbin")

        env = {
            "AWS_ACCESS_KEY_ID": access,
            "AWS_SECRET_ACCESS_KEY": secret,
            "AWS_SESSION_TOKEN": token,
            "AWS_DEFAULT_REGION": region,
        }

        nodes = {
            "codegen": lambda s: node_codegen(s, claude, workdir, prompt),
            "init": lambda s: node_init(s, workdir, env),
            "validate": lambda s: node_validate(s, claude, workdir, env),
            "plan": lambda s: node_plan(s, claude, workdir, env),
            "apply": lambda s: node_apply(s, claude, workdir, env),
            "verify": lambda s: node_verify(s, workdir, env),
            "output": lambda s: node_output(s, workdir, env),
        }

        app, config = build_graph(nodes)
        state = {"steps": {}}
        steps_order = list(nodes.keys())
        total_steps = len(steps_order)

        with st.spinner("Executing workflow..."):
            for chunk in app.stream(state, config=config):
                current_step = chunk.get("current_step", "")
                if current_step in steps_order:
                    pct = int(((steps_order.index(current_step) + 1) / total_steps) * 100)
                    progress_bar.progress(pct)
                    status_box.info(f"Running step: **{current_step}**")
                log_box.write(chunk)

        progress_bar.progress(100)
        status_box.success("Workflow complete!")

        st.subheader("Execution Logs")
        st.json(state.get("steps", {}))

        st.subheader("Generated Terraform Files")
        deduplicate_providers_in_dir(workdir)
        for p in sorted(workdir.rglob("*.tf")):
            st.markdown(f"**{p.name}**")
            st.code(p.read_text(), language="hcl")

        zip_path = workdir / "terraform_files.zip"
        create_zip(workdir, zip_path)
        with open(zip_path, "rb") as f:
            st.download_button("Download All Files as ZIP", f, file_name="terraform_files.zip")

    except Exception as e:
        status_box.error(f"Non-recoverable error: {e}")
        st.error("Please check logs and retry.")
        if st.button("Retry Workflow"):
            run_workflow()
        st.stop()

if run_btn:
    if not prompt.strip():
        st.error("Please enter a prompt.")
        st.stop()
    run_workflow()