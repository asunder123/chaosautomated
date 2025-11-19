import streamlit as st
import boto3, json, time, tempfile, subprocess
from pathlib import Path
from langgraph.graph import StateGraph, END

# ---------------- CONFIG ----------------
MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
MAX_TOKENS = 200
MAX_RETRIES = 3
RAG_BUCKET = "rag-terraform-context"  # S3 bucket for RAG context

# ---------------- Claude Client ----------------
class ClaudeClient:
    def __init__(self, session):
        self.client = session.client("bedrock-runtime")

    def invoke(self, prompt: str, max_tokens: int = MAX_TOKENS) -> str:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": 0,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        }
        resp = self.client.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )
        data = json.loads(resp["body"].read())
        return "".join(c.get("text", "") for c in data.get("content", []) if isinstance(c, dict))

# ---------------- RAG Retrieval ----------------
def fetch_rag_context(s3_client, bucket: str, prefix: str = "terraform/") -> str:
    """Retrieve RAG snippets from S3 bucket."""
    context_parts = []
    try:
        objs = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        for obj in objs.get("Contents", []):
            body = s3_client.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read().decode("utf-8")
            context_parts.append(body[:1000])  # Limit each snippet for token safety
    except Exception as e:
        context_parts.append("Best practices: Use aws_instance for EC2, aws_s3_bucket for S3.")
    return "\n".join(context_parts)

# ---------------- Terraform Engine ----------------
class TerraformEngine:
    def __init__(self, region: str):
        self.region = region

    def run_cmd(self, folder: Path, args: list) -> tuple[int, str]:
        proc = subprocess.run(["terraform", *args], cwd=folder, capture_output=True, text=True)
        return proc.returncode, proc.stdout

    def write_main_tf(self, folder: Path, hcl: str):
        base = f'terraform {{ required_version = ">= 1.0.0" }}\nprovider "aws" {{ region = "{self.region}" }}\n\n'
        (folder / "main.tf").write_text(base + hcl)

    def read_main_tf(self, folder: Path) -> str:
        return (folder / "main.tf").read_text()

# ---------------- LangGraph Workflow ----------------
def build_workflow(ai_client: ClaudeClient, engine: TerraformEngine, workdir: Path, rag_context: str):
    graph = StateGraph()

    # Node: Generate initial HCL
    def generate_hcl(state):
        prompt = (
            f"Context:\n{rag_context}\n\n"
            f"Task: Generate minimal Terraform HCL for AWS based on this description:\n{state['infra_prompt']}\n"
            "Include provider and region. Output only HCL."
        )
        hcl = ai_client.invoke(prompt)
        engine.write_main_tf(workdir, hcl)
        return {"hcl": hcl}

    # Node: Validate HCL
    def validate_hcl(state):
        rc, out = engine.run_cmd(workdir, ["validate"])
        return {"valid": rc == 0, "error": out}

    # Node: Fix HCL
    def fix_hcl(state):
        current_code = engine.read_main_tf(workdir)
        prompt = (
            f"Context:\n{rag_context}\n\n"
            f"Error:\n{state['error'][:600]}\n\n"
            f"Current Terraform code:\n{current_code[:1500]}\n\n"
            "Task: Fix the code so it passes `terraform validate`. Output only full corrected HCL."
        )
        corrected_hcl = ai_client.invoke(prompt)
        engine.write_main_tf(workdir, corrected_hcl)
        return {"hcl": corrected_hcl}

    # Node: Terraform Steps
    def terraform_steps(state):
        for step, args in [("init", ["init", "-input=false"]),
                           ("plan", ["plan", "-input=false", "-no-color"]),
                           ("apply", ["apply", "-input=false", "-auto-approve"])]:
            rc, out = engine.run_cmd(workdir, args)
            if rc != 0:
                return {"success": False, "error": out}
        return {"success": True}

    # Add nodes
    graph.add_node("GenerateHCL", generate_hcl)
    graph.add_node("ValidateHCL", validate_hcl)
    graph.add_node("FixHCL", fix_hcl)
    graph.add_node("TerraformStep", terraform_steps)

    # Edges
    graph.add_edge("GenerateHCL", "ValidateHCL")
    graph.add_conditional_edges("ValidateHCL", lambda s: "TerraformStep" if s["valid"] else "FixHCL")
    graph.add_edge("FixHCL", "ValidateHCL")
    graph.add_conditional_edges("TerraformStep", lambda s: END if s["success"] else "FixHCL")

    return graph.compile()

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="LangGraph Terraform Orchestration with RAG", layout="wide")
st.title("LangGraph-Driven Autonomous Terraform Engine (RAG + Claude)")

# AWS Login
st.header("AWS Login")
with st.form("login"):
    access = st.text_input("Access Key", type="password")
    secret = st.text_input("Secret Key", type="password")
    region = st.text_input("Region", "us-east-1")
    submit = st.form_submit_button("Continue")

if not submit:
    st.stop()

try:
    session = boto3.Session(aws_access_key_id=access, aws_secret_access_key=secret, region_name=region)
    sts = session.client("sts").get_caller_identity()
    st.success(f"Logged in as {sts['Arn']}")
except Exception as e:
    st.error(f"Invalid AWS credentials: {e}")
    st.stop()

# Prompt
st.header("Describe Infrastructure")
infra_prompt = st.text_area("Prompt", "1 EC2 instance and 1 S3 bucket.")
if not st.button("Start Autonomous Provisioning"):
    st.stop()

# Fetch RAG context from S3
s3_client = session.client("s3")
rag_context = fetch_rag_context(s3_client, RAG_BUCKET)

# Initialize
ai_client = ClaudeClient(session)
engine = TerraformEngine(region)
workdir = Path(tempfile.gettempdir()) / f"tf_{int(time.time())}"
workdir.mkdir()

# Build and run workflow
workflow = build_workflow(ai_client, engine, workdir, rag_context)
state = {"infra_prompt": infra_prompt}
result = workflow.run(state)

# Display result
if result.get("success"):
    st.success("Provisioning complete!")
    st.code(engine.read_main_tf(workdir), language="hcl")
else:
    st.error("Failed after retries.")
    st.write("Last error encountered:")
    st.text(result.get("error", "No error details available"))
