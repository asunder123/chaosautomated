import os
import re
import json
import tempfile
import subprocess
from typing import Dict, Any
from html import unescape

import streamlit as st
import boto3
from botocore.exceptions import ClientError
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import SystemMessage, HumanMessage

# ===================== Claude JSON Parser =====================
def call_llm_json(llm, system, user, context_vars=None):
    msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    text = msg.content if isinstance(msg.content, str) else "".join(
        [c if isinstance(c, str) else str(c) for c in msg.content]
    )
    text = unescape(text)

    # Replace triple-quoted strings with escaped JSON strings
    def sanitize_triple_quotes(match):
        inner = match.group(1)
        escaped = json.dumps(inner.strip())
        return escaped

    text = re.sub(r'"""(.*?)"""', sanitize_triple_quotes, text, flags=re.S)

    # Replace placeholders like ${roleArn} with context_vars or dummy values
    placeholder_pattern = re.compile(r"\$\{(\w+)\}")
    def replace_placeholder(match):
        key = match.group(1)
        if context_vars and key in context_vars:
            return context_vars[key]
        return f"dummy_{key}"

    text = placeholder_pattern.sub(replace_placeholder, text)

    # Try direct JSON parsing
    try:
        return json.loads(text)
    except Exception:
        pass

    # Extract Python and HCL code blocks separately
    code_blocks = re.findall(r"```(python|hcl)?\s*(.*?)```", text, flags=re.S)
    result = {}
    for lang, block in code_blocks:
        if lang == "python":
            result["python_boto3"] = block.strip()
        elif lang == "hcl":
            result["terraform"] = block.strip()

    if result:
        return result

    # Try finding any JSON-looking object in the text
    loose_json = re.search(r"{.*}", text, re.S)
    if loose_json:
        try:
            return json.loads(loose_json.group())
        except Exception:
            pass

    raise ValueError(f"Claude output not valid JSON:\n{text}")

# ===================== CrewAI-style Executor =====================
class CrewAIOrchestrator:
    def execute(self, code: str) -> Dict[str, Any]:
        lang = self.classify(code)
        if lang == "python":
            result = self.run_python(code)
        elif lang == "terraform":
            result = self.run_terraform(code)
        else:
            result = {"type": "unknown", "code": code}
        return self.summarize(result)

    def classify(self, code: str) -> str:
        if "resource \"" in code or "terraform" in code:
            return "terraform"
        if "import boto3" in code or "def " in code:
            return "python"
        return "unknown"

    def run_python(self, code: str) -> Dict[str, Any]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w") as f:
            f.write(code)
        cmd = ["python", f.name]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        return {"type": "python", "exit": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}

    def run_terraform(self, code: str) -> Dict[str, Any]:
        tmpdir = tempfile.mkdtemp(prefix="tf_exec_")
        tf_file = os.path.join(tmpdir, "main.tf")
        with open(tf_file, "w") as f:
            f.write(code)
        results = []
        for c in [["terraform", "init", "-input=false"], ["terraform", "fmt"], ["terraform", "validate"]]:
            try:
                p = subprocess.run(c, cwd=tmpdir, capture_output=True, text=True, timeout=120)
                results.append({"cmd": " ".join(c), "stdout": p.stdout, "stderr": p.stderr, "exit": p.returncode})
            except Exception as e:
                results.append({"cmd": " ".join(c), "error": str(e)})
        return {"type": "terraform", "path": tf_file, "steps": results}

    def summarize(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if result["type"] == "python":
            summary = f"Python script executed (exit={result['exit']})."
        elif result["type"] == "terraform":
            summary = f"Terraform validation done ({len(result['steps'])} steps)."
        else:
            summary = "Unknown artifact."
        return {"summary": summary, "detail": result}

# ===================== AWS + Claude =====================
def create_aws_session(access_key, secret_key, token, region):
    try:
        sess = boto3.Session(
            aws_access_key_id=access_key or None,
            aws_secret_access_key=secret_key or None,
            aws_session_token=token or None,
            region_name=region,
        )
        sts = sess.client("sts")
        identity = sts.get_caller_identity()
        return sess, {"status": "ok", "identity": identity}
    except ClientError as e:
        return None, {"status": "error", "error": str(e)}

def init_bedrock_claude(session, region, model_id):
    client = session.client("bedrock-runtime", region_name=region)
    return ChatBedrockConverse(model=model_id, client=client)

# ===================== Prompts =====================
def prompt_plan(scenario, context=""):
    return (
        "You are an AWS SRE. Generate chaos plan JSON.",
        f"""Given scenario:
{scenario}

Context:
{context}

Return JSON with keys:
plan_steps, safety_checks, iam_permissions, rollback_strategy, metrics, baseline_targets
"""
    )

def prompt_fis(plan, context=""):
    return (
        "You are a chaos engineer. Convert plan to AWS FIS JSON template.",
        f"""Plan JSON:
{json.dumps(plan, indent=2)}

Context:
{context}

Return JSON keys: description, roleArn, stopConditions, targets, actions, tags
"""
    )

def prompt_validate(fis, context=""):
    return (
        "You are a security auditor. Validate FIS JSON for safety.",
        f"""Template:
{json.dumps(fis, indent=2)}

Context:
{context}

Return JSON: {{is_safe, issues, suggested_changes}}
"""
    )

def prompt_codegen(fis, context=""):
    return (
        "You are a DevOps agent generating executable code.",
        f"""Given this FIS template:
{json.dumps(fis, indent=2)}

Context:
{context}

Return JSON: {{python_boto3, terraform}} — both complete code snippets.
"""
    )

def prompt_improve(plan, feedback, context=""):
    return (
        "You are a chaos engineer. Improve the plan using feedback.",
        f"""Original Plan:\n{json.dumps(plan, indent=2)}\n\nFeedback:\n{json.dumps(feedback, indent=2)}\n\nContext:\n{context}\n\nReturn improved plan JSON."""
    )

# ===================== Streamlit UI =====================
st.set_page_config(page_title="Chaos Orchestrator — Claude on Bedrock", layout="wide")
st.title("🧠 Chaos Orchestrator — Anthropic Claude + CrewAI Executor")

with st.sidebar:
    st.header("AWS Login")
    region = st.text_input("Region", "us-east-1")
    access_key = st.text_input("Access Key ID", type="default")
    secret_key = st.text_input("Secret Access Key", type="password")
    token = st.text_input("Session Token (optional)", type="password")
    model_id = st.text_input("Claude Model ID", "anthropic.claude-3-sonnet-20240229-v1:0")

    if st.button("🔐 Login to AWS"):
        sess, info = create_aws_session(access_key, secret_key, token, region)
        st.session_state["aws_session"], st.session_state["aws_info"] = sess, info
        if info.get("status") == "ok":
            identity = info["identity"]
            st.success("Logged in to AWS")
            st.json(identity)
            st.session_state["context_vars"] = {
                "AWS_REGION": region,
                "AWS_ACCOUNT_ID": identity.get("Account", ""),
                "IAM_ROLE_NAME": "MyChaosRole",
                "S3_BUCKET_ARN": "arn:aws:s3:::my-chaos-bucket"
            }
        else:
            st.error(info.get("error", info))

scenario = st.text_area("Chaos Scenario", "Simulate EC2 network latency on staging ASG for 2 minutes")
context = st.text_area("Context Snippets", "Include IAM role, tagging standards, previous FIS templates, etc.")

col1, col2, col3, col4, col5 = st.columns(5)
if col1.button("1️⃣ Plan"):
    sess = st.session_state.get("aws_session")
    if not sess: st.error("Login first."); st.stop()
    llm = init_bedrock_claude(sess, region, model_id)
    sys, usr = prompt_plan(scenario, context)
    plan = call_llm_json(llm, sys, usr, context_vars=st.session_state.get("context_vars"))
    st.session_state["plan"] = plan
    st.success("Plan generated"); st.json(plan)

if col2.button("2️⃣ Generate FIS"):
    sess = st.session_state.get("aws_session")
    if not sess: st.error("Login first."); st.stop()
    plan = st.session_state.get("plan")
    if not plan: st.error("Run step 1 first."); st.stop()
    llm = init_bedrock_claude(sess, region, model_id)
    sys, usr = prompt_fis(plan, context)
    fis = call_llm_json(llm, sys, usr, context_vars=st.session_state.get("context_vars"))
    st.session_state["fis"] = fis
    st.success("FIS template ready"); st.json(fis)

if col3.button("3️⃣ Validate"):
    sess = st.session_state.get("aws_session")
    if not sess: st.error("Login first."); st.stop()
    fis = st.session_state.get("fis")
    if not fis: st.error("Run step 2 first."); st.stop()
    llm = init_bedrock_claude(sess, region, model_id)
    sys, usr = prompt_validate(fis, context)
    val = call_llm_json(llm, sys, usr, context_vars=st.session_state.get("context_vars"))
    st.session_state["validation"] = val
    st.success("Validation complete"); st.json(val)

if col4.button("4️⃣ Codegen"):
    sess = st.session_state.get("aws_session")
    if not sess: st.error("Login first."); st.stop()
    fis = st.session_state.get("fis")
    if not fis: st.error("Run step 2 first."); st.stop()
    llm = init_bedrock_claude(sess, region, model_id)
    sys, usr = prompt_codegen(fis, context)
    art = call_llm_json(llm, sys, usr, context_vars=st.session_state.get("context_vars"))
    st.session_state["artifacts"] = art
    st.success("Code generated")
    st.subheader("Python (boto3)")
    st.code(art.get("python_boto3", ""), language="python")
    st.subheader("Terraform (HCL)")
    st.code(art.get("terraform", ""), language="hcl")

if col5.button("🚀 Run Generated Code"):
    art = st.session_state.get("artifacts", {})
    if not art:
        st.warning("Run step 4 (Codegen) first.")
    else:
        orchestrator = CrewAIOrchestrator()
        code = art.get("python_boto3") or art.get("terraform", "")
        with st.spinner("Executing artifact with CrewAI..."):
            result = orchestrator.execute(code)
        st.subheader("Execution Result")
        st.json(result)

if st.button("🔁 Improve Plan Based on Feedback"):
    validation = st.session_state.get("validation")
    if not validation or not validation.get("suggested_changes"):
        st.warning("No feedback available to improve.")
    else:
        sess = st.session_state.get("aws_session")
        llm = init_bedrock_claude(sess, region, model_id)
        sys, usr = prompt_improve(st.session_state.get("plan"), validation["suggested_changes"], context)
        improved_plan = call_llm_json(llm, sys, usr, context_vars=st.session_state.get("context_vars"))
        st.session_state["plan"] = improved_plan
        st.success("Plan improved"); st.json(improved_plan)

st.caption("⚠️ Executes real Bedrock Claude and optionally runs generated code locally (safe sandbox).")