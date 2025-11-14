"""
Terraform Orchestrator — LangGraph + Claude (Bedrock) with IAM Preflight & Recursive Self-Heal
-----------------------------------------------------------------------------------------------
✓ LangGraph graph: codegen → init → validate → plan ⇄ repair → IAM preflight → apply ⇄ repair → verify → output
✓ Recursively fixes Terraform until PLAN succeeds (configurable attempts), then APPLY (configurable attempts)
✓ IAM-aware: Claude inspects config and injects missing IAM roles/policies before apply
✓ Robust JSON parsing of Claude output; safe Terraform path; cross-platform
✓ Live progress + per-step logs in Streamlit
"""

import os, re, json, sys, subprocess, shutil, zipfile, stat, time, datetime, platform
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import streamlit as st

# -------------------------------- Safety --------------------------------
sys.setrecursionlimit(5000)

# ============================== Utilities ==============================

def ensure_str(x) -> str:
    if isinstance(x, str): return x
    try: return json.dumps(x, indent=2, default=str)
    except Exception: return str(x)

def run(cmd: List[str], cwd: Optional[str]=None, env: Optional[Dict[str,str]]=None, timeout: Optional[int]=None) -> Tuple[int,str,str]:
    proc = subprocess.Popen(cmd, cwd=cwd, env=env or os.environ.copy(),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try: out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill(); out, err = proc.communicate()
    return proc.returncode, out, err

def log_step(state: dict, step: str, rc: int, out: Any, err: Any, t0: float, t1: float):
    state.setdefault("steps", {})[step] = {
        "rc": rc,
        "stdout": ensure_str(out),
        "stderr": ensure_str(err),
        "started": datetime.datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
        "ended": datetime.datetime.fromtimestamp(t1).isoformat(timespec="seconds"),
        "duration_s": round(t1 - t0, 3),
    }

def ensure_python_packages(pkgs: List[str]):
    import importlib
    for p in pkgs:
        try:
            importlib.import_module(p)
        except Exception:
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", p], check=False)

# ============================== Terraform ==============================

def terraform_in_path() -> Optional[str]:
    for p in os.environ.get("PATH", "").split(os.pathsep):
        tf = Path(p) / ("terraform.exe" if os.name == "nt" else "terraform")
        if tf.exists() and tf.is_file(): return str(tf)
    return None

def ensure_terraform(target_dir: Path) -> str:
    existing = terraform_in_path()
    if existing: return existing
    target_dir.mkdir(parents=True, exist_ok=True)
    import urllib.request
    sysname, mach = platform.system().lower(), platform.machine().lower()
    arch = "arm64" if mach in ("arm64", "aarch64") else "amd64"
    osid = "windows" if "windows" in sysname else ("darwin" if "darwin" in sysname else "linux")
    ver = "1.9.8"
    zip_name = f"terraform_{ver}_{osid}_{arch}.zip"
    url = f"https://releases.hashicorp.com/terraform/{ver}/{zip_name}"
    zip_path = target_dir / zip_name
    with urllib.request.urlopen(url) as r, open(zip_path, "wb") as f: shutil.copyfileobj(r, f)
    with zipfile.ZipFile(zip_path) as z: z.extractall(target_dir)
    if os.name == "nt":
        tf_bin = target_dir / "terraform.exe"
        if not tf_bin.exists() and (target_dir / "terraform").exists():
            (target_dir / "terraform").rename(tf_bin)
    else:
        tf_bin = target_dir / "terraform"
    try: tf_bin.chmod(tf_bin.stat().st_mode | stat.S_IEXEC)
    except Exception: pass
    os.environ["PATH"] = str(target_dir) + os.pathsep + os.environ.get("PATH", "")
    return str(tf_bin)

def terraform_cmd(args: List[str], cwd: Path, env: Optional[Dict[str,str]]=None) -> Tuple[int,str,str]:
    tf_path = terraform_in_path() or ensure_terraform(Path.home() / ".tfbin")
    tf_bin = Path(tf_path)
    if not tf_bin.exists(): raise FileNotFoundError(f"Terraform binary not found at {tf_bin}")
    if not cwd.exists(): raise FileNotFoundError(f"Working directory missing: {cwd}")
    if not list(cwd.glob("*.tf")): raise FileNotFoundError(f"No Terraform files found in {cwd}.")
    # Clean stale lock automatically
    lock = cwd / ".terraform.tfstate.lock.info"
    if lock.exists():
        try: lock.unlink()
        except Exception: pass
    return run([str(tf_bin)] + args, cwd=str(cwd), env=env)

# ============================== AWS + Bedrock Claude ==============================

def configure_boto3(region: str, access: str, secret: str, token: Optional[str]=None):
    ensure_python_packages(["boto3", "botocore"])
    import boto3
    s = boto3.Session(aws_access_key_id=access or None,
                      aws_secret_access_key=secret or None,
                      aws_session_token=token or None,
                      region_name=region)
    ident = s.client("sts").get_caller_identity()
    return s, ident["Account"], ident["Arn"]

class ClaudeBedrock:
    def __init__(self, boto_sess, model: str, region: str="us-east-1"):
        self.model = model
        self.client = boto_sess.client("bedrock-runtime", region_name=region)
    def complete(self, system: str, user: str, max_tokens=6000, temp=0.2) -> str:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "system": system,
            "messages": [{"role": "user", "content": [{"type": "text", "text": user}]}],
            "max_tokens": max_tokens, "temperature": temp, "top_p": 0.95
        }
        resp = self.client.invoke_model(modelId=self.model, body=json.dumps(body))
        payload = json.loads(resp["body"].read())
        parts = payload.get("output", {}).get("content", []) or payload.get("content", [])
        return "".join([p.get("text", "") for p in parts if p.get("type") == "text"])

# ============================== JSON Extract ==============================

def extract_json(txt: str) -> Optional[Dict[str, str]]:
    if not txt: return None
    t = txt.strip()
    t = re.sub(r"^```(?:json|hcl|terraform)?", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"```$", "", t).strip()
    braces = [m.start() for m in re.finditer(r"[{}]", t)]
    if braces: t = t[braces[0]:braces[-1] + 1]
    t = re.sub(r",\s*([}\]])", r"\1", t)
    if t.count("{") > t.count("}"): t += "}" * (t.count("{") - t.count("}"))
    try:
        j = json.loads(t)
        return j if isinstance(j, dict) else None
    except Exception:
        m = re.search(r"\{[\s\S]*\}", t)
        if m:
            try: return json.loads(m.group(0))
            except Exception: return None
    return None

# ============================== Prompts ==============================

SYSTEM_PROMPT = r"""
You are a Terraform architect and DevOps expert.
Generate Terraform project files as a JSON object mapping filenames to file contents.

REQUIREMENTS (STRICT):
- Output ONLY JSON (no markdown, no prose).
- Must include at least one AWS resource (e.g., aws_s3_bucket, aws_instance, aws_vpc, aws_cloudfront_distribution).
- Use correct random provider for uniqueness:
    provider "random" {}
    resource "random_string" "suffix" {
      length  = 6
      upper   = false
      special = false
    }
    Reference as: ${random_string.suffix.result}
- Always include (root): main.tf, providers.tf, variables.tf, outputs.tf
- Use sane defaults (EC2 t3.micro, S3 AES256, VPC 10.0.0.0/16).
- Return valid JSON ready to write directly to disk.
"""

IAM_PROMPT = r"""
You are an AWS IAM + Terraform expert.
Given the current Terraform files (provided as JSON mapping path→content), infer the IAM actions required for a successful apply.
If the configuration lacks necessary IAM roles/policies/attachments, MODIFY the Terraform to include them
(aws_iam_role, aws_iam_policy, aws_iam_role_policy_attachment, instance profiles, iam:PassRole, cloudfront permissions, etc.).
Return ONLY the corrected files as JSON (same mapping path→content). No prose.
"""

# ============================== Streamlit UI ==============================

st.set_page_config(page_title="Terraform Orchestrator — LangGraph Self-Heal + IAM", layout="wide")
st.title("🧠 Terraform Orchestrator — LangGraph • Self-Heal • IAM Preflight")

with st.sidebar:
    region = st.text_input("AWS Region", "us-east-1")
    access = st.text_input("Access Key ID", type="password")
    secret = st.text_input("Secret Access Key", type="password")
    token  = st.text_input("Session Token (optional)", type="password")
    model  = "anthropic.claude-3-haiku-20240307-v1:0"
    max_plan_attempts  = st.number_input("Max PLAN attempts", 1, 100, 50, 1)
    max_apply_attempts = st.number_input("Max APPLY attempts", 1, 20, 10, 1)
    validate = st.button("Validate AWS")
    ensure   = st.button("Ensure Terraform")

if ensure:
    with st.spinner("Ensuring Terraform..."):
        tfp = ensure_terraform(Path.home()/".tfbin")
        st.success(f"Terraform ready: {tfp}")

if validate or "aws_session" not in st.session_state:
    try:
        sess, acct, arn = configure_boto3(region, access, secret, token or None)
        st.session_state.aws_session = sess
        st.session_state.aws = {"region": region, "access": access, "secret": secret, "token": token}
        st.success(f"AWS validated — {acct} | {arn}")
    except Exception as e:
        st.error(str(e))

prompt = st.text_area("Describe infrastructure (LangGraph will iterate until plan succeeds, then apply)", height=160)
run_btn = st.button("Run — LangGraph Orchestrator")
destroy_btn = st.button("Destroy")

workdir = Path.cwd()/ "tf_langgraph_selfheal"
workdir.mkdir(parents=True, exist_ok=True)

progress = st.progress(0)
status   = st.empty()
logs     = st.empty()
claude_box = st.expander("Claude raw JSON (latest)", expanded=False)

def build_env() -> Dict[str, str]:
    env = os.environ.copy()
    aws = st.session_state.get("aws", {})
    env.update({
        "AWS_ACCESS_KEY_ID": aws.get("access", ""),
        "AWS_SECRET_ACCESS_KEY": aws.get("secret", ""),
        "AWS_SESSION_TOKEN": aws.get("token", ""),
        "AWS_DEFAULT_REGION": aws.get("region", region),
    })
    return env

def render_logs(steps: Dict[str,Any]):
    out = []
    for name, meta in (steps or {}).items():
        out.append(f"### {name.upper()} — rc={meta.get('rc')} — {meta.get('duration_s')}s")
        so = (meta.get("stdout") or "").strip()
        se = (meta.get("stderr") or "").strip()
        if so: out.append(f"```\n{so}\n```")
        if se: out.append(f"```\n{se}\n```")
    logs.markdown("\n".join(out))

def write_files(files: Dict[str,str]) -> List[str]:
    wrote = []
    for rel, content in files.items():
        p = workdir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(ensure_str(content))
        wrote.append(p.relative_to(workdir).as_posix())
    # Ensure random provider if uniqueness hinted
    prov = workdir/"providers.tf"
    if prov.exists():
        txt = prov.read_text()
        if 'provider "random"' not in txt:
            prov.write_text(txt + '\nprovider "random" {}')
    return wrote

# ============================== LangGraph ==============================

ensure_python_packages(["langgraph"])
ensure_terraform(Path.home()/".tfbin")
from langgraph.graph import StateGraph, END
from langgraph.types import RunnableConfig

graph = StateGraph(dict)
claude = ClaudeBedrock(st.session_state.aws_session, model=model, region=region)

def push(label: str, pct: float, raw: Optional[str]=None):
    progress.progress(min(100, max(0, int(pct*100))))
    status.write(f"**{label} — {int(pct*100)}%**")
    if raw:
        with claude_box:
            st.code(raw[:5000] + ("..." if len(raw) > 5000 else ""))

# ---------------- Nodes ----------------

def node_codegen(s: Dict[str,Any]):
    t0 = time.time()
    user = f"Region: {region}\nPrompt: {prompt}\nReturn only JSON mapping filename→content."
    raw = claude.complete(SYSTEM_PROMPT, user)
    s["last_claude"] = raw
    push("Claude codegen", 0.10, raw)
    js = extract_json(raw)
    if not js:
        raise RuntimeError("Claude did not return valid JSON.")
    wrote = write_files(js)
    main_tf = workdir/"main.tf"
    if not main_tf.exists() or "resource" not in main_tf.read_text():
        raise RuntimeError("No resources in main.tf")
    log_step(s, "codegen", 0, wrote, "", t0, time.time()); render_logs(s.get("steps", {}))
    s["plan_attempts"]  = 0
    s["apply_attempts"] = 0
    s["max_plan"]  = int(max_plan_attempts)
    s["max_apply"] = int(max_apply_attempts)
    return s

def node_init(s: Dict[str,Any]):
    t0 = time.time(); rc,out,err = terraform_cmd(["init","-upgrade"], workdir, build_env())
    log_step(s, f"init_{s.get('plan_attempts',0)}", rc, out, err, t0, time.time()); render_logs(s.get("steps", {}))
    s["init_rc"] = rc
    return s

def node_validate(s: Dict[str,Any]):
    t0 = time.time(); rc,out,err = terraform_cmd(["validate"], workdir, build_env())
    log_step(s, f"validate_{s.get('plan_attempts',0)}", rc, out, err, t0, time.time()); render_logs(s.get("steps", {}))
    s["validate_rc"] = rc; s["validate_err"] = err
    return s

def node_plan(s: Dict[str,Any]):
    t0 = time.time(); rc,out,err = terraform_cmd(["plan","-out","plan.tfplan"], workdir, build_env())
    log_step(s, f"plan_{s.get('plan_attempts',0)}", rc, out, err, t0, time.time()); render_logs(s.get("steps", {}))
    s["plan_rc"] = rc; s["plan_err"] = err
    return s

def node_repair_plan(s: Dict[str,Any]):
    s["plan_attempts"] = int(s.get("plan_attempts",0)) + 1
    attempt = s["plan_attempts"]
    push(f"Plan/Validate failed — Claude repairing (attempt {attempt}/{s['max_plan']})", 0.25 + min(0.5, attempt/max(1,s['max_plan']))*0.5)
    files_json = {p.relative_to(workdir).as_posix(): p.read_text() for p in workdir.rglob("*.tf")}
    fb = (
        "Fix this Terraform PLAN/VALIDATE error. Return ONLY corrected files as JSON.\n\n" +
        f"ERROR:\n{s.get('plan_err') or s.get('validate_err','')}\n\n" +
        "Current files (path->content) JSON follows:\n" + json.dumps(files_json)
    )
    raw = claude.complete(SYSTEM_PROMPT, fb)
    js = extract_json(raw)
    if js:
        write_files(js)
        s["last_claude_repair_plan"] = raw
    return s

def node_iam_preflight(s: Dict[str,Any]):
    push("IAM preflight (Claude)", 0.80)
    files_json = {p.relative_to(workdir).as_posix(): p.read_text() for p in workdir.rglob("*.tf")}
    raw = claude.complete(IAM_PROMPT, json.dumps(files_json))
    js = extract_json(raw)
    if js:
        write_files(js)
        s["last_claude_iam"] = raw
    # re-init & validate in case IAM changed
    t0 = time.time(); rc,out,err = terraform_cmd(["init","-upgrade"], workdir, build_env())
    log_step(s, "init_after_iam", rc, out, err, t0, time.time()); render_logs(s.get("steps", {}))
    t0 = time.time(); rc,out,err = terraform_cmd(["validate"], workdir, build_env())
    log_step(s, "validate_after_iam", rc, out, err, t0, time.time()); render_logs(s.get("steps", {}))
    return s

def node_apply(s: Dict[str,Any]):
    t0 = time.time(); rc,out,err = terraform_cmd(["apply","-auto-approve","plan.tfplan"], workdir, build_env())
    log_step(s, f"apply_{s.get('apply_attempts',0)}", rc, out, err, t0, time.time()); render_logs(s.get("steps", {}))
    s["apply_rc"] = rc; s["apply_err"] = err
    return s

def node_repair_apply(s: Dict[str,Any]):
    s["apply_attempts"] = int(s.get("apply_attempts",0)) + 1
    attempt = s["apply_attempts"]
    push(f"Apply failed — Claude repairing IAM/resources (attempt {attempt}/{s['max_apply']})", 0.90)
    files_json = {p.relative_to(workdir).as_posix(): p.read_text() for p in workdir.rglob("*.tf")}
    fb = (
        "Fix this Terraform APPLY error by adjusting IAM roles/policies/resources. Return ONLY corrected files as JSON.\n\n" +
        f"ERROR:\n{s.get('apply_err','')}\n\n" +
        "Current files (path->content) JSON follows:\n" + json.dumps(files_json)
    )
    raw = claude.complete(IAM_PROMPT + "\n\nAlso reconcile resource dependencies.", fb)
    js = extract_json(raw)
    if js:
        write_files(js)
        s["last_claude_apply"] = raw
        # Re-plan after changes
        t0 = time.time(); rc,out,err = terraform_cmd(["plan","-out","plan.tfplan"], workdir, build_env())
        log_step(s, f"plan_after_apply_fix_{attempt}", rc, out, err, t0, time.time()); render_logs(s.get("steps", {}))
    return s

def node_verify(s: Dict[str,Any]):
    t0 = time.time(); rc,out,err = terraform_cmd(["show"], workdir, build_env())
    log_step(s, "verify_show", rc, out, err, t0, time.time()); render_logs(s.get("steps", {}))
    return s

def node_output(s: Dict[str,Any]):
    t0 = time.time(); rc,out,err = terraform_cmd(["output","-json"], workdir, build_env())
    log_step(s, "output_json", rc, out, err, t0, time.time()); render_logs(s.get("steps", {}))
    return s

# ---------------- Graph topology ----------------

graph.add_node("codegen", node_codegen)
graph.add_node("init", node_init)
graph.add_node("validate", node_validate)
graph.add_node("plan", node_plan)
graph.add_node("repair_plan", node_repair_plan)
graph.add_node("iam_preflight", node_iam_preflight)
graph.add_node("apply", node_apply)
graph.add_node("repair_apply", node_repair_apply)
graph.add_node("verify", node_verify)
graph.add_node("output", node_output)

graph.set_entry_point("codegen")
graph.add_edge("codegen", "init")
graph.add_edge("init", "validate")

# Conditional edges for validate → plan/repair/end
def route_after_validate(s: Dict[str,Any]) -> str:
    if s.get("validate_rc", 1) == 0:
        return "plan"
    # failed validate → repair if attempts left
    if int(s.get("plan_attempts",0)) < int(s.get("max_plan",50)):
        return "repair_plan"
    return "__end__"

graph.add_conditional_edges("validate", route_after_validate, {
    "plan": "plan",
    "repair_plan": "repair_plan",
    "__end__": END
})

# After repair_plan → init
graph.add_edge("repair_plan", "init")

# Conditional edges for plan → iam_preflight/repair/end
def route_after_plan(s: Dict[str,Any]) -> str:
    if s.get("plan_rc", 1) == 0:
        return "iam_preflight"
    if int(s.get("plan_attempts",0)) < int(s.get("max_plan",50)):
        return "repair_plan"
    return "__end__"

graph.add_conditional_edges("plan", route_after_plan, {
    "iam_preflight": "iam_preflight",
    "repair_plan": "repair_plan",
    "__end__": END
})

# After iam_preflight → apply
graph.add_edge("iam_preflight", "apply")

# Conditional edges for apply → verify/repair/end
def route_after_apply(s: Dict[str,Any]) -> str:
    if s.get("apply_rc", 1) == 0:
        return "verify"
    if int(s.get("apply_attempts",0)) < int(s.get("max_apply",10)):
        return "repair_apply"
    return "__end__"

graph.add_conditional_edges("apply", route_after_apply, {
    "verify": "verify",
    "repair_apply": "repair_apply",
    "__end__": END
})

# After repair_apply → init (re-init, validate, plan, apply again)
graph.add_edge("repair_apply", "init")

# After verify → output → END
graph.add_edge("verify", "output")

app = graph.compile()
config = RunnableConfig(recursion_limit=5000, max_iterations=5000)

# ============================== Run Button ==============================

if run_btn:
    if not prompt.strip():
        st.warning("Enter a prompt."); st.stop()

    with st.spinner("Running LangGraph orchestrator..."):
        state: Dict[str, Any] = {"steps": {}}
        final = app.invoke(state, config=config)

    st.success("Workflow complete (see logs below).")
    render_logs(final.get("steps", {}))

    st.subheader("Generated Terraform Files")
    for p in sorted(workdir.rglob("*.tf")):
        st.markdown(f"**{p.relative_to(workdir)}**")
        st.code(p.read_text(), language="hcl")

# ============================== Destroy ==============================

if destroy_btn:
    try:
        rc,out,err = terraform_cmd(["destroy","-auto-approve"], workdir, build_env())
        st.text_area("Destroy output", ensure_str(out+err), height=240)
        st.success("Destroy completed." if rc == 0 else "Destroy failed.")
    except Exception as e:
        st.error(str(e))
