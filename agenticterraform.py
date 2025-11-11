"""
Agentic Bedrock Claude Haiku + LangGraph Terraform Orchestrator
---------------------------------------------------------------
✅ Bedrock Claude (Haiku) generates and repairs Terraform HCL.
✅ Full init → validate → plan → repair loop until plan succeeds.
✅ Up to 50 repair attempts before failure (configurable).
✅ Auto-feedback for failed apply → LLM correction.
✅ Enforces correct AWS region & unique S3 bucket suffixes.
✅ Recursion-safe: recursion_limit=100, max_iterations=100.
✅ Optional safe auto-destroy after successful apply.
"""

import os, re, json, sys, subprocess, shutil, zipfile, stat, platform, time, datetime, random, string
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, TypedDict
import streamlit as st

# -------------------------- Utilities --------------------------

def run(cmd: List[str], cwd: Optional[str]=None, env: Optional[Dict[str,str]]=None, timeout: Optional[int]=None) -> Tuple[int,str,str]:
    proc = subprocess.Popen(cmd, cwd=cwd, env=env or os.environ.copy(),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill(); out, err = proc.communicate()
    return proc.returncode, out, err


def log_step(state: dict, step: str, rc: int, out: str, err: str, started: float, ended: float):
    state.setdefault("steps", {})[step] = {
        "rc": rc, "stdout": out, "stderr": err,
        "started": datetime.datetime.fromtimestamp(started).isoformat(timespec="seconds"),
        "ended": datetime.datetime.fromtimestamp(ended).isoformat(timespec="seconds"),
        "duration_s": round(ended - started, 3)
    }

@st.cache_data(show_spinner=False)
def cached_platform():
    return platform.system().lower(), platform.machine().lower()

# -------------------------- Terraform setup --------------------------

def ensure_python_packages(pkgs: List[str]):
    import importlib
    for pkg in pkgs:
        try: importlib.import_module(pkg)
        except Exception: subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", pkg], check=False)

def terraform_in_path() -> Optional[str]:
    for p in os.environ.get("PATH","").split(os.pathsep):
        tf = Path(p)/("terraform.exe" if os.name=="nt" else "terraform")
        if tf.exists(): return str(tf)
    return None

def ensure_terraform(target_dir: Path) -> str:
    existing = terraform_in_path()
    if existing: return existing
    target_dir.mkdir(parents=True, exist_ok=True)
    import urllib.request
    sysname, mach = cached_platform()
    arch = "arm64" if mach in ("arm64","aarch64") else "amd64"
    osid = "windows" if "windows" in sysname else ("darwin" if "darwin" in sysname else "linux")
    ver = "1.9.8"
    zip_name = f"terraform_{ver}_{osid}_{arch}.zip"
    url = f"https://releases.hashicorp.com/terraform/{ver}/{zip_name}"
    zip_path = target_dir/zip_name
    with urllib.request.urlopen(url) as r, open(zip_path,"wb") as f: shutil.copyfileobj(r,f)
    with zipfile.ZipFile(zip_path) as z: z.extractall(target_dir)
    tf_path = target_dir/("terraform.exe" if osid=="windows" else "terraform")
    tf_path.chmod(tf_path.stat().st_mode | stat.S_IEXEC)
    os.environ["PATH"] = str(target_dir)+os.pathsep+os.environ.get("PATH","")
    return str(tf_path)

# -------------------------- AWS + Bedrock --------------------------

def configure_boto3(region: str, access_key: str, secret_key: str, session_token: Optional[str]=None):
    ensure_python_packages(["boto3","botocore"])
    import boto3
    s = boto3.Session(
        aws_access_key_id=access_key or None,
        aws_secret_access_key=secret_key or None,
        aws_session_token=session_token or None,
        region_name=region,
    )
    acct = s.client("sts").get_caller_identity()
    return s, acct["Account"], acct["Arn"]

class ClaudeBedrock:
    def __init__(self, boto3_session, model: str, region: str="us-east-1"):
        self.model = model
        self.client = boto3_session.client("bedrock-runtime", region_name=region)
    def complete(self, system: str, user: str, max_tokens=6000, temperature=0.2):
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "system": system,
            "messages": [{"role":"user","content":[{"type":"text","text":user}]}],
            "max_tokens": max_tokens, "temperature": temperature, "top_p": 0.95
        }
        resp = self.client.invoke_model(modelId=self.model, body=json.dumps(body))
        payload = json.loads(resp["body"].read())
        parts = payload.get("output",{}).get("content",[]) or payload.get("content",[])
        return "".join([p.get("text","") for p in parts if p.get("type")=="text"])

# -------------------------- LangGraph setup --------------------------

ensure_python_packages(["langgraph"])
from langgraph.graph import StateGraph, END

class State(TypedDict, total=False):
    region: str; model: str; prompt: str; workdir: str; hcl: str
    steps: Dict[str,Any]; retries: int; max_retries: int; last_err: str

SYSTEM_CODEGEN = (
    "You are a senior DevOps engineer. Generate a complete Terraform configuration (main.tf) "
    "for AWS. Use provider hashicorp/aws >=5.0 with correct region and unique S3 bucket names. "
    "Return valid HCL only, no markdown."
)

SYSTEM_CRITIC = (
    "You are a Terraform repair agent. Given the main.tf and Terraform error logs, fix the configuration "
    "to make it pass validation and plan. Return full corrected HCL only."
)

CODE_FENCE_RE = re.compile(r"```(?:hcl|terraform)?\n([\s\S]*?)```", re.I)

def random_suffix(n=6): return ''.join(random.choices(string.ascii_lowercase+string.digits,k=n))

def enforce_region_and_suffix(hcl:str, region:str)->str:
    hcl = re.sub(r'region\s*=\s*".*?"', f'region = "{region}"', hcl)
    def sfx(m): return f'bucket = "{m.group(1)}-{random_suffix()}"'
    return re.sub(r'bucket\s*=\s*"([\w\.-]+)"', sfx, hcl)

def node_codegen(s: State, claude: ClaudeBedrock)->State:
    start=time.time()
    txt=claude.complete(SYSTEM_CODEGEN,f"Region: {s['region']}\nPrompt: {s['prompt']}\nReturn raw HCL only.")
    end=time.time()
    m=CODE_FENCE_RE.search(txt); hcl=(m.group(1) if m else txt).strip()
    s["hcl"]=enforce_region_and_suffix(hcl,s["region"])
    log_step(s,"codegen",0,hcl,"",start,end)
    return s

def node_write(s: State)->State:
    w=Path(s["workdir"]); w.mkdir(exist_ok=True)
    (w/"main.tf").write_text(s["hcl"])
    log_step(s,"write",0,"main.tf written","",time.time(),time.time())
    return s

def node_tf(s: State,name:str,args:List[str],env:Dict[str,str])->State:
    w=Path(s["workdir"])
    start=time.time(); rc,out,err=run(["terraform"]+args,cwd=str(w),env=env); end=time.time()
    log_step(s,name,rc,out,err,start,end)
    s["last_err"]=err
    return s

def node_critic(s: State, claude: ClaudeBedrock, step:str)->State:
    hcl=s.get("hcl","")
    err=(s.get("steps",{}).get(step,{}).get("stderr") or "")+(s.get("steps",{}).get(step,{}).get("stdout") or "")
    user=f"Current main.tf:\n{hcl}\n\nTerraform {step} errors:\n{err}\nReturn corrected full HCL."
    start=time.time()
    fix=claude.complete(SYSTEM_CRITIC,user)
    end=time.time()
    m=CODE_FENCE_RE.search(fix); new=(m.group(1) if m else fix).strip()
    s["hcl"]=enforce_region_and_suffix(new,s["region"])
    log_step(s,f"critic_{step}",0,new,"",start,end)
    return s

# -------------------------- Streamlit UI --------------------------

st.set_page_config(page_title="Agentic Terraform Orchestrator (50 attempts)", layout="wide")
st.title("🤖 Agentic Bedrock Claude Haiku + LangGraph Terraform (50 attempts)")

with st.sidebar:
    region = st.text_input("AWS Region", os.environ.get("AWS_REGION","us-east-1"))
    access = st.text_input("Access Key ID", type="password")
    secret = st.text_input("Secret Access Key", type="password")
    token = st.text_input("Session Token (optional)", type="password")
    model = "anthropic.claude-3-haiku-20240307-v1:0"
    auto_destroy = st.toggle("Auto-destroy after apply", value=False)
    validate = st.button("Validate AWS")
    ensure = st.button("Ensure Terraform")

if ensure:
    with st.spinner("Installing Terraform..."):
        tfp=ensure_terraform(Path.home()/".tfbin")
        st.success(f"Terraform ready: {tfp}")

if validate or "aws_session" not in st.session_state:
    try:
        sess,acct,arn=configure_boto3(region,access,secret,token or None)
        st.session_state.aws={"region":region,"access_key":access,"secret_key":secret,"session_token":token}
        st.session_state.aws_session=sess
        st.success(f"AWS validated — {acct}, {arn}")
    except Exception as e:
        st.error(str(e))

prompt=st.text_area("Describe your infrastructure (natural language):",height=140)
run_btn=st.button("Run Agentic Workflow")
destroy_btn=st.button("Destroy")

if "workdir" not in st.session_state:
    st.session_state.workdir=Path.cwd()/ "tf_agentic_50"
workdir=st.session_state.workdir; workdir.mkdir(exist_ok=True)

def build_env()->Dict[str,str]:
    env=os.environ.copy()
    aws=st.session_state.get("aws",{})
    env.update({
        "AWS_ACCESS_KEY_ID":aws.get("access_key",""),
        "AWS_SECRET_ACCESS_KEY":aws.get("secret_key",""),
        "AWS_SESSION_TOKEN":aws.get("session_token",""),
        "AWS_DEFAULT_REGION":aws.get("region",region),
    })
    return env

def render_steps(s:State):
    for n,v in (s.get("steps") or {}).items():
        with st.expander(f"{n.upper()} rc={v['rc']} dur={v['duration_s']}s",expanded=n in ["plan","apply"]):
            st.text_area(f"{n} stdout",v.get("stdout",""),height=100)
            if v.get("stderr"): st.text_area(f"{n} stderr",v["stderr"],height=80)

# -------------------------- Run Workflow --------------------------

if run_btn:
    if not prompt.strip(): st.warning("Enter a prompt."); st.stop()
    if "aws_session" not in st.session_state: st.warning("Validate AWS first."); st.stop()
    ensure_terraform(Path.home()/".tfbin")

    from langgraph.graph import StateGraph, END
    graph=StateGraph(State)
    claude=ClaudeBedrock(st.session_state.aws_session,model=model,region=region)

    def _codegen(s): return node_codegen(s,claude)
    def _write(s): return node_write(s)
    def _init(s): return node_tf(s,"init",["init"],build_env())
    def _validate(s): return node_tf(s,"validate",["validate"],build_env())
    def _plan(s): return node_tf(s,"plan",["plan","-out","plan.tfplan"],build_env())
    def _apply(s): return node_tf(s,"apply",["apply","-auto-approve","plan.tfplan"],build_env())
    def _output(s): return node_tf(s,"output",["output","-json"],build_env())
    def _critic_plan(s): return node_critic(s,claude,"plan")
    def _critic_apply(s): return node_critic(s,claude,"apply")

    for n,f in {
        "codegen":_codegen,"write":_write,"init":_init,"validate":_validate,
        "plan":_plan,"critic_plan":_critic_plan,"apply":_apply,"critic_apply":_critic_apply,"output":_output
    }.items(): graph.add_node(n,f)

    graph.set_entry_point("codegen")
    graph.add_edge("codegen","write"); graph.add_edge("write","init")
    graph.add_edge("init","validate"); graph.add_edge("validate","plan")

    def after_plan(s:State):
        rc=s.get("steps",{}).get("plan",{}).get("rc",1)
        tries=s.get("retries",0)
        if rc!=0 and tries<s.get("max_retries",50):
            s["retries"]=tries+1
            return "critic_plan"
        return "apply" if rc==0 else END
    graph.add_conditional_edges("plan",after_plan,{"critic_plan":"critic_plan","apply":"apply",END:END})
    graph.add_edge("critic_plan","write")

    def after_apply(s:State):
        rc=s.get("steps",{}).get("apply",{}).get("rc",1)
        tries=s.get("retries",0)
        if rc!=0 and tries<s.get("max_retries",50):
            s["retries"]=tries+1
            return "critic_apply"
        return "output" if rc==0 else END
    graph.add_conditional_edges("apply",after_apply,{"critic_apply":"critic_apply","output":"output",END:END})
    graph.add_edge("critic_apply","write")

    app=graph.compile()
    app.recursion_limit=1000  # Prevent overflow

    state:State={"region":region,"model":model,"prompt":prompt,"workdir":str(workdir),
                 "steps":{},"retries":0,"max_retries":50,"last_err":""}

    with st.spinner("Running agentic workflow (up to 50 repairs)..."):
        final=app.invoke(state,max_iterations=100)

    st.subheader("Generated Terraform (main.tf)")
    st.code(final.get("hcl",""),language="hcl")
    st.subheader("Workflow Logs")
    render_steps(final)

    plan_rc=final.get("steps",{}).get("plan",{}).get("rc",1)
    apply_rc=final.get("steps",{}).get("apply",{}).get("rc",1)
    if plan_rc!=0:
        st.error("❌ Plan did not succeed after 50 repair attempts.")
    elif apply_rc==0:
        st.success("✅ Apply succeeded.")
        if auto_destroy:
            with st.spinner("Auto-destroying resources..."):
                rc,out,err=run(["terraform","destroy","-auto-approve"],cwd=str(workdir),env=build_env())
                st.text_area("auto-destroy output",out+err,height=200)
    else:
        st.error("❌ Apply failed after 50 repairs.")

    st.download_button("Download logs (JSON)",data=json.dumps(final.get("steps",{}),indent=2),
                       file_name="tf_agentic_50_steps.json")

# -------------------------- Manual Destroy --------------------------

if destroy_btn:
    rc,out,err=run(["terraform","destroy","-auto-approve"],cwd=str(workdir),env=build_env())
    st.text_area("destroy output",out+err,height=200)
    st.success("Destroy completed." if rc==0 else "Destroy failed.")
