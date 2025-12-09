
import time
import json
import shutil
import re
import html
from pathlib import Path
from difflib import unified_diff
from core.utils import log_step
from core.terraform import terraform_cmd
from core.prompts import SYSTEM_PROMPT

# --- Safe JSON Parsing ---
def safe_json_parse(raw: str):
    if not raw:
        return {}
    raw = re.sub(r"^\s*```(json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```\s*$", "", raw)
    raw = html.unescape(raw)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    cleaned = re.sub(r"[\x00-\x1F\x7F]", "", match.group(0))
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}

# --- Deduplicate Providers ---
def deduplicate_providers_in_dir(workdir: Path):
    seen_providers = set()
    for p in sorted(workdir.rglob("*.tf")):
        lines = p.read_text().splitlines()
        result, buffer = [], []
        inside_provider = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('provider'):
                inside_provider = True
                buffer = [line]
                continue
            if inside_provider:
                buffer.append(line)
                if stripped == '}':
                    try:
                        provider_name = buffer[0].split('"')[1]
                    except Exception:
                        provider_name = None
                    if provider_name and provider_name not in seen_providers:
                        seen_providers.add(provider_name)
                        result.extend(buffer)
                    buffer, inside_provider = [], False
                continue
            result.append(line)
        p.write_text("\n".join(result))

# --- Ensure Required Providers ---
def ensure_required_providers(workdir: Path):
    main_tf = workdir / "main.tf"
    if not main_tf.exists():
        main_tf.write_text("")
    content = main_tf.read_text()
    needs_req = "required_providers" not in content
    needs_provider = 'provider "aws"' not in content
    block_parts = []
    if needs_req:
        block_parts.append("""terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
}""")
    if needs_provider:
        block_parts.append('''provider "aws" {
  region = var.region
}''')
    if block_parts:
        main_tf.write_text(content + ("\n" if content and not content.endswith("\n") else "") + "\n\n".join(block_parts))
    variables_tf = workdir / "variables.tf"
    vars_content = variables_tf.read_text() if variables_tf.exists() else ""
    if 'variable "region"' not in vars_content:
        vars_block = '''variable "region" {
  description = "AWS region"
  type        = string
}'''
        variables_tf.write_text(vars_content + ("\n" if vars_content and not vars_content.endswith("\n") else "") + vars_block)

# --- Cleanup Terraform State ---
def cleanup_terraform(workdir: Path):
    for item in [workdir / ".terraform", workdir / ".terraform.lock.hcl"]:
        if item.exists():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

# --- Report File Changes ---
def report_file_changes(workdir: Path, new_files: dict):
    feedback = []
    for fname, new_content in new_files.items():
        file_path = workdir / fname
        old_content = file_path.read_text() if file_path.exists() else ""
        if old_content != new_content:
            diff = "\n".join(unified_diff(old_content.splitlines(), new_content.splitlines(), fromfile=fname, tofile=fname))
            feedback.append(f"Updated {fname}:\n{diff if diff else '[No diff available]'}")
        file_path.write_text(new_content)
    return feedback

# --- Retry Logic ---
def retry_claude(claude, prompt, files_json):
    raw_retry = claude.complete(
        "STRICT MODE: Fix Terraform error and return ONLY valid JSON with filenames as keys and file content as values. No extra text.",
        json.dumps(files_json)
    )
    return safe_json_parse(raw_retry)

# --- Handle Terraform Errors ---

def handle_terraform_error(s, claude, workdir, error_msg, step_name):
    """
    Centralized repair handler for Terraform errors.

    Behaviors:
      - IAM/STS/Unauthorized => non-recoverable (no LLM attempt)
      - No configuration files => advise codegen (no LLM attempt)
      - Lock/state inconsistencies => clean artifacts, allow a clean re-init on next step
      - Provider registry/network/DNS errors => short-circuit with actionable guidance (no LLM attempt)
      - Otherwise => attempt LLM-driven repair; HTML-unescape fixed files; dedupe providers; cleanup for clean re-init
    """
    msg_raw = error_msg or ""
    msg = msg_raw.strip()
    msg_lc = msg.lower()

    # 1) Non-recoverable IAM/STS/Unauthorized categories
    iam_keywords = [
        "accessdenied", "unauthorizedoperation", "iam", "sts",
        "accessdeniedexception", "expired token", "not authorized", "signaturedoesnotmatch"
    ]
    if any(k in msg_lc for k in iam_keywords):
        s["steps"][f"repair_{step_name}"] = {
            "stdout": f"Non-recoverable IAM/authorization error detected. "
                      f"Please verify credentials/roles and retry.\n\nOriginal error:\n{msg_raw}"
        }
        return s

    # 2) No configs present => cannot repair files; requires codegen
    if not list(workdir.rglob("*.tf")) or "no configuration files" in msg_lc:
        s["steps"][f"repair_{step_name}"] = {
            "stdout": f"No Terraform configuration files present. "
                      f"Please rerun codegen before {step_name}.\n\nOriginal error:\n{msg_raw}"
        }
        return s

    # 3) Lock/state inconsistencies => cleanup local artifacts
    if ("lock file" in msg_lc) or ("state file" in msg_lc) or ("state lock" in msg_lc):
        cleanup_terraform(workdir)
        s["steps"][f"repair_{step_name}"] = {
            "stdout": "Detected inconsistent lock/state. Local Terraform artifacts were cleaned. "
                      "Re-run the step to initialize a clean working directory."
        }
        return s

    # 4) Provider registry/network/DNS/proxy connectivity problems => short-circuit with guidance
    network_hints = [
        "could not connect to registry.terraform.io",
        "failed to query available provider packages",
        "request discovery document",
        "dial tcp", "connection timed out", "connection refused",
        "lookup registry.terraform.io", "temporary failure in name resolution",
        "getaddrinfo", "getaddrinfow", "proxy", "tls handshake timeout"
    ]
    if any(h in msg_lc for h in network_hints):
        s["steps"][f"repair_{step_name}"] = {
            "stdout": (
                "Network/DNS connectivity issue detected while contacting the Terraform provider registry.\n"
                "- Ensure outbound internet access or configure HTTP/HTTPS proxy.\n"
                "- If in a restricted network, use a local provider mirror via 'provider_installation'.\n"
                "- After fixing connectivity, run: terraform init\n\n"
                f"Original error:\n{msg_raw}"
            )
        }
        return s

    # 5) Attempt LLM-driven repair for syntax/config issues
    files_json = {str(p.relative_to(workdir)): p.read_text() for p in workdir.rglob("*.tf")}
    prompt = f"""
The following Terraform error occurred during {step_name}:
{msg_raw}

Fix the issue in the provided files. Return ONLY valid JSON with filenames as keys and corrected file content as values.
"""
    raw = claude.complete(prompt, json.dumps(files_json))
    fixes = safe_json_parse(raw)
    if not fixes:
        fixes = retry_claude(claude, prompt, files_json)
    if not fixes:
        s["steps"][f"repair_{step_name}"] = {
            "stdout": f"Claude could not provide valid JSON fixes after retry.\nOriginal error:\n{msg_raw}"
        }
        return s

    # Unescape HTML entities in repaired files (handles <<EOF, ~>, etc.)
    for fname in list(fixes.keys()):
        fixes[fname] = html.unescape(fixes[fname])

    changes = report_file_changes(workdir, fixes)
    ensure_required_providers(workdir)
    deduplicate_providers_in_dir(workdir)

    # Clean local artifacts so next init/plan starts fresh
    cleanup_terraform(workdir)

    s["steps"][f"repair_{step_name}"] = {
        "stdout": "Error repair applied. Changes:\n" + ("\n".join(changes) if changes else "[No textual diff]")
    }
    return s


# --- Nodes ---
def node_codegen(s, claude, workdir, prompt):
    raw = claude.complete(SYSTEM_PROMPT, prompt)
    files = safe_json_parse(raw) or retry_claude(claude, SYSTEM_PROMPT, {"prompt": prompt})
    if not files:
        s["steps"]["codegen"] = {"stdout": "Invalid JSON from Claude after retry, skipping codegen."}
        return s
    # ✅ Unescape HTML entities in generated files
    for fname in files:
        files[fname] = html.unescape(files[fname])
    changes = report_file_changes(workdir, files)
    ensure_required_providers(workdir)
    deduplicate_providers_in_dir(workdir)
    s["steps"]["codegen"] = {"stdout": f"Files generated. Changes:\n{'\n'.join(changes)}"}
    return s

def node_init(s, workdir, env):
    t0 = time.time()
    if not list(workdir.rglob("*.tf")):
        s["steps"]["init"] = {"stdout": "No Terraform files found. Skipping init."}
        return s
    rc, out, err = terraform_cmd(["init", "-input=false"], workdir, env)
    if rc != 0 and "lock file" in ((err or "") + (out or "")):
        cleanup_terraform(workdir)
        rc, out, err = terraform_cmd(["init", "-upgrade", "-input=false"], workdir, env)
    log_step(s, "init", rc, out, err, t0, time.time())
    return s

def node_validate(s, claude, workdir, env):
    t0 = time.time()
    rc, out, err = terraform_cmd(["validate"], workdir, env)
    log_step(s, "validate", rc, out, err, t0, time.time())
    if rc != 0:
        return handle_terraform_error(s, claude, workdir, err or out, step_name="validate")
    return s

def node_plan(s, claude, workdir, env):
    t0 = time.time()
    max_retries = 3
    attempt = 0
    while attempt < max_retries:
        rc, out, err = terraform_cmd(["plan", "-out", "plan.tfplan", "-input=false"], workdir, env)
        log_step(s, f"plan_attempt_{attempt+1}", rc, out, err, t0, time.time())
        if rc == 0:
            s["steps"]["plan"] = {"stdout": f"Plan succeeded on attempt {attempt+1}"}
            return s
        s["steps"][f"plan_fix_{attempt+1}"] = {"stdout": f"Plan failed. Attempting fix..."}
        s = handle_terraform_error(s, claude, workdir, err or out, step_name=f"plan_attempt_{attempt+1}")
        attempt += 1
    s["steps"]["plan"] = {"stdout": f"Plan failed after {max_retries} attempts. Manual intervention required."}
    return s

def node_apply(s, claude, workdir, env):
    t0 = time.time()
    plan_file = workdir / "plan.tfplan"
    if not plan_file.exists():
        s["steps"]["apply"] = {"stdout": "Cannot apply: plan.tfplan missing. Plan did not succeed."}
        return s
    rc, out, err = terraform_cmd(["apply", "-auto-approve", "plan.tfplan"], workdir, env)
    log_step(s, "apply", rc, out, err, t0, time.time())
    if rc != 0:
        return handle_terraform_error(s, claude, workdir, err or out, step_name="apply")
    return s

def node_verify(s, workdir, env):
    t0 = time.time()
    rc, out, err = terraform_cmd(["show"], workdir, env)
    log_step(s, "verify", rc, out, err, t0, time.time())
    return s

def node_output(s, workdir, env):
    t0 = time.time()
    rc, out, err = terraform_cmd(["output", "-json"], workdir, env)
    log_step(s, "output", rc, out, err, t0, time.time())
    return s
