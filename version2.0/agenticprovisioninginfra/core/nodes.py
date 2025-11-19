import time
import json
import shutil
import re
from pathlib import Path
from difflib import unified_diff
from core.utils import log_step
from core.terraform import terraform_cmd
from core.prompts import SYSTEM_PROMPT

# --- Safe JSON Parsing ---
def safe_json_parse(raw: str):
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    cleaned = re.sub(r"[\x00-\x1F\x7F]", "", match.group(0))
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}

# --- Deduplication Across Directory ---
def deduplicate_providers_in_dir(workdir: Path):
    seen_providers = set()
    for p in sorted(workdir.glob("*.tf")):
        lines = p.read_text().splitlines()
        result, buffer = [], []
        inside_provider = False
        for line in lines:
            if line.strip().startswith('provider'):
                inside_provider = True
                buffer = [line]
                continue
            if inside_provider:
                buffer.append(line)
                if line.strip() == '}':
                    provider_name = buffer[0].split('"')[1]
                    if provider_name not in seen_providers:
                        seen_providers.add(provider_name)
                        result.extend(buffer)
                    buffer, inside_provider = [], False
                continue
            result.append(line)
        p.write_text("\n".join(result))

# --- Provider Injection ---
def ensure_required_providers(workdir: Path):
    main_tf = workdir / "main.tf"
    if not main_tf.exists():
        main_tf.write_text("")
    if "required_providers" not in main_tf.read_text():
        block = """
terraform {
  required_providers {
    aws = { source = "hashicorp/aws" version = "~> 5.0" }
    random = { source = "hashicorp/random" version = "~> 3.0" }
  }
}
provider "aws" { region = var.region }
"""
        main_tf.write_text(main_tf.read_text() + "\n" + block)

# --- Cleanup Terraform State ---
def cleanup_terraform(workdir: Path):
    for item in [workdir / ".terraform", workdir / ".terraform.lock.hcl"]:
        if item.exists():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

# --- File Change Reporter ---
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

# --- Retry Logic for Invalid JSON ---
def retry_claude(claude, prompt, files_json):
    raw_retry = claude.complete(
        "STRICT MODE: Fix Terraform error and return ONLY valid JSON with filenames as keys and file content as values. No extra text.",
        json.dumps(files_json)
    )
    return safe_json_parse(raw_retry)

# --- Handle Terraform Errors ---
def handle_terraform_error(s, claude, workdir, error_msg, step_name):
    # Detect IAM-related errors and stop immediately
    if any(keyword in error_msg for keyword in ["AccessDenied", "UnauthorizedOperation", "IAM", "STS"]):
        s["steps"][f"repair_{step_name}"] = {"stdout": f"Non-recoverable IAM error detected: {error_msg}"}
        return s

    files_json = {p.name: p.read_text() for p in workdir.glob("*.tf")}
    prompt = f"""
    The following Terraform error occurred during {step_name}:
    {error_msg}

    Fix the issue in the provided files. Return ONLY valid JSON with filenames as keys and corrected file content as values.
    """
    raw = claude.complete(prompt, json.dumps(files_json))
    fixes = safe_json_parse(raw)
    if not fixes:
        fixes = retry_claude(claude, prompt, files_json)
    if not fixes:
        s["steps"][f"repair_{step_name}"] = {"stdout": f"Claude could not fix error after retry: {error_msg}"}
        return s

    changes = report_file_changes(workdir, fixes)
    ensure_required_providers(workdir)
    deduplicate_providers_in_dir(workdir)
    cleanup_terraform(workdir)
    s["steps"][f"repair_{step_name}"] = {
        "stdout": f"Error fixed for {step_name}. Changes:\n{'\n'.join(changes)}"
    }
    return s

# --- Nodes ---
def node_codegen(s, claude, workdir, prompt):
    raw = claude.complete(SYSTEM_PROMPT, prompt)
    files = safe_json_parse(raw)
    if not files:
        files = retry_claude(claude, SYSTEM_PROMPT, {"prompt": prompt})
    if not files:
        s["steps"]["codegen"] = {"stdout": "Invalid JSON from Claude after retry, skipping codegen."}
        return s
    changes = report_file_changes(workdir, files)
    ensure_required_providers(workdir)
    deduplicate_providers_in_dir(workdir)
    s["steps"]["codegen"] = {"stdout": f"Files generated. Changes:\n{'\n'.join(changes)}"}
    return s

def node_init(s, workdir, env):
    t0 = time.time()
    rc, out, err = terraform_cmd(["init"], workdir, env)

    # Handle lock file inconsistency
    if rc != 0 and "lock file" in (err or out):
        cleanup_terraform(workdir)
        rc, out, err = terraform_cmd(["init", "-upgrade"], workdir, env)

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
        rc, out, err = terraform_cmd(["plan", "-out", "plan.tfplan"], workdir, env)
        log_step(s, f"plan_attempt_{attempt+1}", rc, out, err, t0, time.time())

        if rc == 0:
            s["steps"]["plan"] = {"stdout": f"Plan succeeded on attempt {attempt+1}"}
            return s

        # If plan fails, try to fix using Claude
        s["steps"][f"plan_fix_{attempt+1}"] = {"stdout": f"Plan failed. Attempting fix..."}
        s = handle_terraform_error(s, claude, workdir, err or out, step_name=f"plan_attempt_{attempt+1}")
        attempt += 1

    # If all retries fail
    s["steps"]["plan"] = {"stdout": f"Plan failed after {max_retries} attempts. Manual intervention required."}
    return s

def node_apply(s, claude, workdir, env):
    t0 = time.time()
    plan_file = workdir / "plan.tfplan"

    # Ensure plan succeeded before apply
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