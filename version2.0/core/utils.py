import json, datetime, subprocess
from typing import Dict, Any, List, Optional

def ensure_str(x) -> str:
    if isinstance(x, str): return x
    try: return json.dumps(x, indent=2, default=str)
    except Exception: return str(x)

def run(cmd: List[str], cwd: Optional[str]=None, env: Optional[Dict[str,str]]=None, timeout: Optional[int]=None):
    proc = subprocess.Popen(cmd, cwd=cwd, env=env or {}, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = proc.communicate(timeout=timeout)
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