import os, platform, zipfile, shutil, stat, urllib.request
from pathlib import Path
from typing import Dict, List, Tuple
from core.utils import run

def terraform_in_path() -> str:
    for p in os.environ.get("PATH", "").split(os.pathsep):
        tf = Path(p) / ("terraform.exe" if os.name == "nt" else "terraform")
        if tf.exists(): return str(tf)
    return ""

def ensure_terraform(target_dir: Path) -> str:
    existing = terraform_in_path()
    if existing: return existing
    target_dir.mkdir(parents=True, exist_ok=True)
    sysname, mach = platform.system().lower(), platform.machine().lower()
    arch = "arm64" if mach in ("arm64", "aarch64") else "amd64"
    osid = "windows" if "windows" in sysname else ("darwin" if "darwin" in sysname else "linux")
    ver = "1.9.8"
    zip_name = f"terraform_{ver}_{osid}_{arch}.zip"
    url = f"https://releases.hashicorp.com/terraform/{ver}/{zip_name}"
    zip_path = target_dir / zip_name
    with urllib.request.urlopen(url) as r, open(zip_path, "wb") as f: shutil.copyfileobj(r, f)
    with zipfile.ZipFile(zip_path) as z: z.extractall(target_dir)
    tf_bin = target_dir / ("terraform.exe" if os.name == "nt" else "terraform")
    tf_bin.chmod(tf_bin.stat().st_mode | stat.S_IEXEC)
    os.environ["PATH"] = str(target_dir) + os.pathsep + os.environ.get("PATH", "")
    return str(tf_bin)

def terraform_cmd(args: List[str], cwd: Path, env: Dict[str,str]) -> Tuple[int,str,str]:
    tf_path = terraform_in_path() or ensure_terraform(Path.home() / ".tfbin")
    return run([tf_path] + args, cwd=str(cwd), env=env)