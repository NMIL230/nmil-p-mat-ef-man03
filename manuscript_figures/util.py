import os
import sys
import shlex
import subprocess
from typing import List, Optional

def script_path(scripts_root: str, script_name: str) -> str:
    p = os.path.join(scripts_root, script_name)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Required script not found: {p}")
    return p

def run_script(script: str, args: List[str], dry_run: bool = False) -> None:
    cmd = [sys.executable, script] + args
    print("Running:", " ".join(shlex.quote(c) for c in cmd))
    if dry_run:
        return
    subprocess.check_call(cmd)
