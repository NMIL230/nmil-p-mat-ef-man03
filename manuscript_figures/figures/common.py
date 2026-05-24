import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT_ROOT = Path(__file__).resolve().parent
FIGURES_DIR = MANUSCRIPT_ROOT / "figures"
GENERATED_DIR = MANUSCRIPT_ROOT / "generated"
DALE_RUNS_DIR = REPO_ROOT / "artifacts" / "results" / "dale_runs"
DEFAULT_CHILD_ENV = {
    "WANDB_MODE": "disabled",
    "WANDB_SILENT": "true",
}


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def figure_destination(figure_name):
    ensure_dir(FIGURES_DIR)
    return FIGURES_DIR / (figure_name + ".pdf")


def generated_output_dir(wrapper_name):
    return ensure_dir(GENERATED_DIR / wrapper_name)


def manuscript_dale_runs_dir():
    return require_dir(DALE_RUNS_DIR, "DALE runs root")


def require_file(path, label=None):
    if not path.is_file():
        raise FileNotFoundError("%s not found: %s" % (label or "File", path))
    return path


def require_dir(path, label=None):
    if not path.is_dir():
        raise FileNotFoundError("%s not found: %s" % (label or "Directory", path))
    return path


def run_python_script(script_relpath, args, dry_run=False, cwd=None, env_overrides=None):
    script_path = require_file(REPO_ROOT / script_relpath, "Low-level script")
    cmd = [sys.executable, str(script_path)] + [str(arg) for arg in args]
    print("Running:", " ".join(shlex.quote(part) for part in cmd))
    if dry_run:
        return script_path
    child_env = os.environ.copy()
    child_env.update(DEFAULT_CHILD_ENV)
    if env_overrides:
        child_env.update({key: str(value) for key, value in env_overrides.items()})
    subprocess.check_call(cmd, cwd=str(cwd or script_path.parent), env=child_env)
    return script_path


def run_python_script_via_runpy(script_relpath, args, prelude_lines, dry_run=False, cwd=None, env_overrides=None):
    script_path = require_file(REPO_ROOT / script_relpath, "Low-level script")
    snippet_lines = [
        "import runpy",
        "import sys",
        "from pathlib import Path",
        "SCRIPT_PATH = Path(%r)" % str(script_path),
    ]
    snippet_lines.extend(prelude_lines)
    snippet_lines.extend(
        [
            "sys.argv = [str(SCRIPT_PATH)] + %r" % [str(arg) for arg in args],
            "runpy.run_path(str(SCRIPT_PATH), run_name='__main__')",
        ]
    )
    cmd = [sys.executable, "-c", "\n".join(snippet_lines)]
    print("Running:", " ".join(shlex.quote(part) for part in cmd))
    if dry_run:
        return script_path
    child_env = os.environ.copy()
    child_env.update(DEFAULT_CHILD_ENV)
    if env_overrides:
        child_env.update({key: str(value) for key, value in env_overrides.items()})
    subprocess.check_call(cmd, cwd=str(cwd or script_path.parent), env=child_env)
    return script_path


def newest_match(pattern):
    matches = sorted(pattern.parent.glob(pattern.name))
    if not matches:
        raise FileNotFoundError("No files matched pattern: %s" % pattern)
    return max(matches, key=lambda path: path.stat().st_mtime)


def newest_rglob(root, pattern):
    matches = sorted(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError("No files matched pattern '%s' under %s" % (pattern, root))
    return max(matches, key=lambda path: path.stat().st_mtime)


def copy_pdf(source_pdf, figure_name):
    source_pdf = require_file(Path(source_pdf), "Source PDF")
    destination = figure_destination(figure_name)
    shutil.copy2(str(source_pdf), str(destination))
    print("Copied %s -> %s" % (source_pdf, destination))
    return destination


def choose_existing_path(paths, label):
    for path in paths:
        if Path(path).exists():
            return Path(path)
    raise FileNotFoundError("Could not resolve %s from candidates: %s" % (label, ", ".join(str(p) for p in paths)))


def choose_existing_run(candidates):
    for basepath, run_id in candidates:
        basepath = Path(basepath)
        if (basepath / run_id).is_dir():
            return basepath, run_id
    raise FileNotFoundError(
        "Could not find any run directory from candidates: %s"
        % ", ".join("%s/%s" % (base, run_id) for base, run_id in candidates)
    )


@contextmanager
def temporary_alias_root(alias_to_candidates, prefix):
    ensure_dir(GENERATED_DIR)
    alias_root = Path(tempfile.mkdtemp(prefix=prefix + "_", dir=str(GENERATED_DIR)))
    try:
        for alias_name, candidates in alias_to_candidates.items():
            target = choose_existing_path(candidates, alias_name)
            os.symlink(str(target), str(alias_root / alias_name))
        yield alias_root
    finally:
        shutil.rmtree(str(alias_root), ignore_errors=True)


@contextmanager
def temporary_copied_files(file_map, prefix):
    ensure_dir(GENERATED_DIR)
    copied_root = Path(tempfile.mkdtemp(prefix=prefix + "_", dir=str(GENERATED_DIR)))
    try:
        for dest_name, source_path in file_map.items():
            source_path = require_file(Path(source_path), "Copied source file")
            shutil.copy2(str(source_path), str(copied_root / dest_name))
        yield copied_root
    finally:
        shutil.rmtree(str(copied_root), ignore_errors=True)
