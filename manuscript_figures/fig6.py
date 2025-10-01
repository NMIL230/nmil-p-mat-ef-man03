from typing import List, Optional, Union
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LOW_LEVEL_SCRIPT_RELATIVE = Path("src/analysis/dale_syn_data/generate_plots_Figure_06.py")
LOW_LEVEL_SCRIPT = REPO_ROOT / LOW_LEVEL_SCRIPT_RELATIVE

DEFAULT_RESULT_DIR = REPO_ROOT / "src" / "results" / "dale_sim_runs"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "src"
    / "analysis"
    / "dale_syn_data"
    / "plot"
    / "Figure06_v04_SampleDistributions_A_100"
)
DEFAULT_TARGET_PDF = "compare_methods_2D_sessions.pdf"
DEFAULT_TARGET_PNG_PATTERN = "compare_methods_2D_sessions_page_*.png"
FIGURE_ID = 6


def _ensure_low_level_script() -> Path:
    if not LOW_LEVEL_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Expected low-level script at {LOW_LEVEL_SCRIPT_RELATIVE} relative to repo root;"
            f" resolved path {LOW_LEVEL_SCRIPT} was not found."
        )
    return LOW_LEVEL_SCRIPT


def _resolve_relative(base: Path, candidate: Union[str, Path]) -> Path:
    candidate_path = Path(candidate)
    return candidate_path if candidate_path.is_absolute() else base / candidate_path


def _select_pdf(output_dir: Path, preferred_name: str) -> Path:
    target = output_dir / preferred_name
    if target.is_file():
        return target

    candidates = sorted(output_dir.glob("compare_methods_*_sessions*.pdf"))
    if not candidates:
        raise FileNotFoundError(f"No comparison PDFs found in {output_dir}.")

    def sort_key(path: Path) -> tuple[int, float]:
        contains_2d = 0 if "2D" in path.name else 1
        return (contains_2d, -path.stat().st_mtime)

    candidates.sort(key=sort_key)
    return candidates[0]


def _select_png(output_dir: Path, default_pattern: str, pdf_stem: str) -> Path:
    exact = output_dir / f"{pdf_stem}.png"
    if exact.is_file():
        return exact

    matches = sorted(output_dir.glob(default_pattern))
    if not matches:
        matches = sorted(output_dir.glob(f"{pdf_stem}*.png"))
    if not matches:
        raise FileNotFoundError(
            f"No PNG outputs found in {output_dir} matching {default_pattern} or {pdf_stem}*.png."
        )
    return matches[0]


def generate(
    resultdir: Optional[Union[Path, str]] = None,
    output_dir: Optional[Union[Path, str]] = None,
    passthrough: Optional[List[str]] = None,
    dry_run: bool = False,
) -> None:
    """Generate Figure 6 via the DALE sampling distribution plotter and save standardized copies."""

    script_path = _ensure_low_level_script()

    resolved_result_dir = DEFAULT_RESULT_DIR if resultdir is None else _resolve_relative(script_path.parent, resultdir)
    resolved_output_dir = DEFAULT_OUTPUT_DIR if output_dir is None else _resolve_relative(script_path.parent, output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    if not dry_run and not resolved_result_dir.is_dir():
        raise FileNotFoundError(
            f"Required result directory not found: {resolved_result_dir}. Provide --resultdir via passthrough or generate it first."
        )

    args: List[str] = []
    if resolved_result_dir is not None:
        args.extend(["--resultdir", str(resolved_result_dir)])
    if resolved_output_dir is not None:
        args.extend(["--output_dir", str(resolved_output_dir)])
    if passthrough:
        args.extend(passthrough)

    cmd = [sys.executable, str(script_path)] + args
    print("Running:", " ".join(shlex.quote(c) for c in cmd))
    if dry_run:
        return
    try:
        subprocess.check_call(cmd, cwd=str(script_path.parent))
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Figure 6 generation script failed; ensure required result folders exist or pass overrides via passthrough."
        ) from exc

    target_pdf = _select_pdf(resolved_output_dir, DEFAULT_TARGET_PDF)
    target_png = _select_png(resolved_output_dir, DEFAULT_TARGET_PNG_PATTERN, target_pdf.stem)

    generated_dir = REPO_ROOT / "manuscript_figures" / "generated_figures"
    generated_dir.mkdir(parents=True, exist_ok=True)

    destination_pdf = generated_dir / f"Figure_{FIGURE_ID:02d}.pdf"
    destination_png = generated_dir / f"Figure_{FIGURE_ID:02d}.png"

    shutil.copy2(target_pdf, destination_pdf)
    shutil.copy2(target_png, destination_png)
    print(f"Copied {target_pdf} -> {destination_pdf}")
    print(f"Copied {target_png} -> {destination_png}")


if __name__ == "__main__":
    generate()
