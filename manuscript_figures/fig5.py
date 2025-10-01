from typing import List, Optional, Union
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LOW_LEVEL_SCRIPT_RELATIVE = Path("src/analysis/dale_syn_data/generate_plots_Figure_05.py")
LOW_LEVEL_SCRIPT = REPO_ROOT / LOW_LEVEL_SCRIPT_RELATIVE

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "src"
    / "analysis"
    / "dale_syn_data"
    / "plot"
    / "Figure05_v03_TestingFitEvolution"
)
DEFAULT_OUTPUT_FILENAME = "Figure05_v03_TestingFitEvolution"
DEFAULT_CSV_FILE = (
    REPO_ROOT
    / "src"
    / "analysis"
    / "dale_syn_data"
    / "plot_data"
    / "2d-final"
    / "dale_comparison_data.csv"
)
FIGURE_ID = 5


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


def _select_output_file(output_dir: Path, desired_stem: Optional[str]) -> Path:
    if desired_stem:
        target = output_dir / f"{desired_stem}.pdf"
        if not target.is_file():
            raise FileNotFoundError(f"Expected PDF {target} was not found.")
        return target

    candidates = sorted(output_dir.glob("*.pdf"))
    if not candidates:
        raise FileNotFoundError(f"No PDF outputs found in {output_dir}.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def generate(
    csv_file: Optional[Union[Path, str]] = None,
    output_dir: Optional[Union[Path, str]] = None,
    output_filename: Optional[str] = None,
    passthrough: Optional[List[str]] = None,
    dry_run: bool = False,
) -> None:
    """Generate Figure 5 via the DALE comparison plotter and save standardized copies."""

    script_path = _ensure_low_level_script()

    resolved_csv = DEFAULT_CSV_FILE if csv_file is None else _resolve_relative(REPO_ROOT, csv_file)
    resolved_output_dir = DEFAULT_OUTPUT_DIR if output_dir is None else _resolve_relative(script_path.parent, output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    if not dry_run and not resolved_csv.is_file():
        raise FileNotFoundError(
            f"Required CSV file not found: {resolved_csv}. Provide --csv_file via passthrough or generate it first."
        )

    args: List[str] = []
    if resolved_csv is not None:
        args.extend(["--csv_file", str(resolved_csv)])
    if resolved_output_dir is not None:
        args.extend(["--output_dir", str(resolved_output_dir)])
    if output_filename is not None:
        args.extend(["--output_filename", output_filename])
    else:
        output_filename = DEFAULT_OUTPUT_FILENAME
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
            "Figure 5 generation script failed; ensure required data files exist or pass overrides via passthrough."
        ) from exc

    target_pdf = _select_output_file(resolved_output_dir, output_filename)
    target_png = target_pdf.with_suffix(".png")
    if not target_png.is_file():
        raise FileNotFoundError(f"Expected PNG companion {target_png} was not found.")

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
