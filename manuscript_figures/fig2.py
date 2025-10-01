from typing import List, Optional
import shlex
import subprocess
import sys
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LOW_LEVEL_SCRIPT_RELATIVE = Path("src/analysis/create_synthetic_individuals/create_ground_truth_sessions.py")
LOW_LEVEL_SCRIPT = REPO_ROOT / LOW_LEVEL_SCRIPT_RELATIVE

DEFAULT_LATENT_DIM = 2
FIGURE_ID = 2


def _ensure_low_level_script() -> Path:
    if not LOW_LEVEL_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Expected low-level script at {LOW_LEVEL_SCRIPT_RELATIVE} relative to repo root;"
            f" resolved path {LOW_LEVEL_SCRIPT} was not found."
        )
    return LOW_LEVEL_SCRIPT


def generate(
    latent_dim: int = DEFAULT_LATENT_DIM,
    passthrough: Optional[List[str]] = None,
    dry_run: bool = False,
) -> None:
    """Generate Figure 2 using the repository's ground-truth session creator."""

    script_path = _ensure_low_level_script()
    args = [
        "--latent_dim",
        str(latent_dim),
    ]
    if passthrough:
        args.extend(passthrough)
    cmd = [sys.executable, str(script_path)] + args
    print("Running:", " ".join(shlex.quote(c) for c in cmd))
    if dry_run:
        return
    subprocess.check_call(cmd, cwd=str(script_path.parent))

    # Locate generated PDF and PNG, copy to manuscript_figures/generated_figures/Figure_XX.{pdf,png}
    output_dir = script_path.parent / "synthetic_sessions_ground_truth"
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Expected output directory {output_dir} was not created.")

    pattern = f"D{latent_dim}_latent_space_parameter_mapping_*.pdf"
    candidates = sorted(output_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find any files matching {pattern} in {output_dir}."
        )

    # Pick the most recently modified match in case multiple exist
    target_pdf = max(candidates, key=lambda p: p.stat().st_mtime)
    target_png = target_pdf.with_suffix(".png")
    if not target_png.is_file():
        raise FileNotFoundError(f"Expected companion PNG for {target_pdf} was not found.")

    generated_dir = REPO_ROOT / "manuscript_figures" / "generated_figures"
    generated_dir.mkdir(parents=True, exist_ok=True)

    for source, suffix in ((target_pdf, ".pdf"), (target_png, ".png")):
        destination = generated_dir / f"Figure_{FIGURE_ID:02d}{suffix}"
        shutil.copy2(source, destination)
        print(f"Copied {source} -> {destination}")


if __name__ == "__main__":
    generate()
