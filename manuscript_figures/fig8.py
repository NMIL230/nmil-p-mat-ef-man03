from typing import List, Optional, Union
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LOW_LEVEL_SCRIPT_RELATIVE = Path("src/analysis/dale_syn_data/visualize_DALE_latent_trajectory.py")
LOW_LEVEL_SCRIPT = REPO_ROOT / LOW_LEVEL_SCRIPT_RELATIVE

DEFAULT_DALE_RUN_ID = "exp_c6_2d_dale_ps2"
DEFAULT_MAX_LENGTH_TO_PLOT = 100
DEFAULT_PERFORMANCE_TRACKING_BASEPATH = (
    REPO_ROOT / "src" / "results" / "dale_sim_runs"
)
FIGURE_ID = 8


def _ensure_low_level_script() -> Path:
    if not LOW_LEVEL_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Expected low-level script at {LOW_LEVEL_SCRIPT_RELATIVE} relative to repo root;"
            f" resolved path {LOW_LEVEL_SCRIPT} was not found."
        )
    return LOW_LEVEL_SCRIPT


def generate(
    dale_run_id: str = DEFAULT_DALE_RUN_ID,
    max_length_to_plot: int = DEFAULT_MAX_LENGTH_TO_PLOT,
    performance_tracking_basepath: Optional[Union[Path, str]] = DEFAULT_PERFORMANCE_TRACKING_BASEPATH,
    passthrough: Optional[List[str]] = None,
    dry_run: bool = False,
) -> None:
    """Generate Figure 8 using the repository's latent trajectory visualizer."""

    script_path = _ensure_low_level_script()
    if performance_tracking_basepath is None:
        resolved_basepath = DEFAULT_PERFORMANCE_TRACKING_BASEPATH
    else:
        resolved_basepath = Path(performance_tracking_basepath)
        if not resolved_basepath.is_absolute():
            resolved_basepath = (script_path.parent / resolved_basepath).resolve()

    if not dry_run and not resolved_basepath.is_dir():
        raise FileNotFoundError(
            f"Performance tracking base path not found: {resolved_basepath}"
        )

    args = [
        "--performance-tracking-basepath",
        str(resolved_basepath),
        "--dale_run_id",
        dale_run_id,
        "--max_length_to_plot",
        str(max_length_to_plot),
    ]
    if passthrough:
        args.extend(passthrough)
    cmd = [sys.executable, str(script_path)] + args
    print("Running:", " ".join(shlex.quote(c) for c in cmd))
    if dry_run:
        return
    subprocess.check_call(cmd, cwd=str(script_path.parent))

    output_dir = (
        script_path.parent
        / "trajectory_visualization"
        / dale_run_id
        / f"DALE_max_{max_length_to_plot}_points"
    )
    if not output_dir.is_dir():
        raise FileNotFoundError(
            f"Expected output directory {output_dir} was not created."
        )

    stem = f"{dale_run_id}_RMSE_vs_LogProb_scatter"
    generated_dir = REPO_ROOT / "manuscript_figures" / "generated_figures"
    generated_dir.mkdir(parents=True, exist_ok=True)

    for suffix in (".pdf", ".png"):
        source = output_dir / f"{stem}{suffix}"
        if not source.is_file():
            raise FileNotFoundError(
                f"Expected scatter plot {source} was not found."
            )
        destination = generated_dir / f"Figure_{FIGURE_ID:02d}{suffix}"
        shutil.copy2(source, destination)
        print(f"Copied {source} -> {destination}")


if __name__ == "__main__":
    generate()
