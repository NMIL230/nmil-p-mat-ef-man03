from typing import List, Optional, Union
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

try:  # Support both package and script execution styles
    from manuscript_figures._ground_truth_utils import ensure_ground_truth_file, resolve_relative
except ImportError:  # pragma: no cover - fallback for direct script execution
    sys.path.append(str(Path(__file__).resolve().parent))
    from _ground_truth_utils import ensure_ground_truth_file, resolve_relative


REPO_ROOT = Path(__file__).resolve().parents[1]
LOW_LEVEL_SCRIPT_RELATIVE = Path("src/analysis/dlvm_imle_comparison/plot_merged_curves.py")
LOW_LEVEL_SCRIPT = REPO_ROOT / LOW_LEVEL_SCRIPT_RELATIVE

DEFAULT_GROUND_TRUTH_DIR = (
    REPO_ROOT
    / "src"
    / "analysis"
    / "create_synthetic_individuals"
    / "synthetic_sessions_ground_truth"
    / "simulated_data"
)
DEFAULT_PARAMS_DIR = (
    REPO_ROOT
    / "src"
    / "analysis"
    / "dlvm_imle_comparison"
    / "fitted_parameters"
    / "COLL10_SIM"
)
DEFAULT_METRIC = "kld"
DEFAULT_EVAL_DATASET_TYPE = "validation_simulated"
FIGURE_ID = 4


def _ensure_low_level_script() -> Path:
    if not LOW_LEVEL_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Expected low-level script at {LOW_LEVEL_SCRIPT_RELATIVE} relative to repo root;"
            f" resolved path {LOW_LEVEL_SCRIPT} was not found."
        )
    return LOW_LEVEL_SCRIPT


def generate(
    ground_truth_dir: Union[Path, str] = DEFAULT_GROUND_TRUTH_DIR,
    params_dir: Union[Path, str] = DEFAULT_PARAMS_DIR,
    metric: str = DEFAULT_METRIC,
    eval_dataset_type: str = DEFAULT_EVAL_DATASET_TYPE,
    normalize_errors: bool = True,
    plot_std_dev: bool = True,
    show_single_dlvm_plot: bool = True,
    show_single_imle_plot: bool = True,
    passthrough: Optional[List[str]] = None,
    dry_run: bool = False,
) -> None:
    """Generate Figure 4 by running the merged curves script and copying the combined comparison plot."""

    script_path = _ensure_low_level_script()
    metric = metric.lower()
    if metric not in {"kld", "rmse"}:
        raise ValueError("metric must be 'kld' or 'rmse'")

    params_dir_path = resolve_relative(REPO_ROOT, params_dir)
    if not params_dir_path.is_dir():
        raise FileNotFoundError(f"Parameters directory not found: {params_dir_path}")

    ground_truth_dir_path = resolve_relative(REPO_ROOT, ground_truth_dir)
    ground_truth_dir_path.mkdir(parents=True, exist_ok=True)
    ground_truth_file = ground_truth_dir_path / "all_data-best_mle_params_mpf100.pt"
    resolved_ground_truth = ensure_ground_truth_file(ground_truth_file, params_dir_path)
    if resolved_ground_truth != ground_truth_file:
        shutil.copy2(resolved_ground_truth, ground_truth_file)

    args = [
        "--ground_truth_dir",
        str(ground_truth_dir_path),
        "--params_dir",
        str(params_dir_path),
        "--metric",
        metric,
        "--eval_dataset_type",
        eval_dataset_type,
    ]
    if normalize_errors:
        args.append("--normalize_errors")
    if plot_std_dev:
        args.append("--plot_std_dev")
    if show_single_dlvm_plot:
        args.append("--show_single_dlvm_plot")
    if show_single_imle_plot:
        args.append("--show_single_imle_plot")
    if passthrough:
        args.extend(passthrough)

    cmd = [sys.executable, str(script_path)] + args
    print("Running:", " ".join(shlex.quote(c) for c in cmd))
    if dry_run:
        return
    subprocess.check_call(cmd, cwd=str(script_path.parent))

    dataset_name = params_dir_path.name
    plots_dir = (
        script_path.parent
        / "fitted_parameters"
        / dataset_name
        / "plots"
        / eval_dataset_type
        / "main_figure"
    )
    if not plots_dir.is_dir():
        raise FileNotFoundError(
            f"Expected plots directory {plots_dir} was not created."
        )

    error_metric_suffix = "stddev" if plot_std_dev else "stderr"
    norm_suffix = "normalized" if normalize_errors else "unnormalized"
    final_suffix = f"{error_metric_suffix}_{norm_suffix}"
    metric_upper = metric.upper()
    expected_stem = f"{metric_upper}_combined_{dataset_name}_combined_{final_suffix}_log_scale"

    target_pdf = plots_dir / f"{expected_stem}.pdf"
    if not target_pdf.is_file():
        candidates = sorted(plots_dir.glob(f"{metric_upper}_combined_{dataset_name}_combined_*_log_scale.pdf"))
        if not candidates:
            raise FileNotFoundError(
                f"Could not find any combined log plot for metric {metric_upper} in {plots_dir}."
            )
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
