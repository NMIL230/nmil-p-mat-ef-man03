from typing import List, Optional, Union
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

try:  # Support both `python manuscript_figures/fig3.py` and `python -m manuscript_figures.runner`
    from manuscript_figures._ground_truth_utils import ensure_ground_truth_file, resolve_relative
except ImportError:  # pragma: no cover - fallback for direct script execution
    sys.path.append(str(Path(__file__).resolve().parent))
    from _ground_truth_utils import ensure_ground_truth_file, resolve_relative


REPO_ROOT = Path(__file__).resolve().parents[1]
LOW_LEVEL_SCRIPT_RELATIVE = Path("src/analysis/dlvm_imle_comparison/plot_marginal_median.py")
LOW_LEVEL_SCRIPT = REPO_ROOT / LOW_LEVEL_SCRIPT_RELATIVE

DEFAULT_GROUND_TRUTH_PT = REPO_ROOT / "src" / "data" / "COLL10_SIM" / "all_data-best_mle_params_mpf100.pt"
DEFAULT_PARAMS_DIR = (
    REPO_ROOT
    / "src"
    / "analysis"
    / "dlvm_imle_comparison"
    / "fitted_parameters"
    / "COLL10_SIM"
    / "D2_all_data-best_mle_params_mpf100_lean302run5"
)
DEFAULT_SYNTHETIC_DATA_DIR = (
    REPO_ROOT
    / "src"
    / "analysis"
    / "generate_synthetic_item_observations"
    / "synthetic_data"
    / "COLL10_SIM"
    / "all_data-best_mle_params_mpf100_lean302run5"
)
DEFAULT_METRIC = "kld"
DEFAULT_EVAL_DATASET_TYPE = "training_set"
FIGURE_ID = 3


def _ensure_low_level_script() -> Path:
    if not LOW_LEVEL_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Expected low-level script at {LOW_LEVEL_SCRIPT_RELATIVE} relative to repo root;"
            f" resolved path {LOW_LEVEL_SCRIPT} was not found."
        )
    return LOW_LEVEL_SCRIPT


def generate(
    ground_truth_pt_file: Union[Path, str] = DEFAULT_GROUND_TRUTH_PT,
    params_dir: Union[Path, str] = DEFAULT_PARAMS_DIR,
    synthetic_data_dir: Union[Path, str] = DEFAULT_SYNTHETIC_DATA_DIR,
    metric: str = DEFAULT_METRIC,
    eval_dataset_type: str = DEFAULT_EVAL_DATASET_TYPE,
    normalize_errors: bool = True,
    passthrough: Optional[List[str]] = None,
    dry_run: bool = False,
) -> None:
    """Generate Figure 3 by running the marginal median plot script and copying the median output."""

    script_path = _ensure_low_level_script()
    metric = metric.lower()
    if metric not in {"kld", "rmse"}:
        raise ValueError("metric must be 'kld' or 'rmse'")

    params_dir_path = resolve_relative(REPO_ROOT, params_dir)
    synthetic_data_dir_path = resolve_relative(REPO_ROOT, synthetic_data_dir)
    if not params_dir_path.is_dir():
        raise FileNotFoundError(f"Parameters directory not found: {params_dir_path}")
    if not synthetic_data_dir_path.is_dir():
        raise FileNotFoundError(f"Synthetic data directory not found: {synthetic_data_dir_path}")

    allowed_sessions: Optional[list[str]] = None
    participant_csv_candidates = [
        REPO_ROOT / "src" / "data" / "COLL10_SIM" / "participant_ids_not_normed.csv",
    ]
    for candidate in participant_csv_candidates:
        if candidate.is_file():
            allowed_sessions = (
                pd.read_csv(candidate)["ids"].dropna().astype(str).tolist()
            )
            break

    ground_truth_path = ensure_ground_truth_file(
        ground_truth_pt_file,
        params_dir_path,
        allowed_sessions=allowed_sessions,
    )

    args = [
        "--ground_truth_pt_file",
        str(ground_truth_path),
        "--params_dir",
        str(params_dir_path),
        "--synthetic_data_dir",
        str(synthetic_data_dir_path),
        "--metric",
        metric,
        "--eval_dataset_type",
        eval_dataset_type,
    ]
    if normalize_errors:
        args.append("--normalize_errors")
    if passthrough:
        args.extend(passthrough)
    cmd = [sys.executable, str(script_path)] + args
    print("Running:", " ".join(shlex.quote(c) for c in cmd))
    if dry_run:
        return
    subprocess.check_call(cmd, cwd=str(script_path.parent))

    search_root = script_path.parent / "fitted_parameters"
    pattern = f"median_session_*_{metric.upper()}.pdf"
    candidates = [p for p in search_root.rglob(pattern) if eval_dataset_type in p.parts]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find any files matching {pattern} within {search_root} (eval_dataset_type={eval_dataset_type})."
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
