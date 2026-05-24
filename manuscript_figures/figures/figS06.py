"""
Generate `Figure_S06.pdf`.

Required base data on disk
- `data/COLL10_SIM/all_data-best_mle_params_mpf100.pt`
- `artifacts/analysis/dlvm_imle_comparison/fitted_parameters/COLL10_SIM/`
  with training-fit subdirectories
  `D1_all_data-best_mle_params_mpf100/`,
  `D2_all_data-best_mle_params_mpf100/`, and
  `D3_all_data-best_mle_params_mpf100/`, each containing
  `synthetic_mle_params_N*.pt` and
  `synthetic_dlvm_params_gradient_descent_D*_N*.pt`
"""

from wrapper_utils import REPO_ROOT, copy_pdf, run_python_script


FIGURE_NAME = "Figure_S06"
GROUND_TRUTH_DIR = REPO_ROOT / "data/COLL10_SIM"
PARAMS_DIR = REPO_ROOT / "artifacts/analysis/dlvm_imle_comparison/fitted_parameters/COLL10_SIM"
SOURCE_PDF = (
    PARAMS_DIR
    / "plots/training_set/appendix_figure/KLD_combined_COLL10_SIM_combined_stddev_normalized_log_scale.pdf"
)


def generate(dry_run=False):
    run_python_script(
        "src/nmil_dlvm/analysis/dlvm_imle_comparison/plot_merged_curves.py",
        [
            "--ground_truth_dir", GROUND_TRUTH_DIR,
            "--params_dir", PARAMS_DIR,
            "--metric", "kld",
            "--eval_dataset_type", "training_set",
            "--normalize_errors",
            "--plot_std_dev",
            "--show_single_imle_plot",
        ],
        dry_run=dry_run,
    )
    if dry_run:
        return
    copy_pdf(SOURCE_PDF, FIGURE_NAME)


if __name__ == "__main__":
    generate()
