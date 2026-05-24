"""
Generate `Figure_S01.pdf`.

Required base data on disk
- `artifacts/analysis/create_synthetic_individuals/synthetic_sessions_ground_truth/simulated_data/`
  with the manuscript validation trio:
  `D1_synthetic_ground_truth_parameters_wolverine-zoom-7298.pt`,
  `D2_synthetic_ground_truth_parameters_mongoose-dive-7464.pt`, and
  `D3_synthetic_ground_truth_parameters_beaver-slide-5310.pt`
- `artifacts/analysis/dlvm_imle_comparison/fitted_parameters/COLL10_SIM/`
  with validation-fit subdirectories such as
  `D1_synthetic_ground_truth_parameters_*`,
  `D2_synthetic_ground_truth_parameters_*`, and
  `D3_synthetic_ground_truth_parameters_*`, each containing
  `synthetic_mle_params_N*.pt` and
  `synthetic_dlvm_params_gradient_descent_D*_N*.pt`
"""

from wrapper_utils import REPO_ROOT, copy_pdf, run_python_script, temporary_copied_files


FIGURE_NAME = "Figure_S01"
GROUND_TRUTH_DIR = REPO_ROOT / "artifacts/analysis/create_synthetic_individuals/synthetic_sessions_ground_truth/simulated_data"
PARAMS_DIR = REPO_ROOT / "artifacts/analysis/dlvm_imle_comparison/fitted_parameters/COLL10_SIM"
SOURCE_PDF = (
    PARAMS_DIR
    / "plots/validation_simulated/appendix_figure/KLD_combined_COLL10_SIM_combined_stddev_normalized_log_scale.pdf"
)
GROUND_TRUTH_FILES = {
    "D1_synthetic_ground_truth_parameters_wolverine-zoom-7298.pt":
        GROUND_TRUTH_DIR / "D1_synthetic_ground_truth_parameters_wolverine-zoom-7298.pt",
    "D2_synthetic_ground_truth_parameters_mongoose-dive-7464.pt":
        GROUND_TRUTH_DIR / "D2_synthetic_ground_truth_parameters_mongoose-dive-7464.pt",
    "D3_synthetic_ground_truth_parameters_beaver-slide-5310.pt":
        GROUND_TRUTH_DIR / "D3_synthetic_ground_truth_parameters_beaver-slide-5310.pt",
}


def generate(dry_run=False):
    if dry_run:
        run_python_script(
            "src/nmil_dlvm/analysis/dlvm_imle_comparison/plot_merged_curves.py",
            [
                "--ground_truth_dir", GROUND_TRUTH_DIR,
                "--params_dir", PARAMS_DIR,
                "--metric", "kld",
                "--eval_dataset_type", "validation_simulated",
                "--normalize_errors",
                "--plot_std_dev",
                "--show_single_imle_plot",
            ],
            dry_run=True,
        )
        return

    with temporary_copied_files(GROUND_TRUTH_FILES, "figS01_ground_truth") as ground_truth_dir:
        run_python_script(
            "src/nmil_dlvm/analysis/dlvm_imle_comparison/plot_merged_curves.py",
            [
                "--ground_truth_dir", ground_truth_dir,
                "--params_dir", PARAMS_DIR,
                "--metric", "kld",
                "--eval_dataset_type", "validation_simulated",
                "--normalize_errors",
                "--plot_std_dev",
                "--show_single_imle_plot",
            ],
        )
    copy_pdf(SOURCE_PDF, FIGURE_NAME)


if __name__ == "__main__":
    generate()
