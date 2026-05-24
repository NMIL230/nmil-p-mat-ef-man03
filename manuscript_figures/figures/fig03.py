"""
Generate `Figure_03.pdf`.

Required base data on disk
- `data/COLL10_SIM/all_data-best_mle_params_mpf100.pt`
- `artifacts/analysis/dlvm_imle_comparison/fitted_parameters/COLL10_SIM/D2_all_data-best_mle_params_mpf100/`
  with `synthetic_mle_params_N2.pt`, `synthetic_mle_params_N50.pt`, `synthetic_mle_params_N200.pt`,
  and `synthetic_dlvm_params_gradient_descent_D2_N2.pt`,
  `synthetic_dlvm_params_gradient_descent_D2_N50.pt`,
  `synthetic_dlvm_params_gradient_descent_D2_N200.pt`
- `artifacts/analysis/generate_synthetic_item_observations/synthetic_data/COLL10_SIM/all_data-best_mle_params_mpf100/`
  with `all_synthetic_data_N2.pt` and `all_synthetic_data_N50.pt`
"""

from pathlib import Path

from wrapper_utils import REPO_ROOT, copy_pdf, newest_rglob, run_python_script


FIGURE_NAME = "Figure_03"
PARAMS_DIR = REPO_ROOT / "artifacts/analysis/dlvm_imle_comparison/fitted_parameters/COLL10_SIM/D2_all_data-best_mle_params_mpf100"
GROUND_TRUTH = REPO_ROOT / "data/COLL10_SIM/all_data-best_mle_params_mpf100.pt"
SYNTHETIC_DATA_DIR = REPO_ROOT / "artifacts/analysis/generate_synthetic_item_observations/synthetic_data/COLL10_SIM/all_data-best_mle_params_mpf100"


def generate(dry_run=False):
    run_python_script(
        "src/nmil_dlvm/analysis/dlvm_imle_comparison/plot_representative_marginal_fits.py",
        [
            "--ground_truth_pt_file", GROUND_TRUTH,
            "--params_dir", PARAMS_DIR,
            "--synthetic_data_dir", SYNTHETIC_DATA_DIR,
            "--metric", "kld",
            "--eval_dataset_type", "training_set",
            "--normalize_errors",
        ],
        dry_run=dry_run,
    )
    if dry_run:
        return
    search_root = REPO_ROOT / "artifacts/analysis/dlvm_imle_comparison/fitted_parameters/COLL10_SIM/plots/marginals/training_set"
    copy_pdf(newest_rglob(search_root, "median_session_*_KLD.pdf"), FIGURE_NAME)


if __name__ == "__main__":
    generate()
