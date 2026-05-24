"""
Generate `Figure_07.pdf`.

Required base data on disk
- `artifacts/models/COLL10_SIM/heldout_obsmulti/variationalNN_relevant_only_latentdim2_mongoose-dive-7464.pt`
- `artifacts/analysis/create_synthetic_individuals/synthetic_sessions_ground_truth/simulated_data/D2_synthetic_latent_variables_mongoose-dive-7464.pt`
- `artifacts/analysis/generate_synthetic_item_observations/synthetic_data/COLL10_SIM/D2_synthetic_ground_truth_parameters_mongoose-dive-7464/all_synthetic_data_N240.pt`
- `artifacts/results/dale_runs/exp_c6_2d_dale_ps2/`
  containing per-session folders such as `LD2-001/` with
  `analysis/performance_tracking_session_LD2-001.csv`
"""

from wrapper_utils import choose_existing_run, copy_pdf, generated_output_dir, manuscript_dale_runs_dir, run_python_script


FIGURE_NAME = "Figure_07"


def generate(dry_run=False):
    basepath, run_id = choose_existing_run([
        (manuscript_dale_runs_dir(), "exp_c6_2d_dale_ps2"),
    ])
    output_root = generated_output_dir("fig07")
    run_python_script(
        "src/nmil_dlvm/analysis/dale_simulation/visualize_DALE_latent_trajectory.py",
        [
            "--performance-tracking-basepath", basepath,
            "--dale_run_id", run_id,
            "--output-dir", output_root,
            "--max_length_to_plot", "100",
        ],
        dry_run=dry_run,
    )
    if dry_run:
        return
    source_pdf = output_root / run_id / "DALE_max_100_points" / (
        run_id + "_3x3_RMSE_LogProb_colorbar_True_axis_labels_True.pdf"
    )
    copy_pdf(source_pdf, FIGURE_NAME)


if __name__ == "__main__":
    generate()
