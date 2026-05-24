"""
Generate `Figure_02.pdf`.

Required base data on disk
- `artifacts/models/COLL10_SIM/**/variationalNN_relevant_only_latentdim2_mongoose-dive-7464.pt`
  The underlying script resolves this filename recursively under `artifacts/models/COLL10_SIM/`.
"""

from wrapper_utils import copy_pdf, generated_output_dir, run_python_script


FIGURE_NAME = "Figure_02"


def generate(dry_run=False):
    output_dir = generated_output_dir("fig02")
    run_python_script(
        "src/nmil_dlvm/analysis/create_synthetic_individuals/create_ground_truth_sessions.py",
        [
            "--latent_dim", "2",
            "--model-id", "mongoose-dive-7464",
            "--output_dir", output_dir,
        ],
        dry_run=dry_run,
    )
    if dry_run:
        return
    copy_pdf(output_dir / "D2_latent_space_parameter_mapping_mongoose-dive-7464.pdf", FIGURE_NAME)


if __name__ == "__main__":
    generate()
