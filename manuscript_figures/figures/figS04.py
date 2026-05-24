"""
Generate `Figure_S04.pdf`.

Required base data on disk
- `artifacts/analysis/dale_simulation/plot_data/post_run/validation/d2/dale_comparison_data.csv`
"""

from wrapper_utils import REPO_ROOT, copy_pdf, generated_output_dir, run_python_script


FIGURE_NAME = "Figure_S04"
CSV_FILE = REPO_ROOT / "artifacts/analysis/dale_simulation/plot_data/post_run/validation/d2/dale_comparison_data.csv"


def generate(dry_run=False):
    output_dir = generated_output_dir("figS04")
    run_python_script(
        "src/nmil_dlvm/analysis/dale_simulation/plot_dale_comparison_from_csv.py",
        [
            "--csv_file", CSV_FILE,
            "--output_dir", output_dir,
            "--output_filename", "figureS04_source",
            "--metric", "kld",
            "--dlvm_label_dim", "2",
        ],
        dry_run=dry_run,
    )
    if dry_run:
        return
    copy_pdf(output_dir / "figureS04_source.pdf", FIGURE_NAME)


if __name__ == "__main__":
    generate()
