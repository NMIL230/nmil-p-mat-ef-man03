"""
Generate `Figure_05.pdf`.

Required base data on disk
- `artifacts/analysis/dale_simulation/plot_data/post_run/validation/d2/dale_comparison_data.csv`
"""

from wrapper_utils import REPO_ROOT, copy_pdf, generated_output_dir, run_python_script_via_runpy


FIGURE_NAME = "Figure_05"
CSV_FILE = REPO_ROOT / "artifacts/analysis/dale_simulation/plot_data/post_run/validation/d2/dale_comparison_data.csv"


def generate(dry_run=False):
    output_dir = generated_output_dir("fig05")
    # Manuscript Figure 05 requires the grid-enabled rendering, but the existing
    # low-level script does not expose grid control on its CLI.
    run_python_script_via_runpy(
        "src/nmil_dlvm/analysis/dale_simulation/plot_validation_d2_dale_comparison.py",
        [
            "--csv_file", CSV_FILE,
            "--output_dir", output_dir,
            "--output_filename", "figure05_source",
            "--metric", "kld",
            "--dlvm_label_dim", "2",
        ],
        prelude_lines=[
            "import sys",
            "sys.path.insert(0, %r)" % str(REPO_ROOT / "src"),
            "from nmil_dlvm.analysis_utils import utils_plot as _utils_plot",
            "_original_plot_generic_comparison = _utils_plot.plot_generic_comparison",
            "def _plot_with_grid(*args, **kwargs):",
            "    kwargs.setdefault('show_grid', True)",
            "    return _original_plot_generic_comparison(*args, **kwargs)",
            "_utils_plot.plot_generic_comparison = _plot_with_grid",
        ],
        dry_run=dry_run,
    )
    if dry_run:
        return
    copy_pdf(output_dir / "figure05_source.pdf", FIGURE_NAME)


if __name__ == "__main__":
    generate()
