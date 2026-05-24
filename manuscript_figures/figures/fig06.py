"""
Generate `Figure_06.pdf`.

Required base data on disk
- `artifacts/results/dale_runs/exp_c6_2d_dale_ps2/`
- one random-comparison run available at either `artifacts/results/dale_runs/exp_c8_2d_random/`
  or `artifacts/results/dale_runs/exp_vo4_2d_random/`
- Inside each experiment directory, per-session folders for the selected `LD2-*`
  sessions with data files matching `data/num_tests_run_*_update_w_data_session_*.pt`
- `TB` is not loaded from disk; the low-level plotting script constructs it as a synthetic
  fixed baseline distribution.
"""

from wrapper_utils import (
    copy_pdf,
    generated_output_dir,
    manuscript_dale_runs_dir,
    run_python_script,
    temporary_alias_root,
)


FIGURE_NAME = "Figure_06"


def generate(dry_run=False):
    output_dir = generated_output_dir("fig06")
    result_root = manuscript_dale_runs_dir()
    alias_map = {
        "exp_c6_2d_dale_ps2": [
            result_root / "exp_c6_2d_dale_ps2",
        ],
        "exp_c8_2d_random": [
            result_root / "exp_c8_2d_random",
            result_root / "exp_vo4_2d_random",
        ],
    }
    if dry_run:
        run_python_script(
            "src/nmil_dlvm/analysis/dale_simulation/plot_sampled_task_mix_by_session.py",
            [
                "--resultdir", result_root,
                "--output_dir", output_dir,
                "--include_2d",
                "--methods_x", "tb,random,ps2",
                "--no_png",
            ],
            dry_run=True,
        )
        return

    with temporary_alias_root(alias_map, "fig06_result_aliases") as aliased_result_root:
        run_python_script(
            "src/nmil_dlvm/analysis/dale_simulation/plot_sampled_task_mix_by_session.py",
            [
                "--resultdir", aliased_result_root,
                "--output_dir", output_dir,
                "--include_2d",
                "--methods_x", "tb,random,ps2",
                "--no_png",
            ],
        )
    if dry_run:
        return
    copy_pdf(output_dir / "compare_methods_2D_sessions_selected.pdf", FIGURE_NAME)


if __name__ == "__main__":
    generate()
