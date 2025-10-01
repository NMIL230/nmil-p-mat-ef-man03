Figures 2–8 generator

Overview
- This folder provides a clean, unified CLI for generating Figures 2–8.
- Each figure has a small wrapper that calls the corresponding low-level script so you don’t have to remember arguments.

Quick start
- Example (Figure 2) using the in-repo defaults:
  python runner.py fig2 --dale-run-id exp_c6_2d_dale_ps2 --max-length-to-plot 100

- You can pass extra args to the low-level script after a “--” separator:
  python runner.py fig2 --dale-run-id exp_c6_2d_dale_ps2 --max-length-to-plot 100 -- --some-low-level-flag value

- Dry-run to see the command without executing:
  python runner.py fig2 --dale-run-id exp_c6_2d_dale_ps2 --max-length-to-plot 100 --dry-run

Notes
- output-dir is optional and only used if your low-level scripts support an output flag; pass it via the “-- …” passthrough if needed.
- For Figures 3–8:
  - Edit fig3.py … fig8.py to set the low-level script name and arguments if you need custom behavior.
  - Then call:
    python runner.py figN [figure-specific-args] [-- passthrough]
- If you want to test alternate copies of the low-level scripts, add `--scripts-root /path/to/scripts` to override the default `../src/analysis` tree.
