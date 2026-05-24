# figS04

Target wrapper

- `python manuscript_figures/figures/figS04.py`

Wrapper inputs

- `artifacts/analysis/dale_simulation/plot_data/post_run/validation/d2/dale_comparison_data.csv`

How the upstream artifacts are generated

1. Follow the validation 2D DALE pipeline in [fig05.md](fig05.md) to create:
   - the `exp_c5_2d_dale_ps0`, `exp_c6_2d_dale_ps2`, `exp_c7_2d_dale_ps4`, and `exp_c8_2d_random` run directories,
   - the aggregated CSV at `artifacts/analysis/dale_simulation/plot_data/post_run/validation/d2/dale_comparison_data.csv`.

2. Generate the figure PDF:

```bash
python manuscript_figures/figures/figS04.py
```
