# fig06

Target wrapper

- `python manuscript_figures/figures/fig06.py`

Wrapper inputs

- `artifacts/results/dale_runs/exp_c6_2d_dale_ps2/`
- `artifacts/results/dale_runs/exp_c8_2d_random/` or `artifacts/results/dale_runs/exp_vo4_2d_random/`

How the upstream artifacts are generated

1. Make sure the 2D validation model, ground-truth parameters, and `all_synthetic_data_N240.pt` exist. The commands are listed in [fig04.md](fig04.md).

2. Generate the `exp_c6_2d_dale_ps2` and random-comparison run directories. Use the same `PYTHONPATH=src python -m nmil_dlvm.cli.run_dale` commands shown in [fig05.md](fig05.md) for `exp_c6_2d_dale_ps2` and `exp_c8_2d_random`.

3. Generate the figure PDF:

```bash
python manuscript_figures/figures/fig06.py
```

Notes

- `TB` is synthesized inside the plotting code; there is no separate run directory for it.
- If only `exp_vo4_2d_random` exists locally, the wrapper can alias it into the `exp_c8_2d_random` name expected by the older plotting script.
