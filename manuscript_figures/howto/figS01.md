# figS01

Target wrapper

- `python manuscript_figures/figures/figS01.py`

Wrapper inputs

- The same validation ground-truth trio and fitted-parameter tree used by [fig04.md](fig04.md)

How the upstream artifacts are generated

1. Follow the artifact-generation steps in [fig04.md](fig04.md) to create:
   - the `D1`, `D2`, and `D3` synthetic ground-truth parameter files,
   - the validation synthetic observation grids,
   - the validation fitted-parameter directories under `artifacts/analysis/dlvm_imle_comparison/fitted_parameters/COLL10_SIM/`.

2. Generate the figure PDF:

```bash
python manuscript_figures/figures/figS01.py
```

Notes

- As with `fig04`, the wrapper is the safest way to reproduce the exact manuscript input set when `simulated_data/` contains extra ground-truth files.
