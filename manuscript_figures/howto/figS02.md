# figS02

Target wrapper

- `python manuscript_figures/figures/figS02.py`

Wrapper inputs

- The same D2 model, latent-variable cache, synthetic `N=240` file, and `exp_c6_2d_dale_ps2` run directory used by [fig07.md](fig07.md)

How the upstream artifacts are generated

1. Follow the artifact-generation steps in [fig07.md](fig07.md).

2. Generate the figure PDF:

```bash
python manuscript_figures/figures/figS02.py
```

Notes

- This wrapper uses the same trajectory source data as `fig07`, but requests the longer `max_length_to_plot=240` view.
