# Bayesian Distributional Models of Executive Functioning

This repository share the code necessary for regenerating the Figures 2–8 of the manuscipt "Bayesian Distributional Models of Executive Functioning". The
scripts live under `manuscript_figures/` and wrap the underlying analysis code in `src/analysis/`.
Everything else—data, models, utilities under `src/`. Run the
commands below from the repository root so relative paths resolve correctly.

## 1. Environment Setup
- Python 3.9 (or newer) with Conda recommended. Install Miniconda if you don't have it.
- Create and activate the project environment:
  ```bash
  conda env create -f src/environment.yml
  conda activate nmil-dlvm-nn
  ```
- Optional: install CUDA tooling if you plan to use a GPU; the defaults run on CPU.

## 2. Data Checklist
Download the project bundle from OSF and place the
contents exactly as follows:
- `data` → `src/data/` 
- `results/` → `src/results/`
- `analysis/<subfolders>` → matching paths under `src/analysis/`

Once those folders exist the figure scripts will find the required inputs automatically.

## 3. Generate Individual Figures
Each command writes `Figure_0X.(pdf|png)` to `manuscript_figures/generated_figures/`.

| Figure | Command |
| --- | --- |
| 2 | `python manuscript_figures/generate_all.py --figures fig2` |
| 3 | `python manuscript_figures/generate_all.py --figures fig3` |
| 4 | `python manuscript_figures/generate_all.py --figures fig4` |
| 5 | `python manuscript_figures/generate_all.py --figures fig5` |
| 6 | `python manuscript_figures/generate_all.py --figures fig6` |
| 7 | `python manuscript_figures/generate_all.py --figures fig7` |
| 8 | `python manuscript_figures/generate_all.py --figures fig8` |


## 4. Generate All Figures at Once
```bash
python manuscript_figures/generate_all.py
```
Add `--dry-run` to confirm all input files exist before running, or `--continue-on-error`
