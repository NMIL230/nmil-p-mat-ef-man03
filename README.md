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
Download the required data from [OSF](https://osf.io/ynbdr/files/osfstorage) and place the
contents exactly as follows:
- `data` → `src/data/` 
- extract the zip files (sim_runs_part_A.zip and 
 sim_runs_part_B.zip) and place their contents into `results/dale_sim_runs` into `src/results/dale_sim_runs`.
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

## 5. Troubleshooting
- Missing OSF data assets – run `python manuscript_figures/generate_all.py --dry-run` to list absent inputs, then confirm the OSF archives are fully extracted into `src/data/`, `src/results/dale_sim_runs/`, and the matching `src/analysis/` subfolders.
- Environment activation failures – ensure `conda activate nmil-dlvm-nn` succeeds; if the environment is missing, recreate it with `conda env create -f src/environment.yml`.
- CUDA-related crashes – skip GPU options and stay on the default CPU setup, or reinstall matching CUDA toolkit and drivers before rerunning the figure commands.
- Figure generation stops early – use `--continue-on-error` to identify the failing figure, then rerun the specific `--figures figX` invocation once the underlying data or model configuration is fixed.
- Large intermediate caches – delete stale outputs under `manuscript_figures/generated_inputs/` when they grow too large; the scripts rebuild them automatically on the next execution.
