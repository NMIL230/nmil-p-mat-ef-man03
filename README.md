# Bayesian Distributional Models of Executive Functioning

This repository shares the code necessary for regenerating Figures 2–8 and supplemental figures of the manuscript "Bayesian Distributional Models of Executive Functioning". The figure scripts live under `manuscript_figures/figures/` and wrap the underlying analysis code in `src/nmil_dlvm/`. Everything else—data, models, analysis artifacts—lives under `data/`, `artifacts/`. Run the commands below from the repository root so relative paths resolve correctly.

## 1. Environment Setup
- Python 3.9 (or newer) with Conda recommended. Install Miniconda if you don't have it.
- Create and activate the project environment:
  ```bash
  conda env create -f environment.yml
  conda activate nmil-dlvm-nn
  ```
- Optional: install CUDA tooling if you plan to use a GPU; the defaults run on CPU.

## 2. Data Checklist
Download the required data from [OSF](https://osf.io/ynbdr/files/osfstorage) and uncompress them and place the contents exactly as follows:
- `data/` → `data/` at the project root
- `artifacts/`→ `artifacts/` at the project root

Once those folders exist the figure scripts will find the required inputs automatically.

## 3. Generate Individual Figures
Each command writes `Figure_XX.pdf` to `manuscript_figures/figures/`.

| Figure | Command |
| --- | --- |
| 2 | `python manuscript_figures/figures/generate_all.py --figures fig02` |
| 3 | `python manuscript_figures/figures/generate_all.py --figures fig03` |
| 4 | `python manuscript_figures/figures/generate_all.py --figures fig04` |
| 5 | `python manuscript_figures/figures/generate_all.py --figures fig05` |
| 6 | `python manuscript_figures/figures/generate_all.py --figures fig06` |
| 7 | `python manuscript_figures/figures/generate_all.py --figures fig07` |
| 8 | `python manuscript_figures/figures/generate_all.py --figures fig08` |
| S1 | `python manuscript_figures/figures/generate_all.py --figures figS01` |
| S2 | `python manuscript_figures/figures/generate_all.py --figures figS02` |
| S3 | `python manuscript_figures/figures/generate_all.py --figures figS03` |
| S4 | `python manuscript_figures/figures/generate_all.py --figures figS04` |
| S5 | `python manuscript_figures/figures/generate_all.py --figures figS05` |
| S6 | `python manuscript_figures/figures/generate_all.py --figures figS06` |
| S7 | `python manuscript_figures/figures/generate_all.py --figures figS07` |
| S8 | `python manuscript_figures/figures/generate_all.py --figures figS08` |
| S9 | `python manuscript_figures/figures/generate_all.py --figures figS09` |
| S10 | `python manuscript_figures/figures/generate_all.py --figures figS10` |
| S11 | `python manuscript_figures/figures/generate_all.py --figures figS11` |

## 4. Generate All Figures at Once
```bash
python manuscript_figures/figures/generate_all.py
```
Add `--dry-run` to confirm all input files exist before running, or `--continue-on-error` to keep going after a failure.

## 5. Troubleshooting
- **Missing OSF data assets** – run `python manuscript_figures/figures/generate_all.py --dry-run` to list absent inputs, then confirm the OSF archives are fully extracted into `data/COLL10_SIM/`, `artifacts/results/dale_runs/`, and the matching `artifacts/analysis/` subfolders.
- **Environment activation failures** – ensure `conda activate nmil-dlvm-nn` succeeds; if the environment is missing, recreate it with `conda env create -f environment.yml`.
- **CUDA-related crashes** – set `USE_GPU_DEVICE` to `no` in `configurations.json`, or reinstall matching CUDA toolkit and drivers before rerunning.
- **Figure generation stops early** – use `--continue-on-error` to identify the failing figure, then rerun the specific `--figures figXX` invocation once the underlying data or model configuration is fixed.
- **Large intermediate caches** – delete stale outputs under `manuscript_figures/generated/` when they grow too large; the scripts rebuild them automatically on the next execution.
