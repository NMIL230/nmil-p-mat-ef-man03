Manuscript figures Reproduction

Thin wrappers around analysis scripts in `src/`. Each wrapper generates a figure PDF and copies it to `manuscript_figures/figures/`.

Scope
- Figure 1 is an illustration and thus is not reproducible from this repository. It is intentionally omitted.
- Main figures: Figure 02–08.
- Supplemental figures: Figure S01–S11.

Usage
- All figures:   python3 manuscript_figures/figures/generate_all.py
- Subset:        python3 manuscript_figures/figures/generate_all.py --figures fig02 fig05 figS04
- Dry-run:       python3 manuscript_figures/figures/generate_all.py --dry-run
- Single figure: python3 manuscript_figures/figures/fig02.py

Dependencies
- Python 3, conda environment defined in `environment.yml` (`nmil-dlvm-nn`).
- Local data expected under `data/`, `artifacts/models/`, `artifacts/analysis/`, and `artifacts/results/`.

The artificats can be downloaded from the OSF repository. `manuscript_figures/howto/` shows how these could be regenerated per-figure but it would take sometime - we recommend using the already provided artifacts.
