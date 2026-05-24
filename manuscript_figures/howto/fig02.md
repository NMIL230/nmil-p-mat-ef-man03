# fig02

Target wrapper

- `python manuscript_figures/figures/fig02.py`

Wrapper inputs

- `artifacts/models/COLL10_SIM/**/variationalNN_relevant_only_latentdim2_mongoose-dive-7464.pt`

How the artifact class is generated

1. Train a 2D heldout-obsmulti model with the recorded `mongoose-dive-7464` hyperparameters:

```bash
PYTHONPATH=src python -m nmil_dlvm.cli.train_model \
  --held_out_session_ids 303run1 306run6 303run5 307run6 306run1 303run0 306run2 408run0 303run3 303run4 306run5 403run7 306run7 \
  --latent_dim 2 \
  --n_epochs 10000 \
  --n_samples 100 \
  --lr 0.001 \
  --kld_factor 0.01 \
  --run_mode run
```

2. Build the latent-space parameter mapping and synthetic-session cache from the manuscript checkpoint:

```bash
python src/nmil_dlvm/analysis/create_synthetic_individuals/create_ground_truth_sessions.py \
  --latent_dim 2 \
  --model-id mongoose-dive-7464 \
  --output_dir artifacts/analysis/create_synthetic_individuals/synthetic_sessions_ground_truth
```

3. Generate the figure PDF:

```bash
python manuscript_figures/figures/fig02.py
```

Notes

- Fresh training runs receive a new auto-generated model ID. If you train a replacement model, call `create_ground_truth_sessions.py --model-path <checkpoint>` directly or update the wrapper input.
- Source training row: `artifacts/training/COLL10_SIM/runs_data_until_022425.csv` for `mongoose-dive-7464`.
