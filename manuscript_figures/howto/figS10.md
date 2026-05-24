# figS10

Target wrapper

- `python manuscript_figures/figures/figS10.py`

Wrapper inputs

- `artifacts/models/COLL10_SIM/**/variationalNN_relevant_only_latentdim2_cheetah-swoop-5350.pt`

How the artifact class is generated

1. Train the outlier-inclusive 2D model family with the recorded `cheetah-swoop-5350` runtime args:

```bash
PYTHONPATH=src python -m nmil_dlvm.cli.train_model \
  --held_out_session_ids \
  --latent_dim 2 \
  --n_epochs 8000 \
  --n_samples 100 \
  --lr 0.001 \
  --kld_factor 0.01 \
  --run_mode run \
  --keep_outliers
```

2. Build the 101-point latent-space mapping used by the supplemental figure:

```bash
python src/nmil_dlvm/analysis/create_synthetic_individuals/create_ground_truth_sessions.py \
  --latent_dim 2 \
  --num_points 101 \
  --model-id cheetah-swoop-5350 \
  --output_dir artifacts/analysis/create_synthetic_individuals/synthetic_sessions_ground_truth
```

3. Generate the figure PDF:

```bash
python manuscript_figures/figures/figS10.py
```

Notes

- The `held_out_obs_none` location for this manuscript model comes from passing an empty `--held_out_session_ids` list and keeping outliers in the training set.
- Source training row: `artifacts/training/COLL10_SIM/runs_data.csv` for `cheetah-swoop-5350`.
- That row records empty runtime `held_out_session_ids` in `all_runtime_args`, while the summary columns also contain derived holdout/validation session lists. The command above follows the runtime-args payload.
