# fig04

Target wrapper

- `python manuscript_figures/figures/fig04.py`

Wrapper inputs

- `artifacts/analysis/create_synthetic_individuals/synthetic_sessions_ground_truth/simulated_data/D1_synthetic_ground_truth_parameters_wolverine-zoom-7298.pt`
- `artifacts/analysis/create_synthetic_individuals/synthetic_sessions_ground_truth/simulated_data/D2_synthetic_ground_truth_parameters_mongoose-dive-7464.pt`
- `artifacts/analysis/create_synthetic_individuals/synthetic_sessions_ground_truth/simulated_data/D3_synthetic_ground_truth_parameters_beaver-slide-5310.pt`
- `artifacts/analysis/dlvm_imle_comparison/fitted_parameters/COLL10_SIM/`

How the upstream artifacts are generated

1. Train the manuscript validation model families with the recorded hyperparameters if you need fresh checkpoints:

```bash
PYTHONPATH=src python -m nmil_dlvm.cli.train_model \
  --held_out_session_ids 303run1 306run6 303run5 307run6 306run1 303run0 306run2 408run0 303run3 303run4 306run5 403run7 306run7 \
  --latent_dim 1 \
  --n_epochs 10000 \
  --n_samples 100 \
  --lr 0.001 \
  --kld_factor 0.01 \
  --run_mode run
```

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

```bash
PYTHONPATH=src python -m nmil_dlvm.cli.train_model \
  --held_out_session_ids 303run1 306run6 303run5 307run6 306run1 303run0 306run2 408run0 303run3 303run4 306run5 403run7 306run7 \
  --latent_dim 3 \
  --n_epochs 5000 \
  --n_samples 100 \
  --lr 0.001 \
  --kld_factor 0.01 \
  --run_mode run
```

2. Generate the exact validation ground-truth trio:

```bash
python src/nmil_dlvm/analysis/create_synthetic_individuals/create_ground_truth_sessions.py \
  --latent_dim 1 \
  --model-id wolverine-zoom-7298 \
  --output_dir artifacts/analysis/create_synthetic_individuals/synthetic_sessions_ground_truth
```

```bash
python src/nmil_dlvm/analysis/create_synthetic_individuals/create_ground_truth_sessions.py \
  --latent_dim 2 \
  --model-id mongoose-dive-7464 \
  --output_dir artifacts/analysis/create_synthetic_individuals/synthetic_sessions_ground_truth
```

```bash
python src/nmil_dlvm/analysis/create_synthetic_individuals/create_ground_truth_sessions.py \
  --latent_dim 3 \
  --model-id beaver-slide-5310 \
  --output_dir artifacts/analysis/create_synthetic_individuals/synthetic_sessions_ground_truth
```

3. For each ground-truth file above, generate the synthetic observation grids needed for the curve plots. The historical fit scripts used the `N` grid `1 2 3 5 10 20 50 100 200 500`.

Example for the 2D validation set:

```bash
python src/nmil_dlvm/analysis/generate_synthetic_item_observations/generate_simulation_data.py \
  --N 50 \
  --ground_truth_param_file artifacts/analysis/create_synthetic_individuals/synthetic_sessions_ground_truth/simulated_data/D2_synthetic_ground_truth_parameters_mongoose-dive-7464.pt \
  --use_n_per_task \
  --sampling_method random
```

4. For each `N` and each latent dimension, fit IMLE and DLVM parameters:

```bash
python src/nmil_dlvm/analysis/dlvm_imle_comparison/fit_dlvm_and_imle_models_to_data.py \
  --fit-imle \
  --fit-dlvm \
  --latent-dim 2 \
  --model-id mongoose-dive-7464 \
  --max_N 50 \
  --synthetic_data_dir artifacts/analysis/generate_synthetic_item_observations/synthetic_data/COLL10_SIM/D2_synthetic_ground_truth_parameters_mongoose-dive-7464 \
  --eval_dataset_type validation_simulated
```

Use the same command shape for `latent_dim 1` with `wolverine-zoom-7298` and for `latent_dim 3` with `beaver-slide-5310`.

5. Generate the figure PDF:

```bash
python manuscript_figures/figures/fig04.py
```

Notes

- Use the wrapper for exact manuscript reproduction. It temporarily isolates only the manuscript trio before calling `plot_merged_curves.py`.
- Source training rows:
  - `artifacts/training/COLL10_SIM/runs_data_until_05242026.csv` for `wolverine-zoom-7298`
  - `artifacts/training/COLL10_SIM/runs_data_until_022425.csv` for `mongoose-dive-7464`
  - `artifacts/training/COLL10_SIM/runs_data_until_022425.csv` for `beaver-slide-5310`
