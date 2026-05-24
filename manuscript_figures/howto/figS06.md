# figS06

Target wrapper

- `python manuscript_figures/figures/figS06.py`

Wrapper inputs

- `data/COLL10_SIM/all_data-best_mle_params_mpf100.pt`
- `artifacts/analysis/dlvm_imle_comparison/fitted_parameters/COLL10_SIM/`

How the upstream artifacts are generated

1. Train or reuse the manuscript 1D, 2D, and 3D training-set DLVM checkpoints using the recorded model-training hyperparameters:

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

2. Generate the training-set synthetic observations. The historical training fit script used the `N` grid `1 2 3 5 10 20 50 100 200 500`.

Example:

```bash
python src/nmil_dlvm/analysis/generate_synthetic_item_observations/generate_simulation_data.py \
  --N 50 \
  --ground_truth_param_file data/COLL10_SIM/all_data-best_mle_params_mpf100.pt \
  --use_n_per_task \
  --sampling_method random
```

3. For each `N` in that grid, fit IMLE and DLVM parameters for all three latent dimensions:

```bash
python src/nmil_dlvm/analysis/dlvm_imle_comparison/fit_dlvm_and_imle_models_to_data.py \
  --fit-imle \
  --fit-dlvm \
  --latent-dim 1 \
  --model-id wolverine-zoom-7298 \
  --max_N 50 \
  --synthetic_data_dir artifacts/analysis/generate_synthetic_item_observations/synthetic_data/COLL10_SIM/all_data-best_mle_params_mpf100 \
  --eval_dataset_type training_set
```

```bash
python src/nmil_dlvm/analysis/dlvm_imle_comparison/fit_dlvm_and_imle_models_to_data.py \
  --fit-imle \
  --fit-dlvm \
  --latent-dim 2 \
  --model-id mongoose-dive-7464 \
  --max_N 50 \
  --synthetic_data_dir artifacts/analysis/generate_synthetic_item_observations/synthetic_data/COLL10_SIM/all_data-best_mle_params_mpf100 \
  --eval_dataset_type training_set
```

```bash
python src/nmil_dlvm/analysis/dlvm_imle_comparison/fit_dlvm_and_imle_models_to_data.py \
  --fit-imle \
  --fit-dlvm \
  --latent-dim 3 \
  --model-id beaver-slide-5310 \
  --max_N 50 \
  --synthetic_data_dir artifacts/analysis/generate_synthetic_item_observations/synthetic_data/COLL10_SIM/all_data-best_mle_params_mpf100 \
  --eval_dataset_type training_set
```

4. Generate the figure PDF:

```bash
python manuscript_figures/figures/figS06.py
```

Notes

- Source training rows:
  - `artifacts/training/COLL10_SIM/runs_data_until_05242026.csv` for `wolverine-zoom-7298`
  - `artifacts/training/COLL10_SIM/runs_data_until_022425.csv` for `mongoose-dive-7464`
  - `artifacts/training/COLL10_SIM/runs_data_until_022425.csv` for `beaver-slide-5310`
