# figS11

Target wrapper

- `python manuscript_figures/figures/figS11.py`

Wrapper inputs

- `artifacts/analysis/dale_simulation/plot_data/post_run/validation_w_outliers/d2/dale_comparison_data.csv`

How the upstream artifacts are generated

1. Train or reuse the outlier-inclusive 2D model family. The recorded `cheetah-swoop-5350` runtime args are:

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

2. Generate the outlier-inclusive D2 ground truth and `N=240` synthetic data:

```bash
python src/nmil_dlvm/analysis/create_synthetic_individuals/create_ground_truth_sessions.py \
  --latent_dim 2 \
  --num_points 101 \
  --model-id cheetah-swoop-5350 \
  --output_dir artifacts/analysis/create_synthetic_individuals/synthetic_sessions_ground_truth
```

```bash
python src/nmil_dlvm/analysis/generate_synthetic_item_observations/generate_simulation_data.py \
  --N 240 \
  --ground_truth_param_file artifacts/analysis/create_synthetic_individuals/synthetic_sessions_ground_truth/simulated_data/D2_synthetic_ground_truth_parameters_cheetah-swoop-5350.pt \
  --use_n_per_task \
  --sampling_method random
```

3. Generate the four outlier-inclusive 2D DALE run families:
   - `exp_vo1_2d_dale_ps0`
   - `exp_vo2_2d_dale_ps2`
   - `exp_vo3_2d_dale_ps4`
   - `exp_vo4_2d_random`

Use the same `PYTHONPATH=src python -m nmil_dlvm.cli.run_dale` command shape as [fig05.md](fig05.md), but switch to:
   - `model_path artifacts/models/COLL10_SIM/heldout_obs_none/variationalNN_relevant_only_latentdim2_cheetah-swoop-5350.pt`,
   - the outlier-inclusive D2 ground-truth and synthetic-data paths,
   - `LD2-001` through `LD2-101` sessions,
   - the `exp_vo*` run names shown above.

4. Aggregate the runs:

```bash
python src/nmil_dlvm/analysis/dale_simulation/build_dale_comparison_csv.py \
  --dale_ps0_dir artifacts/results/dale_runs/exp_vo1_2d_dale_ps0 \
  --dale_ps1_dir artifacts/results/dale_runs/__no_ps1_for_validation_w_outliers_d2 \
  --dale_ps2_dir artifacts/results/dale_runs/exp_vo2_2d_dale_ps2 \
  --dale_ps4_dir artifacts/results/dale_runs/exp_vo3_2d_dale_ps4 \
  --random_dir artifacts/results/dale_runs/exp_vo4_2d_random \
  --output_dir artifacts/analysis/dale_simulation/plot_data/post_run/validation_w_outliers/d2 \
  --model_path artifacts/models/COLL10_SIM/heldout_obs_none/variationalNN_relevant_only_latentdim2_cheetah-swoop-5350.pt \
  --latent_dim 2 \
  --mle_params_file artifacts/analysis/create_synthetic_individuals/synthetic_sessions_ground_truth/simulated_data/D2_synthetic_ground_truth_parameters_cheetah-swoop-5350.pt \
  --synthetic_data_path artifacts/analysis/generate_synthetic_item_observations/synthetic_data/COLL10_SIM/D2_synthetic_ground_truth_parameters_cheetah-swoop-5350/all_synthetic_data_N240.pt \
  --session_number 101 \
  --max_iterations 240 \
  --imle_task_switch_interval 30 \
  --workers 4 \
  --stream_csv
```

5. Generate the figure PDF:

```bash
python manuscript_figures/figures/figS11.py
```

Notes

- Source training row: `artifacts/training/COLL10_SIM/runs_data.csv` for `cheetah-swoop-5350`.
- That row records empty runtime `held_out_session_ids` in `all_runtime_args`, while the summary columns also contain derived holdout/validation session lists. The command above follows the runtime-args payload.
