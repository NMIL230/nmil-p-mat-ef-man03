# figS05

Target wrapper

- `python manuscript_figures/figures/figS05.py`

Wrapper inputs

- `artifacts/analysis/dale_simulation/plot_data/post_run/validation/d3/dale_comparison_data.csv`

How the upstream artifacts are generated

1. Train or reuse the manuscript 3D validation checkpoint. The recorded `beaver-slide-5310` training command is:

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

2. Generate the 3D validation ground truth and `N=240` synthetic data:

```bash
python src/nmil_dlvm/analysis/create_synthetic_individuals/create_ground_truth_sessions.py \
  --latent_dim 3 \
  --model-id beaver-slide-5310 \
  --output_dir artifacts/analysis/create_synthetic_individuals/synthetic_sessions_ground_truth
```

```bash
python src/nmil_dlvm/analysis/generate_synthetic_item_observations/generate_simulation_data.py \
  --N 240 \
  --ground_truth_param_file artifacts/analysis/create_synthetic_individuals/synthetic_sessions_ground_truth/simulated_data/D3_synthetic_ground_truth_parameters_beaver-slide-5310.pt \
  --use_n_per_task \
  --sampling_method random
```

3. Generate the four 3D validation DALE run families:
   - `exp_c1_3d_dale_ps0`
   - `exp_c2_3d_dale_ps2`
   - `exp_c3_3d_dale_ps4`
   - `exp_c4_3d_random`

Use the same `PYTHONPATH=src python -m nmil_dlvm.cli.run_dale` command shape as [fig05.md](fig05.md), but switch to:
   - `latent_dim 3`,
   - `beaver-slide-5310`,
   - the 3D ground-truth and synthetic-data paths,
   - `num_restarts 100`,
   - `LD3-001` through `LD3-088` sessions.

4. Aggregate the runs into the CSV consumed by the wrapper:

```bash
python src/nmil_dlvm/analysis/dale_simulation/build_dale_comparison_csv.py \
  --dale_ps0_dir artifacts/results/dale_runs/exp_c1_3d_dale_ps0 \
  --dale_ps1_dir artifacts/results/dale_runs/__no_ps1_for_validation_d3 \
  --dale_ps2_dir artifacts/results/dale_runs/exp_c2_3d_dale_ps2 \
  --dale_ps4_dir artifacts/results/dale_runs/exp_c3_3d_dale_ps4 \
  --random_dir artifacts/results/dale_runs/exp_c4_3d_random \
  --output_dir artifacts/analysis/dale_simulation/plot_data/post_run/validation/d3 \
  --model_path artifacts/models/COLL10_SIM/heldout_obsmulti/variationalNN_relevant_only_latentdim3_beaver-slide-5310.pt \
  --latent_dim 3 \
  --mle_params_file artifacts/analysis/create_synthetic_individuals/synthetic_sessions_ground_truth/simulated_data/D3_synthetic_ground_truth_parameters_beaver-slide-5310.pt \
  --synthetic_data_path artifacts/analysis/generate_synthetic_item_observations/synthetic_data/COLL10_SIM/D3_synthetic_ground_truth_parameters_beaver-slide-5310/all_synthetic_data_N240.pt \
  --session_number 88 \
  --max_iterations 240 \
  --imle_task_switch_interval 30 \
  --workers 4 \
  --stream_csv
```

5. Generate the figure PDF:

```bash
python manuscript_figures/figures/figS05.py
```

Notes

- Source training row: `artifacts/training/COLL10_SIM/runs_data_until_022425.csv` for `beaver-slide-5310`.
