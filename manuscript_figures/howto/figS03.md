# figS03

Target wrapper

- `python manuscript_figures/figures/figS03.py`

Wrapper inputs

- `artifacts/analysis/dale_simulation/plot_data/post_run/validation/d1/dale_comparison_data.csv`

How the upstream artifacts are generated

1. Make sure the 1D validation model, ground-truth parameters, and `all_synthetic_data_N240.pt` exist. The manuscript 1D validation model was trained with:

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

2. Generate the 1D validation ground truth and `N=240` synthetic data:

```bash
python src/nmil_dlvm/analysis/create_synthetic_individuals/create_ground_truth_sessions.py \
  --latent_dim 1 \
  --model-id wolverine-zoom-7298 \
  --output_dir artifacts/analysis/create_synthetic_individuals/synthetic_sessions_ground_truth
```

```bash
python src/nmil_dlvm/analysis/generate_synthetic_item_observations/generate_simulation_data.py \
  --N 240 \
  --ground_truth_param_file artifacts/analysis/create_synthetic_individuals/synthetic_sessions_ground_truth/simulated_data/D1_synthetic_ground_truth_parameters_wolverine-zoom-7298.pt \
  --use_n_per_task \
  --sampling_method random
```

3. Generate the four 1D validation DALE run families:
   - `exp_d1_1d_dale_ps0`
   - `exp_d2_1d_dale_ps2`
   - `exp_d3_1d_dale_ps4`
   - `exp_d4_1d_random`

Use the same `PYTHONPATH=src python -m nmil_dlvm.cli.run_dale` command shape as [fig05.md](fig05.md), but switch to:
   - `latent_dim 1`,
   - `wolverine-zoom-7298`,
   - the `D1_synthetic_ground_truth_parameters_wolverine-zoom-7298.pt` ground truth,
   - the `D1_synthetic_ground_truth_parameters_wolverine-zoom-7298/all_synthetic_data_N240.pt` synthetic data,
   - `num_restarts 3`,
   - `LD1-001` through `LD1-088` sessions.

4. Aggregate the runs into the CSV consumed by the wrapper:

```bash
python src/nmil_dlvm/analysis/dale_simulation/build_dale_comparison_csv.py \
  --dale_ps0_dir artifacts/results/dale_runs/exp_d1_1d_dale_ps0 \
  --dale_ps1_dir artifacts/results/dale_runs/__no_ps1_for_validation_d1 \
  --dale_ps2_dir artifacts/results/dale_runs/exp_d2_1d_dale_ps2 \
  --dale_ps4_dir artifacts/results/dale_runs/exp_d3_1d_dale_ps4 \
  --random_dir artifacts/results/dale_runs/exp_d4_1d_random \
  --output_dir artifacts/analysis/dale_simulation/plot_data/post_run/validation/d1 \
  --model_path artifacts/models/COLL10_SIM/heldout_obsmulti/variationalNN_relevant_only_latentdim1_wolverine-zoom-7298.pt \
  --latent_dim 1 \
  --mle_params_file artifacts/analysis/create_synthetic_individuals/synthetic_sessions_ground_truth/simulated_data/D1_synthetic_ground_truth_parameters_wolverine-zoom-7298.pt \
  --synthetic_data_path artifacts/analysis/generate_synthetic_item_observations/synthetic_data/COLL10_SIM/D1_synthetic_ground_truth_parameters_wolverine-zoom-7298/all_synthetic_data_N240.pt \
  --session_number 88 \
  --max_iterations 240 \
  --imle_task_switch_interval 30 \
  --workers 4 \
  --stream_csv
```

5. Generate the figure PDF:

```bash
python manuscript_figures/figures/figS03.py
```

Notes

- Source training row: `artifacts/training/COLL10_SIM/runs_data_until_05242026.csv` for `wolverine-zoom-7298`.
