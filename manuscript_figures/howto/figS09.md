# figS09

Target wrapper

- `python manuscript_figures/figures/figS09.py`

Wrapper inputs

- `artifacts/analysis/dale_simulation/plot_data/post_run/training/d3/dale_comparison_data.csv`

How the upstream artifacts are generated

1. Generate the shared training synthetic data if needed:

```bash
python src/nmil_dlvm/analysis/generate_synthetic_item_observations/generate_simulation_data.py \
  --N 240 \
  --ground_truth_param_file data/COLL10_SIM/all_data-best_mle_params_mpf100.pt \
  --use_n_per_task \
  --sampling_method random
```

2. Generate the 3D training DALE run families:
   - `exp_t9_train_d3_dale_ps0`
   - `exp_t10_train_d3_dale_ps2`
   - `exp_t11_train_d3_dale_ps4`
   - `exp_t12_train_d3_random`

Use the same `PYTHONPATH=src python -m nmil_dlvm.cli.run_dale` command shape as [fig05.md](fig05.md), but switch to:
   - `latent_dim 3`,
   - `beaver-slide-5310`,
   - the training-set MLE and shared training synthetic-data paths,
   - `num_restarts 3`,
   - participant-ID shards from `data/COLL10_SIM/participant_ids_not_normed.csv`.

3. Aggregate the runs:

```bash
python src/nmil_dlvm/analysis/dale_simulation/build_dale_comparison_csv.py \
  --dale_ps0_dir artifacts/results/dale_runs/exp_t9_train_d3_dale_ps0 \
  --dale_ps1_dir artifacts/results/dale_runs/__no_ps1_for_training_d3 \
  --dale_ps2_dir artifacts/results/dale_runs/exp_t10_train_d3_dale_ps2 \
  --dale_ps4_dir artifacts/results/dale_runs/exp_t11_train_d3_dale_ps4 \
  --random_dir artifacts/results/dale_runs/exp_t12_train_d3_random \
  --output_dir artifacts/analysis/dale_simulation/plot_data/post_run/training/d3 \
  --model_path artifacts/models/COLL10_SIM/heldout_obsmulti/variationalNN_relevant_only_latentdim3_beaver-slide-5310.pt \
  --latent_dim 3 \
  --mle_params_file data/COLL10_SIM/all_data-best_mle_params_mpf100.pt \
  --synthetic_data_path artifacts/analysis/generate_synthetic_item_observations/synthetic_data/COLL10_SIM/all_data-best_mle_params_mpf100/all_synthetic_data_N240.pt \
  --session_number 88 \
  --max_iterations 240 \
  --imle_task_switch_interval 30 \
  --workers 4 \
  --stream_csv
```

4. Generate the figure PDF:

```bash
python manuscript_figures/figures/figS09.py
```
