# fig05

Target wrapper

- `python manuscript_figures/figures/fig05.py`

Wrapper inputs

- `artifacts/analysis/dale_simulation/plot_data/post_run/validation/d2/dale_comparison_data.csv`

How the upstream artifacts are generated

1. Make sure the 2D validation model, ground-truth parameters, and `all_synthetic_data_N240.pt` exist. The commands are listed in [fig04.md](fig04.md).

2. Run the four 2D validation DALE experiments. The historical shell jobs only sharded the same `LD2-001` through `LD2-088` session list.

```bash
PYTHONPATH=src python -m nmil_dlvm.cli.run_dale \
  --run_name exp_c5_2d_dale_ps0 \
  --run_mode run \
  --test_budget 240 \
  --use_synthetic_data True \
  --num_restarts 10 \
  --latent_dim 2 \
  --model_path artifacts/models/COLL10_SIM/heldout_obsmulti/variationalNN_relevant_only_latentdim2_mongoose-dive-7464.pt \
  --mle_params_file artifacts/analysis/create_synthetic_individuals/synthetic_sessions_ground_truth/simulated_data/D2_synthetic_ground_truth_parameters_mongoose-dive-7464.pt \
  --synthetic_data_file artifacts/analysis/generate_synthetic_item_observations/synthetic_data/COLL10_SIM/D2_synthetic_ground_truth_parameters_mongoose-dive-7464/all_synthetic_data_N240.pt \
  --eval_test_session_ids LD2-001 LD2-002 LD2-003 ... LD2-088
```

```bash
PYTHONPATH=src python -m nmil_dlvm.cli.run_dale \
  --run_name exp_c6_2d_dale_ps2 \
  --run_mode run \
  --test_budget 240 \
  --use_synthetic_data True \
  --num_restarts 10 \
  --latent_dim 2 \
  --enable_primer_sequence True \
  --primer_sequence_task_repetitions 2 \
  --model_path artifacts/models/COLL10_SIM/heldout_obsmulti/variationalNN_relevant_only_latentdim2_mongoose-dive-7464.pt \
  --mle_params_file artifacts/analysis/create_synthetic_individuals/synthetic_sessions_ground_truth/simulated_data/D2_synthetic_ground_truth_parameters_mongoose-dive-7464.pt \
  --synthetic_data_file artifacts/analysis/generate_synthetic_item_observations/synthetic_data/COLL10_SIM/D2_synthetic_ground_truth_parameters_mongoose-dive-7464/all_synthetic_data_N240.pt \
  --eval_test_session_ids LD2-001 LD2-002 LD2-003 ... LD2-088
```

```bash
PYTHONPATH=src python -m nmil_dlvm.cli.run_dale \
  --run_name exp_c7_2d_dale_ps4 \
  --run_mode run \
  --test_budget 240 \
  --use_synthetic_data True \
  --num_restarts 10 \
  --latent_dim 2 \
  --enable_primer_sequence True \
  --primer_sequence_task_repetitions 4 \
  --model_path artifacts/models/COLL10_SIM/heldout_obsmulti/variationalNN_relevant_only_latentdim2_mongoose-dive-7464.pt \
  --mle_params_file artifacts/analysis/create_synthetic_individuals/synthetic_sessions_ground_truth/simulated_data/D2_synthetic_ground_truth_parameters_mongoose-dive-7464.pt \
  --synthetic_data_file artifacts/analysis/generate_synthetic_item_observations/synthetic_data/COLL10_SIM/D2_synthetic_ground_truth_parameters_mongoose-dive-7464/all_synthetic_data_N240.pt \
  --eval_test_session_ids LD2-001 LD2-002 LD2-003 ... LD2-088
```

```bash
PYTHONPATH=src python -m nmil_dlvm.cli.run_dale \
  --run_name exp_c8_2d_random \
  --run_mode run \
  --test_budget 240 \
  --use_synthetic_data True \
  --num_restarts 10 \
  --latent_dim 2 \
  --random_baseline True \
  --model_path artifacts/models/COLL10_SIM/heldout_obsmulti/variationalNN_relevant_only_latentdim2_mongoose-dive-7464.pt \
  --mle_params_file artifacts/analysis/create_synthetic_individuals/synthetic_sessions_ground_truth/simulated_data/D2_synthetic_ground_truth_parameters_mongoose-dive-7464.pt \
  --synthetic_data_file artifacts/analysis/generate_synthetic_item_observations/synthetic_data/COLL10_SIM/D2_synthetic_ground_truth_parameters_mongoose-dive-7464/all_synthetic_data_N240.pt \
  --eval_test_session_ids LD2-001 LD2-002 LD2-003 ... LD2-088
```

3. Aggregate the run directories into the CSV consumed by the figure wrapper:

```bash
python src/nmil_dlvm/analysis/dale_simulation/build_dale_comparison_csv.py \
  --dale_ps0_dir artifacts/results/dale_runs/exp_c5_2d_dale_ps0 \
  --dale_ps1_dir artifacts/results/dale_runs/__no_ps1_for_validation_d2 \
  --dale_ps2_dir artifacts/results/dale_runs/exp_c6_2d_dale_ps2 \
  --dale_ps4_dir artifacts/results/dale_runs/exp_c7_2d_dale_ps4 \
  --random_dir artifacts/results/dale_runs/exp_c8_2d_random \
  --output_dir artifacts/analysis/dale_simulation/plot_data/post_run/validation/d2 \
  --model_path artifacts/models/COLL10_SIM/heldout_obsmulti/variationalNN_relevant_only_latentdim2_mongoose-dive-7464.pt \
  --latent_dim 2 \
  --mle_params_file artifacts/analysis/create_synthetic_individuals/synthetic_sessions_ground_truth/simulated_data/D2_synthetic_ground_truth_parameters_mongoose-dive-7464.pt \
  --synthetic_data_path artifacts/analysis/generate_synthetic_item_observations/synthetic_data/COLL10_SIM/D2_synthetic_ground_truth_parameters_mongoose-dive-7464/all_synthetic_data_N240.pt \
  --session_number 88 \
  --max_iterations 240 \
  --imle_task_switch_interval 30 \
  --workers 4 \
  --stream_csv
```

4. Generate the figure PDF:

```bash
python manuscript_figures/figures/fig05.py
```

Notes

- The final wrapper is preferred over the raw plotting script because it enables the grid variant required by the manuscript.
