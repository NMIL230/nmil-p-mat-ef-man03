# fig08

Target wrapper

- `python manuscript_figures/figures/fig08.py`

Wrapper inputs

- The same upstream artifacts used by [fig07.md](fig07.md)

How the upstream artifacts are generated

1. Generate the D2 ground-truth session cache and the `N=240` synthetic observations exactly as shown in [fig07.md](fig07.md).

2. Generate the primary manuscript DALE run directory exactly as shown in [fig07.md](fig07.md):

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

3. Generate the figure PDF:

```bash
python manuscript_figures/figures/fig08.py
```
