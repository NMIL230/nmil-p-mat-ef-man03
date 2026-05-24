#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$SCRIPT_DIR"

# N values to fit.
N_values=(50 100 200 500 1 2 3 5 10 20)

# Limit the number of parallel jobs.
max_jobs=1

base_synthetic_data_dir="${REPO_ROOT}/artifacts/analysis/generate_synthetic_item_observations/synthetic_data/COLL10"
d2_validation_w_outliers_synthetic_data_dir="$base_synthetic_data_dir/D2_synthetic_ground_truth_parameters_wolverine-race-9990"
model_path="${REPO_ROOT}/artifacts/models/COLL10/heldout_obs_none/variationalNN_relevant_only_latentdim2_wolverine-race-9990.pt"

latent_dims=(2)
current_jobs=0

for N in "${N_values[@]}"; do
    echo "Fitting models for N=$N (eval_dataset_type=validation_w_outliers_simulated)"
    (
        for latent_dim in "${latent_dims[@]}"; do
            python3 "${SCRIPT_DIR}/fit_dlvm_and_imle_models_to_data.py" \
                --fit-dlvm \
                --latent-dim "$latent_dim" \
                --model-path "$model_path" \
                --max_N "$N" \
                --synthetic_data_dir "$d2_validation_w_outliers_synthetic_data_dir" \
                --eval_dataset_type "validation_w_outliers_simulated"
        done
    ) &

    ((current_jobs++))
    if ((current_jobs >= max_jobs)); then
        wait -n
        ((current_jobs--))
    fi
done

wait
echo "Done. eval_dataset_type=validation_w_outliers_simulated"
