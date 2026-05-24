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
d2_synthetic_data_dir="$base_synthetic_data_dir/D2_synthetic_ground_truth_parameters_mongoose-dive-7464"
d3_synthetic_data_dir="$base_synthetic_data_dir/D3_synthetic_ground_truth_parameters_beaver-slide-5310"
d1_synthetic_data_dir="$base_synthetic_data_dir/D1_synthetic_ground_truth_parameters_wolverine-zoom-7298"

latent_dims=(1 2 3)
current_jobs=0

for N in "${N_values[@]}"; do
    echo "Fitting models for N=$N (eval_dataset_type=validation_simulated)"
    (
        for latent_dim in "${latent_dims[@]}"; do
            synthetic_data_dir=""
            model_id=""

            case "$latent_dim" in
                1)
                    model_id="wolverine-zoom-7298"
                    synthetic_data_dir="$d1_synthetic_data_dir"
                    ;;
                2)
                    model_id="mongoose-dive-7464"
                    synthetic_data_dir="$d2_synthetic_data_dir"
                    ;;
                3)
                    model_id="beaver-slide-5310"
                    synthetic_data_dir="$d3_synthetic_data_dir"
                    ;;
                *)
                    echo "Unsupported latent_dim: $latent_dim" >&2
                    exit 1
                    ;;
            esac

            python3 "${SCRIPT_DIR}/fit_dlvm_and_imle_models_to_data.py" \
                --fit-dlvm \
                --latent-dim "$latent_dim" \
                --model-id "$model_id" \
                --max_N "$N" \
                --synthetic_data_dir "$synthetic_data_dir" \
                --eval_dataset_type "validation_simulated"
        done
    ) &

    ((current_jobs++))
    if ((current_jobs >= max_jobs)); then
        wait -n
        ((current_jobs--))
    fi
done

wait
echo "Done. eval_dataset_type=validation_simulated"
