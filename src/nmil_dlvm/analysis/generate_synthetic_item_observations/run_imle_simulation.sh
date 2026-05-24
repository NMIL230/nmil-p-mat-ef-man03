#!/bin/bash
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CURRENT_DIR="${SCRIPT_DIR}"
# # Array of N values
# N_values=(1 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 240 500)
# CURRENT_DIR=$(pwd)
# BASE_GROUND_TRUTH_DIR="$CURRENT_DIR/../create_synthetic_individuals/synthetic_sessions_ground_truth/simulated_data"
# mle_params_file_d2="$BASE_GROUND_TRUTH_DIR/D2_synthetic_ground_truth_parameters_mongoose-dive-7464.pt"
# mle_params_file_d3="$BASE_GROUND_TRUTH_DIR/D3_synthetic_ground_truth_parameters_beaver-slide-5310.pt"
# mle_params_file_d1="$BASE_GROUND_TRUTH_DIR/D1_synthetic_ground_truth_parameters_wolverine-zoom-7298.pt"

# # Loop through each N value and run the command to generate the synthetic data for the validation data
# for N in "${N_values[@]}"; do
#     for mle_params_file in "$mle_params_file_d2" "$mle_params_file_d3" "$mle_params_file_d1"; do
#         python generate_simulation_data.py --N "$N" --ground_truth_param_file "$mle_params_file" --use_n_per_task --sampling_method "random"
#     done
# done

# # now generate for the training data
# mle_params_file_d2="../../data/COLL10/all_data-best_mle_params_mpf100.pt"

# N_values=( 1 2 3 10 20 50 100 200 500 240)

# for N in "${N_values[@]}"; do
#     python generate_simulation_data.py --N "$N" --ground_truth_param_file "$mle_params_file_d2" --use_n_per_task --sampling_method "random"
# done

# now generate for the validation model trained with outliers
VALIDATION_W_OUTLIERS_MODEL_ID="${VALIDATION_W_OUTLIERS_MODEL_ID:-cheetah-swoop-5350}"
N_values=( 1 2 3 10 20 50 100 200 500 240)
BASE_GROUND_TRUTH_DIR="$REPO_ROOT/artifacts/analysis/create_synthetic_individuals/synthetic_sessions_ground_truth/simulated_data"
mle_params_file_d2="$BASE_GROUND_TRUTH_DIR/D2_synthetic_ground_truth_parameters_${VALIDATION_W_OUTLIERS_MODEL_ID}.pt"

# Loop through each N value and run the command to generate the synthetic data for the validation data
for N in "${N_values[@]}"; do
    python "${SCRIPT_DIR}/generate_simulation_data.py" --N "$N" --ground_truth_param_file "$mle_params_file_d2" --use_n_per_task --sampling_method "random"
done
