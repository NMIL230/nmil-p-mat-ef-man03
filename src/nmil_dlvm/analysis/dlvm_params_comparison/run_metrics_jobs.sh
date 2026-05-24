#!/usr/bin/env bash
# run_metric_jobs.sh
set -euo pipefail

# --- Configuration ---
readonly MAX_N=50
# expand the MAX_N variable into the array
readonly Ns=(1 2 3 5 10 20 "$MAX_N")

readonly Rs=(1 5 10)
readonly LATENT_DIM=3
readonly MODEL_ID="honest-frost-2316"
readonly MAX_JOBS=2
readonly PY=python
readonly NUM_POINTS=(10 50 300)
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
readonly DATA_GEN_SCRIPT="${REPO_ROOT}/src/nmil_dlvm/analysis/generate_synthetic_item_observations/generate_simulation_data_with_maxN.py"

# --- Main Logic ---
# Kill all jobs on Ctrl‑C
trap 'echo "Caught SIGINT, terminating…"; kill 0; exit 1' SIGINT SIGTERM

echo "--- Generating all synthetic data files in a single run ---"
# Call the script only ONCE to ensure data consistency
"${PY}" "${DATA_GEN_SCRIPT}" \
    --N_values "${Ns[@]}" \
    --max_N "${MAX_N}" \
    --use_n_per_task
echo "✅ All synthetic data files generated."


echo -e "\n--- Launching all metric cache jobs ---"
for N in "${Ns[@]}"; do
    echo "▶ Launching metric job for N=${N}"
    "${PY}" "${SCRIPT_DIR}/build_metric_cache.py" \
        --Ns "${N}" \
        --latent_dim "${LATENT_DIM}" \
        --n_restarts "${Rs[@]}" \
        --model_id "${MODEL_ID}" \
        --num_points "${NUM_POINTS[@]}" &

    # throttle parallelism
    while (( $(jobs -pr | wc -l) >= MAX_JOBS )); do
        # wait for any job to finish
        wait -n
    done
done

wait # wait for all remaining background jobs
echo "✅ All metric-cache jobs finished."
