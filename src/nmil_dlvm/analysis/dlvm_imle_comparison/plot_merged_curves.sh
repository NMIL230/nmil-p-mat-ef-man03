#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 [validation|training|both]"
    echo "       $0 --mode {validation|training|both}"
}

MODE="validation"
while [[ $# -gt 0 ]]; do
    case "$1" in
        validation|training|both)
            MODE="$1"
            shift
            ;;
        --mode|-m)
            MODE="${2:-}"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

case "$MODE" in
    validation|training|both) ;;
    *)
        echo "Invalid mode: $MODE"
        usage
        exit 1
        ;;
esac

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$SCRIPT_DIR"

GROUND_TRUTH_VALIDATION_DIR="${REPO_ROOT}/artifacts/analysis/create_synthetic_individuals/synthetic_sessions_ground_truth/simulated_data"
GROUND_TRUTH_TRAINING_DIR="${REPO_ROOT}/data/COLL10"
PARAMS_DIR="./fitted_parameters/COLL10"

run_validation() {
    echo "Running validation plots..."

    python3 "${SCRIPT_DIR}/plot_merged_curves.py" --ground_truth_dir "$GROUND_TRUTH_VALIDATION_DIR" --params_dir "$PARAMS_DIR" --metric "kld" --normalize_errors --plot_std_dev --eval_dataset_type "validation_simulated" --show_single_dlvm_plot --show_single_imle_plot
    python3 "${SCRIPT_DIR}/plot_merged_curves.py" --ground_truth_dir "$GROUND_TRUTH_VALIDATION_DIR" --params_dir "$PARAMS_DIR" --metric "kld" --normalize_errors --plot_std_dev --eval_dataset_type "validation_simulated" --show_single_imle_plot
    # python3 plot_merged_curves.py --ground_truth_dir "$GROUND_TRUTH_VALIDATION_DIR" --params_dir "$PARAMS_DIR" --metric "rmse" --plot_std_dev --normalize_errors --eval_dataset_type "validation_simulated"

    # Validation marginal medians (kept disabled by default).
    # d2_ground_truth_pt_file="$GROUND_TRUTH_VALIDATION_DIR/D2_synthetic_ground_truth_parameters_mongoose-dive-7464.pt"
    # d3_ground_truth_pt_file="$GROUND_TRUTH_VALIDATION_DIR/D3_synthetic_ground_truth_parameters_beaver-slide-5310.pt"
    # d1_ground_truth_pt_file="$GROUND_TRUTH_VALIDATION_DIR/D1_synthetic_ground_truth_parameters_wolverine-zoom-7298.pt"
    # d2_params_dir="$PARAMS_DIR/D2_synthetic_ground_truth_parameters_mongoose-dive-7464"
    # d3_params_dir="$PARAMS_DIR/D3_synthetic_ground_truth_parameters_beaver-slide-5310"
    # d1_params_dir="$PARAMS_DIR/D1_synthetic_ground_truth_parameters_wolverine-zoom-7298"
    # d2_synthetic_data_dir="../generate_synthetic_item_observations/synthetic_data/COLL10/D2_synthetic_ground_truth_parameters_mongoose-dive-7464"
    # d3_synthetic_data_dir="../generate_synthetic_item_observations/synthetic_data/COLL10/D3_synthetic_ground_truth_parameters_beaver-slide-5310"
    # d1_synthetic_data_dir="../generate_synthetic_item_observations/synthetic_data/COLL10/D1_synthetic_ground_truth_parameters_wolverine-zoom-7298"
    # python3 plot_marginal_median.py --ground_truth_pt_file "$d2_ground_truth_pt_file" --params_dir "$d2_params_dir" --metric "kld" --normalize_errors --synthetic_data_dir "$d2_synthetic_data_dir"
    # python3 plot_marginal_median.py --ground_truth_pt_file "$d3_ground_truth_pt_file" --params_dir "$d3_params_dir" --metric "rmse" --normalize_errors --synthetic_data_dir "$d3_synthetic_data_dir"
    # python3 plot_marginal_median.py --ground_truth_pt_file "$d1_ground_truth_pt_file" --params_dir "$d1_params_dir" --metric "rmse" --normalize_errors --synthetic_data_dir "$d1_synthetic_data_dir"
}

run_training() {
    echo "Running training plots..."

    python3 plot_merged_curves.py --ground_truth_dir "$GROUND_TRUTH_TRAINING_DIR" --params_dir "$PARAMS_DIR" --metric "kld" --normalize_errors --plot_std_dev --eval_dataset_type "training_set" --show_single_imle_plot
    python3 plot_merged_curves.py --ground_truth_dir "$GROUND_TRUTH_TRAINING_DIR" --params_dir "$PARAMS_DIR" --metric "rmse" --plot_std_dev --normalize_errors --eval_dataset_type "training_set" --show_single_imle_plot

    synthetic_data_dir="${REPO_ROOT}/artifacts/analysis/generate_synthetic_item_observations/synthetic_data/COLL10/all_data-best_mle_params_mpf100"
    ground_truth_pt_file="$GROUND_TRUTH_TRAINING_DIR/all_data-best_mle_params_mpf100.pt"
    d2_params_dir="$PARAMS_DIR/D2_all_data-best_mle_params_mpf100"

    python3 "${SCRIPT_DIR}/plot_marginal_median.py" --ground_truth_pt_file "$ground_truth_pt_file" --params_dir "$d2_params_dir" --metric "kld" --normalize_errors --synthetic_data_dir "$synthetic_data_dir" --eval_dataset_type "training_set"
    # python3 plot_marginal_median.py --ground_truth_pt_file "$ground_truth_pt_file" --params_dir "$d2_params_dir" --metric "rmse" --normalize_errors --synthetic_data_dir "$synthetic_data_dir" --eval_dataset_type "training_set"
}

case "$MODE" in
    validation)
        run_validation
        ;;
    training)
        run_training
        ;;
    both)
        run_validation
        run_training
        ;;
esac

echo "Done. Mode: $MODE"
