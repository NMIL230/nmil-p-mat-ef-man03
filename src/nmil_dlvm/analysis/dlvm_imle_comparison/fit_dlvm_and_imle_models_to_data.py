# fit_mle_on_sy.py

"""
fit_mle_on_sy.py

This script fits MLE parameters to synthetic data using the get_mle_params_per_metric function.
It can also predict parameters using a DLVM model. You can choose to fit IMLE, DLVM, or both
by specifying command-line arguments. The script processes synthetic data files corresponding
to different N values, fits or predicts parameters for each synthetic run within those datasets,
and saves the results to new .pt files. Detailed progress and diagnostics are logged to help
monitor and troubleshoot the fitting process. All outputs (fitted parameters and logs) are saved
in the same directory as this script.
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
import time  # For timing operations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

# Import necessary modules
from nmil_dlvm.paths import ANALYSIS_ARTIFACTS_ROOT, SAVED_MODELS_ROOT, ensure_dir
from nmil_dlvm.utils.data_distribution_utils import CURR_METRICS_DICT, DATASET, RELEVANT_METRICS, load_trained_model
from nmil_dlvm.utils.grid_search_utils import predict_parameters_from_data
from nmil_dlvm.utils.mle_utils import get_mle_params_per_metric

# ============================================
# Logging Configuration
# ============================================

def setup_logging(log_file):
    """
    Configure logging to output to both console and a file.

    Args:
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger.
    """
    logging.basicConfig(
        level=logging.INFO,  # Change to DEBUG for more verbosity
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )
    logger = logging.getLogger()
    return logger

# ============================================
# Path Configuration
# ============================================

SCRIPT_DIR = os.fspath(Path(__file__).resolve().parent)
LOGS_DIR = ensure_dir(ANALYSIS_ARTIFACTS_ROOT / "dlvm_imle_comparison" / "logs")

# Define dataset name
dataset_name = DATASET

# Define paths using relative paths
# synthetic_data_dir = os.path.join(script_dir, 'synthetic_data', dataset_name)
# mle_params_dir = os.path.join(script_dir, 'synthetic_data', dataset_name, "param_fits")
# dlvm_params_dir = os.path.join(script_dir, 'synthetic_data', dataset_name, "param_fits")
# os.makedirs(mle_params_dir, exist_ok=True)
# os.makedirs(dlvm_params_dir, exist_ok=True)
plots_dir = os.fspath(ensure_dir(ANALYSIS_ARTIFACTS_ROOT / "dlvm_imle_comparison" / "plots" / dataset_name))

# Ensure the plots directory exists
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    print(f"Created directory: {plots_dir}")

# Define logging file path
log_file = os.fspath(LOGS_DIR / f"fit_mle_{dataset_name}.log")
logger = setup_logging(log_file)
logger.info("===== Starting Parameter Fitting Script =====")

# ============================================
# Helper Functions
# ============================================

def load_simulated_data(file_path, logger):
    """
    Load data from a .pt file.

    Args:
        file_path (str): Path to the .pt file.
        logger (logging.Logger): Logger for logging messages.

    Returns:
        dict: Loaded data.
    """
    if not os.path.exists(file_path):
        logger.error(f"Data file not found at {file_path}")
        raise FileNotFoundError(f"Data file not found at {file_path}")

    try:
        data = torch.load(file_path, map_location='cpu')
        logger.info(f"Successfully loaded data from '{file_path}' with {len(data)} runs.")
        return data
    except Exception as e:
        logger.error(f"Failed to load data from '{file_path}': {e}")
        raise

def save_fitted_params(data, file_path, logger):
    """
    Save data to a .pt file.

    Args:
        data (dict): Data to be saved.
        file_path (str): Path to the output .pt file.
        logger (logging.Logger): Logger for logging messages.
    """
    try:
        torch.save(data, file_path)
        logger.info(f"Results saved to '{file_path}'")
    except Exception as e:
        logger.error(f"Failed to save results to '{file_path}': {e}")
        raise

# ============================================
# Main Function
# ============================================

def main():
    """
    Main function to fit parameters for multiple N datasets based on command-line arguments.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Fit parameters to synthetic data.')
    parser.add_argument('--fit-imle', action='store_true', help='Fit IMLE parameters')
    parser.add_argument('--fit-dlvm', action='store_true', help='Predict parameters using DLVM')
    parser.add_argument('--compare-grid-search', action='store_true', help='Compare DLVM with and without grid search')
    parser.add_argument('--latent-dim', type=int, default=3, help='Latent dimension of the DLVM model (default: 3)')
    parser.add_argument('--model-path', type=str, default=None, help='Path to the DLVM model file')
    parser.add_argument("--model-id", type=str, default=None, help="Model ID to load")
    parser.add_argument("--max_N", type=int, default=1, help="Maximum N to process")
    parser.add_argument("--synthetic_data_dir", type=str, default=None, required=True, help="Path to the synthetic data directory")
    parser.add_argument(
        "--eval_dataset_type",
        type=str,
        default="validation_simulated",
        choices=["validation_simulated", "validation_w_outliers_simulated", "training_set"],
        help="Type of experiment",
    )
    args = parser.parse_args()

    synthetic_data_dir = args.synthetic_data_dir

    # get the directory name from the synthetic_data_dir
    synthetic_data_dir_name = os.path.basename(synthetic_data_dir)
    if args.eval_dataset_type == "training_set":
        # attach the latent dim to the synthetic data dir name to ensure that we don't mix up the fits of different latent dims
        synthetic_data_dir_name = f"D{args.latent_dim}_{synthetic_data_dir_name}"

    dlvm_params_dir = os.path.join(SCRIPT_DIR, 'fitted_parameters', dataset_name, synthetic_data_dir_name)
    os.makedirs(dlvm_params_dir, exist_ok=True)

    mle_params_dir = os.path.join(SCRIPT_DIR, 'fitted_parameters', dataset_name, synthetic_data_dir_name)
    os.makedirs(mle_params_dir, exist_ok=True)


    # Check if at least one fitting method is selected
    if not args.fit_imle and not args.fit_dlvm:
        logger.error("No fitting method selected. Use --fit-imle, --fit-dlvm, or both.")
        sys.exit(1)

    if args.compare_grid_search and not args.fit_dlvm:
        logger.warning("--compare-grid-search is specified without --fit-dlvm. The comparison will not run.")

    # Identify all synthetic_data_N{N}.pt files
    synthetic_files = [
        f for f in os.listdir(synthetic_data_dir)
        if f.startswith("all_synthetic_data_N") and f.endswith(".pt")
    ]
    if not synthetic_files:
        logger.error(f"No synthetic data files found in '{synthetic_data_dir}'.")
        sys.exit(1)

    logger.info(f"Found {len(synthetic_files)} synthetic data files to process.")

    # Load the DLVM model if needed
    if args.fit_dlvm:
        if args.model_id is not None:
            args.model_path = os.fspath(
                SAVED_MODELS_ROOT / DATASET / "heldout_obsmulti" /
                f"variationalNN_relevant_only_latentdim{args.latent_dim}_{args.model_id}.pt"
            )

        if args.model_path is None or not os.path.exists(args.model_path):
            logger.error("DLVM model path is not specified or does not exist. Use --model-path to specify the model file.")
            sys.exit(1)
        model = load_trained_model(args.latent_dim, model_path=args.model_path)
        model.eval()  # Set model to evaluation mode
        logger.info(f"Loaded DLVM model from '{args.model_path}'")

    # Iterate over each synthetic data file
    for synthetic_file in synthetic_files:
        # Extract N value from filename
        try:
            N_str = synthetic_file.replace("all_synthetic_data_N", "").replace(".pt", "")
            N = int(N_str)
            if N != args.max_N:
                logger.info(f"Skipping N={N} as it is not the maximum specified value of {args.max_N}.")
                continue
        except ValueError:
            logger.warning(f"Filename '{synthetic_file}' does not conform to 'all_synthetic_data_N{{N}}.pt' format. Skipping.")
            continue

        synthetic_data_path = os.path.join(synthetic_data_dir, synthetic_file)
        logger.info(f"Processing synthetic data file for N={N}: '{synthetic_data_path}'")

        # Load synthetic data for current N
        try:
            synthetic_data = load_simulated_data(synthetic_data_path, logger)
        except Exception as e:
            logger.error(f"Skipping N={N} due to loading error: {e}")
            continue

        # Initialize dictionaries to hold fitted parameters for current N
        synthetic_mle_params = {}
        synthetic_dlvm_params = {}
        synthetic_dlvm_params_grid_search = {}

        # Get total number of runs for logging
        total_runs = len(synthetic_data)
        logger.info(f"Total synthetic runs to process for N={N}: {total_runs}")

        # Define logging interval
        logging_interval = 100  # Log progress every 100 runs

        # Track processed runs to avoid duplicates
        processed_runs = set()

        # Iterate over each synthetic run
        for idx, (run_id, metrics_data) in enumerate(synthetic_data.items(), 1):

            # Check for duplicate run IDs
            if run_id in processed_runs:
                logger.warning(f"Duplicate run ID detected: '{run_id}'. Skipping.")
                continue
            processed_runs.add(run_id)

            logger.info(f"----- Processing Run '{run_id}' ({idx}/{total_runs}) -----")

            # Start timing the fitting process for this run
            start_time = time.time()

            try:
                # Fit IMLE parameters if selected
                if args.fit_imle:
                    mle_params = get_mle_params_per_metric(metrics_data, mpf=100, metrics=RELEVANT_METRICS)
                    synthetic_mle_params[run_id] = mle_params

                # Predict DLVM parameters if selected
                if args.fit_dlvm:
                    latent_dim = model.meu_z.shape[1]
                    if latent_dim == 1:
                        num_points = 3
                    elif latent_dim == 2:
                        num_points = 10
                    else: # latent_dim == 3
                        num_points = 100
                    # num_points = 1
                    if args.compare_grid_search:
                        # --- Without Grid Search ---
                        start_time_no_grid = time.time()
                        predicted_parameters_no_grid = predict_parameters_from_data(metrics_data, model, num_points=num_points, use_grid_search=False)
                        synthetic_dlvm_params[run_id] = predicted_parameters_no_grid
                        end_time_no_grid = time.time()
                        logger.info(f"DLVM without grid search for run '{run_id}' took {end_time_no_grid - start_time_no_grid:.2f} seconds.")

                        # --- With Grid Search ---
                        start_time_grid = time.time()
                        predicted_parameters_grid = predict_parameters_from_data(metrics_data, model, use_grid_search=True)
                        synthetic_dlvm_params_grid_search[run_id] = predicted_parameters_grid
                        end_time_grid = time.time()
                        logger.info(f"DLVM with grid search for run '{run_id}' took {end_time_grid - start_time_grid:.2f} seconds.")
                    else:
                        predicted_parameters = predict_parameters_from_data(metrics_data, model, num_points=num_points, use_grid_search=False)
                        synthetic_dlvm_params[run_id] = predicted_parameters

                # End timing
                end_time = time.time()
                elapsed_time = end_time - start_time

                logger.info(f"Successfully processed run '{run_id}' in {elapsed_time:.2f} seconds.")

            except Exception as e:
                logger.error(f"Error processing run '{run_id}': {e}", exc_info=True)
                continue  # Skip to the next run

            # Log progress at defined intervals
            if idx % logging_interval == 0 or idx == total_runs:
                logger.info(f"Processed {idx}/{total_runs} runs for N={N}.")

        # Save the results based on selected methods
        if args.fit_imle:
            # Define output path for fitted IMLE parameters
            output_path_imle = os.path.join(mle_params_dir, f"synthetic_mle_params_N{N}.pt")
            # Save the fitted IMLE parameters to a .pt file for current N
            try:
                save_fitted_params(synthetic_mle_params, output_path_imle, logger)
            except Exception as e:
                logger.error(f"Failed to save IMLE parameters for N={N}: {e}")
                continue  # Proceed to the next N

        if args.fit_dlvm:
            # Define output path for DLVM predicted parameters
            output_path_dlvm = os.path.join(dlvm_params_dir, f"synthetic_dlvm_params_gradient_descent_D{args.latent_dim}_N{N}.pt")
            # Save the DLVM predictions to a .pt file for current N
            try:
                save_fitted_params(synthetic_dlvm_params, output_path_dlvm, logger)
            except Exception as e:
                logger.error(f"Failed to save DLVM parameters for N={N}: {e}")
                continue

            if synthetic_dlvm_params_grid_search:
                output_path_dlvm_grid = os.path.join(dlvm_params_dir, f"synthetic_dlvm_params_grid_search_D{args.latent_dim}_N{N}.pt")
                try:
                    save_fitted_params(synthetic_dlvm_params_grid_search, output_path_dlvm_grid, logger)
                except Exception as e:
                    logger.error(f"Failed to save DLVM parameters with grid search for N={N}: {e}")
                    continue

        logger.info(f"===== Completed Processing for N={N} =====\n")

    logger.info("===== Parameter Fitting Process Completed Successfully =====")

# ============================================
# Execute Main Function
# ============================================

if __name__ == "__main__":
    main()
