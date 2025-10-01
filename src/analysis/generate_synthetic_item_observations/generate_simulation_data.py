# imle_sim_data.py

"""
imle_sim_data.py

This script performs IMLE simulations by generating synthetic data using pre-computed individual MLE parameters,
estimating parameters from the synthetic data, calculating evaluation metrics, and aggregating the results.
All outputs (synthetic data, logs) are saved in the same directory as this script.
"""

import argparse
import hashlib
import sys
import os
import time
import logging
import numpy as np
import pandas as pd
import torch
import random

# Determine the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================
# Import Utilities
# ============================================

# Add the parent directory to sys.path to access 'utils' module
parent_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(parent_dir)

from utils.set_seed import set_seed

from utils.data_distribution_utils import (
    DATASET,
    RANDOM_SEED,
    RELEVANT_METRICS,
    CURR_METRICS_DICT,
    SUMMARIZED_METRICS_METRIC_TYPES,
    mle_params_to_dist,
    COMPUTE_DEVICE,
    SUMMARIZED_METRICS,
)

# ============================================
# Seed Initialization
# ============================================

# Set random seed for reproducibility
set_seed(RANDOM_SEED)


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
        level=logging.DEBUG,  # Set to DEBUG for detailed logs
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )
    logger = logging.getLogger(__name__)
    return logger


# ============================================
# Path Configuration
# ============================================


# Define dataset name
dataset_name = DATASET  # Replace with your actual dataset name

# Define logging file path
log_file = os.path.join(script_dir, 'imle_simulation.log')
logger = setup_logging(log_file)
logger.info("===== Starting IMLE Simulation Script =====")

# ============================================
# Metric Types Configuration
# ============================================

# Define metric types based on RELEVANT_METRICS
metric_types = {metric: CURR_METRICS_DICT[metric]["type"] for metric in RELEVANT_METRICS}

# Debug: Log metric types
logger.debug("Metric Types Mapping:")
for metric, mtype in metric_types.items():
    logger.debug(f"  {metric}: {mtype}")


# ============================================
# Synthetic Data Generation Function
# ============================================

def generate_synthetic_dataset(metric, metric_type, N, metric_params, counts=1, verbose=False):
    """
    Generate synthetic data points and return parameters for a given metric using ground truth parameters.

    Args:
    - metric (str): The name of the metric.
    - metric_type (str): The type of the metric (e.g., 'span', 'binary', 'timing').
    - N (int): Number of synthetic observations to generate.
    - metric_params (list or tensor): Parameters for the distribution.
    - counts (int): Number of trials (used for binary metrics).
    - verbose (bool): If True, logs detailed information.

    Returns:
    - data_points (list): Synthetic data points or parameters.
    - parameters (list or None): Parameters for multi-parameter metrics.
    - counts (list or None): Counts for binary metrics.
    """
    try:
        mle_dist_params = metric_params
        if mle_dist_params is None:
            logger.error(f"Parameters for metric '{metric}' are None.")
            return [], None, None  # Return empty data, parameters, and counts

        # Convert MLE parameters to a PyTorch distribution
        mle_dist = mle_params_to_dist(metric, mle_dist_params, metric_type, counts=counts, metrics_dict=CURR_METRICS_DICT)
        if mle_dist is None:
            logger.error(f"Could not create distribution for metric '{metric}'.")
            return [], None, None  # Return empty data, parameters, and counts

        # Generate synthetic data points
        simulated_data_points = mle_dist.sample((N,)).tolist()
        simulated_data_points = [int(round(p)) for p in simulated_data_points]
        return simulated_data_points, None

    except Exception as e:
        logger.error(f"Unexpected error for metric '{metric}': {e}")
        return [], None


# ============================================
# Main Simulation Function
# ============================================

def main():

    parser = argparse.ArgumentParser(description="IMLE Simulation Script")
    parser.add_argument(
        "--N", type=int, default=1, help="Number of synthetic observations to generate for each metric."
    )
    parser.add_argument("--use_n_per_task", action="store_true", help="Use N per task")
    parser.add_argument("--num_synthetic_runs", type=int, default=1, help="Number of synthetic runs per original run")
    parser.add_argument("--ground_truth_param_file", type=str, default=None, help="Path to the ground truth parameters file")
    parser.add_argument("--sampling_method", type=str, default="random", choices=["round-robin", "random"],
                        help="Sampling method to use when use_n_per_task is enabled.")

    args = parser.parse_args()
    # extract the N value from the command line arguments
    N = args.N

    if args.ground_truth_param_file is not None:
        candidate_path = args.ground_truth_param_file
        if os.path.isabs(candidate_path):
            ground_truth_params_path = os.path.abspath(candidate_path)
        else:
            # First, treat the provided path as relative to this script directory
            candidate_relative = os.path.abspath(os.path.join(script_dir, candidate_path))
            if os.path.exists(candidate_relative):
                ground_truth_params_path = candidate_relative
            else:
                ground_truth_params_path = os.path.abspath(
                    os.path.join(script_dir, '..', '..', 'data', dataset_name, candidate_path)
                )
    else:
        ground_truth_params_path = os.path.abspath(
            os.path.join(
                script_dir,
                '..',
                '..',
                'data',
                dataset_name,
                'D3_synthetic_ground_truth_parameters_beaver-slide-5310.pt',
            )
        )

    # add the parameter to the logger
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    # This function reads the number of synthetic observations (N) from the command line,
    # verifies the existence of the ground truth parameters file, loads the parameters,
    # and performs synthetic data generation for various N values. The generated synthetic
    # data is saved to .pt files.
    # Steps:
    # 1. Define the number of synthetic runs per original run.
    # 2. Define the number of synthetic observations per metric per simulation run.
    # 3. Verify the existence of the ground truth parameters file.
    # 4. Load aggregated ground truth parameters from a .pt file.
    # 5. Define simulation runs based on the loaded parameters.
    # 6. For each N value, generate synthetic data for each simulation run and metric.
    # 7. Save the generated synthetic data to .pt files.
    # Raises:
    #     SystemExit: If the ground truth parameters file is not found or fails to load.

    # use argparse to read the N value from the command line


    """
    Main function to perform synthetic data generation.
    """
    # Define Number of Synthetic Runs per Original Run
    num_synthetic_runs = args.num_synthetic_runs  # Generate 100 synthetic runs per original run

    # Define Number of Synthetic Observations per Metric per Simulation Run

    # Verify ground truth parameters file exists
    if not os.path.exists(ground_truth_params_path):
        logger.error(f"Ground truth parameters file not found at {ground_truth_params_path}")
        sys.exit(1)
 
    # extract the file name from the ground truth file name without the extension
    ground_truth_params_file_name = os.path.basename(ground_truth_params_path)
    ground_truth_params_file_name_without_extension = os.path.splitext(ground_truth_params_file_name)[0]
    logger.info(f"Ground truth parameters file name without extension: {ground_truth_params_file_name_without_extension}")
    # create a new directory for the synthetic data
    synthetic_data_dir = os.path.join(script_dir, 'synthetic_data', dataset_name, ground_truth_params_file_name_without_extension)
    os.makedirs(synthetic_data_dir, exist_ok=True)

    # Load Aggregated Ground Truth Parameters from .pt File
    try:
        ground_truth_params = torch.load(ground_truth_params_path, map_location=COMPUTE_DEVICE)
        logger.info(f"Ground truth parameters loaded successfully from '{ground_truth_params_path}'.")
    except Exception as e:
        logger.error(f"Failed to load ground truth parameters from '{ground_truth_params_path}': {e}")
        sys.exit(1)

    # Define Simulation Runs
    simulation_runs = list(ground_truth_params.keys())
    num_simulations = len(simulation_runs)
    logger.info(f"Number of simulation runs: {num_simulations}")

    logger.info(f"===== Starting synthetic data generation for N={N} =====")

    all_synthetic_data = {}

    for sim_run in simulation_runs:
        logger.info(f"Running simulation for original run: {sim_run}")

        sim_run_params = ground_truth_params.get(sim_run, None)
        if sim_run_params is None:
            logger.warning(f"Ground truth parameters for simulation run '{sim_run}' not found. Skipping.")
            continue

        for sim_num in range(1, num_synthetic_runs + 1):
            synthetic_run_id = f"{sim_run}_sim{sim_num}"
            run_id_hash = int(hashlib.sha256(synthetic_run_id.encode('utf-8')).hexdigest(), 16)
            unique_seed = (run_id_hash + RANDOM_SEED) % (10 ** 9)
            set_seed(unique_seed)
            logger.debug(f"Generating synthetic run: {synthetic_run_id}")

            # Initialize with empty lists for all relevant metrics to ensure all keys exist
            synthetic_data = {metric: [] for metric in RELEVANT_METRICS}

            if args.use_n_per_task:
                if args.sampling_method == 'random':
                    logger.info("Using random sampling per task.")
                    # Pass the initialized dict to be filled
                    synthetic_data = generate_synthetic_data_random(N, sim_run, sim_run_params, synthetic_run_id,
                                                                    synthetic_data)
                else:  # default to round-robin
                    logger.info("Using round-robin sampling per task.")
                    # Pass the initialized dict to be filled
                    synthetic_data = generate_synthetic_spans_summmarized(N, sim_run, sim_run_params, synthetic_run_id,
                                                                          synthetic_data)
            else:
                # Pass the initialized dict to be filled
                synthetic_data = generate_synthetic_metrics(N, sim_run, sim_run_params, synthetic_run_id,
                                                            synthetic_data)

            all_synthetic_data[synthetic_run_id] = synthetic_data

        synthetic_output_path = os.path.join(synthetic_data_dir, f'all_synthetic_data_N{N}.pt')

        try:
            torch.save(all_synthetic_data, synthetic_output_path)
            logger.info(f"Synthetic data saved successfully to '{synthetic_output_path}'")
        except Exception as e:
            logger.error(f"Failed to save synthetic data for N={N}: {e}")

        logger.info("===== Synthetic Data Generation Script Completed Successfully =====")


def generate_synthetic_metrics(N, sim_run, sim_run_params, synthetic_run_id, synthetic_data):
    """
    Generate synthetic metrics data for a given simulation run.

    This function processes a list of relevant metrics, generates synthetic data for each metric based on its type and parameters,
    and stores the results in the provided synthetic_data dictionary.

    Args:
        N (int): The number of data points to generate for each metric.
        sim_run (str): The identifier for the current simulation run.
        sim_run_params (dict): A dictionary containing parameters for the simulation run.
        synthetic_run_id (str): The identifier for the synthetic data generation run.
        synthetic_data (dict): A dictionary to store the generated synthetic data.

    Returns:
        dict: The updated synthetic_data dictionary with generated metrics data.

    Raises:
        Exception: If an error occurs during the processing of a metric, it is logged and the metric is set to None in the synthetic_data dictionary.

    Notes:
        - The function uses a predefined list of relevant metrics (RELEVANT_METRICS) and a dictionary of metric types (metric_types).
        - For 'binarySpan' metrics, the function retrieves parameters using a 'summary_metric_label' key.
        - The function handles different metric types ('span', 'timing', 'binary') and stores the generated data or parameters accordingly.
        - If metric parameters are not found or an error occurs, the metric is skipped and set to None in the synthetic_data dictionary.
    """
    for metric in RELEVANT_METRICS:
        logger.debug(f"Processing metric: {metric}")
        metric_type = metric_types.get(metric, None)
        if metric_type is None:
            logger.warning(f"Metric type for '{metric}' is not defined. Skipping.")
            synthetic_data[metric] = []
            continue

        try:
            if metric_type == 'binarySpan':
                metric_label = CURR_METRICS_DICT[metric].get("summary_metric_label", None)
                if metric_label:
                    metric_params = sim_run_params.get(metric_label, None)
                else:
                    logger.warning(f"'summary_metric_label' not defined for metric '{metric}'.")
                    metric_params = []
            else:
                metric_params = sim_run_params.get(metric, None)

            if metric_params is None:
                logger.warning(f"MLE parameters not found for metric '{metric}' in simulation run '{sim_run}'")
                synthetic_data[metric] = []
                continue

            data_points, _ = generate_synthetic_dataset(
                        metric=metric,
                        metric_type=metric_type,
                        N=N,
                        metric_params=metric_params,
                        counts=1,
                        verbose=False)

            synthetic_data[metric] = data_points

        except Exception as e:
            logger.error(f"Error processing metric '{metric}' in synthetic run '{synthetic_run_id}': {e}")
            synthetic_data[metric] = []
    return synthetic_data


# random sampling function
def generate_synthetic_data_random(N, sim_run, sim_run_params, synthetic_run_id, synthetic_data):
    """
    For each summarized metric, generate N data points.
    If metric is 'binarySpan', randomly sample a length (2-10) for each point.
    Otherwise, generate N points for the metric directly.
    """
    for summary_metric in SUMMARIZED_METRICS:
        metric_type = SUMMARIZED_METRICS_METRIC_TYPES.get(summary_metric, None)
        if metric_type is None:
            logger.warning(f"Metric type for '{summary_metric}' is not defined. Skipping.")
            continue

        try:
            # random sample for 'binarySpan'
            if metric_type == 'binarySpan':
                logger.debug(f"Random sampling for main task: {summary_metric}. Generating {N} points.")
                for _ in range(N):
                    random_length = random.randint(2, 10)
                    metric_to_sample = f"{summary_metric}_correct_w_len_{random_length}"

                    if metric_to_sample not in CURR_METRICS_DICT:
                        logger.warning(f"Constructed metric '{metric_to_sample}' is not valid. Skipping.")
                        continue

                    metric_label = CURR_METRICS_DICT[metric_to_sample].get("summary_metric_label")
                    metric_params = sim_run_params.get(metric_label) if metric_label else None

                    if metric_params is None:
                        logger.warning(f"MLE params not found for label '{metric_label}' in run '{sim_run}'")
                        continue

                    data_points, _ = generate_synthetic_dataset(
                        metric=metric_to_sample,
                        metric_type=metric_type,
                        N=1,
                        metric_params=metric_params,
                        counts=1,
                        verbose=False
                    )

                    if data_points:
                        synthetic_data[metric_to_sample].append(data_points[0])

            # ELSE：other tasks in normal
            else:
                logger.debug(f"Standard sampling for non-span task: {summary_metric}. Generating {N} points.")
                metric_params = sim_run_params.get(summary_metric, None)
                if metric_params is None:
                    logger.warning(
                        f"MLE parameters not found for metric '{summary_metric}' in simulation run '{sim_run}'")
                    continue

                data_points, _ = generate_synthetic_dataset(
                    metric=summary_metric,
                    metric_type=metric_type,
                    N=N,
                    metric_params=metric_params,
                    counts=1,
                    verbose=False
                )
                synthetic_data[summary_metric].extend(data_points)

        except Exception as e:
            logger.error(f"Error during random sampling for '{summary_metric}' in run '{synthetic_run_id}': {e}",
                         exc_info=True)

    return synthetic_data


def generate_synthetic_spans_summmarized(N, sim_run, sim_run_params, synthetic_run_id, synthetic_data):
    """
    Generate synthetic summarized metrics data for a given simulation run.

    This function processes a list of summarized metrics, generates synthetic data for each metric based on its type and parameters,
    and stores the results in the provided synthetic_data dictionary.

    Args:
        N (int): The number of data points to generate for each metric.
        sim_run (str): The identifier for the current simulation run.
        sim_run_params (dict): A dictionary containing parameters for the simulation run.
        synthetic_run_id (str): The identifier for the synthetic data generation run.
        synthetic_data (dict): A dictionary to store the generated synthetic data.

    Returns:
        dict: The updated synthetic_data dictionary with generated metrics data.

    Raises:
        Exception: If an error occurs during the processing of a metric, it is logged and the metric is set to None in the synthetic_data dictionary.

    Notes:
        - The function uses a predefined list of summarized metrics (SUMMARIZED_METRICS) and a dictionary of metric types (SUMMARIZED_METRICS_METRIC_TYPES).
        - For 'binarySpan' metrics, the function distributes N observations across lengths dynamically.
        - The function handles different metric types ('span', 'timing', 'binary') and stores the generated data or parameters accordingly.
        - If metric parameters are not found or an error occurs, the metric is skipped and set to None in the synthetic_data dictionary.
    """
    for summary_metric in SUMMARIZED_METRICS:
        logger.debug(f"Processing metric: {summary_metric}")
        metric_type = SUMMARIZED_METRICS_METRIC_TYPES.get(summary_metric, None)

        if metric_type is None:
            logger.warning(f"Metric type for '{summary_metric}' is not defined. Skipping.")
            if metric_type == 'binarySpan':
                for length in range(2, 11):

                    metric = f"{summary_metric}_correct_w_len_{length}"
                    if metric in CURR_METRICS_DICT.keys():
                        synthetic_data[metric] = []
            else:
                synthetic_data[summary_metric] = []
            continue

        try:
            if metric_type == 'binarySpan':
                # Distribute N observations across lengths (2 to 10) using Round-Robin.
                for length in range(2, 11):
                    metric = f"{summary_metric}_correct_w_len_{length}"
                    if metric not in CURR_METRICS_DICT.keys():
                        continue
                    metric_label = CURR_METRICS_DICT[metric].get("summary_metric_label", None)

                    # Determine the number of observations for this length based on N
                    num_obs = max(0, N // 9)
                    if length - 2 < N % 9:
                        num_obs += 1

                    if metric_label:
                        metric_params = sim_run_params.get(metric_label, None)
                    else:
                        logger.warning(f"'summary_metric_label' not defined for metric '{metric}'.")
                        metric_params = None

                    if metric_params is None:
                        logger.warning(f"MLE parameters not found for metric '{metric}' in simulation run '{sim_run}'")
                        synthetic_data[metric] = []
                        continue

                    if num_obs > 0:
                        data_points, _ = generate_synthetic_dataset(
                            metric=metric,
                            metric_type=metric_type,
                            N=num_obs,
                            metric_params=metric_params,
                            counts=1,
                            verbose=False
                        )
                        synthetic_data[metric] = data_points
                    else:
                        synthetic_data[metric] = []
            else:
                metric_params = sim_run_params.get(summary_metric, None)

                if metric_params is None:
                    logger.warning(f"MLE parameters not found for metric '{summary_metric}' in simulation run '{sim_run}'")
                    synthetic_data[summary_metric] = []
                    continue

                data_points, _ = generate_synthetic_dataset(
                    metric=summary_metric,
                    metric_type=metric_type,
                    N=N,
                    metric_params=metric_params,
                    counts=1,
                    verbose=False
                )
                synthetic_data[summary_metric] = data_points

        except Exception as e:
            logger.error(f"Error processing metric '{summary_metric}' in synthetic run '{synthetic_run_id}': {e}", exc_info=True)

            if metric_type == 'binarySpan':
                for length in range(2, 11):
                    metric = f"{summary_metric}_correct_w_len_{length}"
                    if metric in CURR_METRICS_DICT.keys():
                        synthetic_data[metric] = []
            else:
                synthetic_data[summary_metric] = []

    return synthetic_data

# ============================================
# Execute Main Function
# ============================================

if __name__ == "__main__":
    main()
