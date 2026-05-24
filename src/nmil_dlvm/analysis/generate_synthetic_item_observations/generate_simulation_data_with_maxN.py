# imle_sim_data.py

"""
imle_sim_data.py

This script performs IMLE simulations by generating synthetic data using pre-computed individual MLE parameters,
estimating parameters from the synthetic data, calculating evaluation metrics, and aggregating the results.
Outputs are written under artifacts/analysis/generate_synthetic_item_observations.
"""

import argparse, sys, os, time
import logging, random
from collections import Counter 
import numpy as np
import pandas as pd
import torch
from pathlib import Path

SCRIPT_DIR = os.fspath(Path(__file__).resolve().parent)
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

# ============================================
# Import Utilities
# ============================================

from nmil_dlvm.paths import ANALYSIS_ARTIFACTS_ROOT, data_dir, ensure_dir
from nmil_dlvm.utils.set_seed import set_seed


from nmil_dlvm.utils.data_distribution_utils import (
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

LOGS_DIR = ANALYSIS_ARTIFACTS_ROOT / "generate_synthetic_item_observations" / "logs"
SYNTHETIC_DATA_ROOT = ANALYSIS_ARTIFACTS_ROOT / "generate_synthetic_item_observations" / "synthetic_data"
DEFAULT_TRAINING_GROUND_TRUTH = data_dir(dataset_name) / "all_data-best_mle_params_mpf100.pt"

# Define logging file path
ensure_dir(LOGS_DIR)
log_file = os.fspath(LOGS_DIR / 'imle_simulation.log')
logger = setup_logging(log_file)
logger.info("===== Starting IMLE Simulation Script =====")


def resolve_ground_truth_params_path(raw_path=None):
    """Resolve ground-truth parameter inputs while preserving legacy basename behavior."""
    if raw_path is None:
        return Path(os.path.abspath(os.fspath(DEFAULT_TRAINING_GROUND_TRUTH)))

    candidate = Path(os.path.expanduser(raw_path))
    if candidate.is_absolute():
        return Path(os.path.abspath(os.fspath(candidate)))

    if candidate.parent == Path("."):
        return Path(os.path.abspath(os.fspath(data_dir(dataset_name) / candidate.name)))

    for base_dir in (Path.cwd(), Path(SCRIPT_DIR), REPO_ROOT):
        resolved = Path(os.path.abspath(os.fspath(base_dir / candidate)))
        if resolved.exists():
            return resolved

    return Path(os.path.abspath(os.fspath(Path.cwd() / candidate)))



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

def generate_synthetic_spans_summmarized(sim_run, sim_run_params, synthetic_run_id, allocation, synthetic_data):
    """
    Generate synthetic summarized metrics data based on a pre-defined allocation of observations.
    """
    for summary_metric in SUMMARIZED_METRICS:
        if SUMMARIZED_METRICS_METRIC_TYPES.get(summary_metric) != 'binarySpan':
            continue

        try:
            # Generate synthetic data based on the final allocation
            for length, num_obs in allocation.items():
                metric = f"{summary_metric}_correct_w_len_{length}"
                if metric not in CURR_METRICS_DICT.keys():
                    continue
                
                if num_obs == 0:
                    synthetic_data[metric] = []
                    continue

                metric_label = CURR_METRICS_DICT[metric].get("summary_metric_label", None)
                metric_params = sim_run_params.get(metric_label) if metric_label else None

                if metric_params is None:
                    logger.warning(f"MLE parameters not found for metric '{metric}' in simulation run '{sim_run}'")
                    synthetic_data[metric] = []
                    continue
                
                data_points, _ = generate_synthetic_dataset(
                    metric=metric,
                    metric_type='binarySpan',
                    N=num_obs,
                    metric_params=metric_params,
                    counts=1,
                    verbose=False
                )
                synthetic_data[metric] = data_points

        except Exception as e:
            logger.error(f"Error processing metric '{summary_metric}' in synthetic run '{synthetic_run_id}': {e}", exc_info=True)
            for length in range(2, 11):
                metric = f"{summary_metric}_correct_w_len_{length}"
                if metric in CURR_METRICS_DICT.keys():
                    synthetic_data[metric] = []
    
    # Handle non-span metrics if they are part of the summarized group
    for summary_metric in SUMMARIZED_METRICS:
        metric_type = SUMMARIZED_METRICS_METRIC_TYPES.get(summary_metric)
        if metric_type == 'binarySpan':
            continue
        
        # This part handles regular metrics if they were mixed in
        # (Assuming N for these is max_N)
        max_n_val = sum(allocation.values())
        metric_params = sim_run_params.get(summary_metric, None)
        if metric_params is None:
            synthetic_data[summary_metric] = []
            continue
        data_points, _ = generate_synthetic_dataset(
            metric=summary_metric,
            metric_type=metric_type,
            N=max_n_val,
            metric_params=metric_params,
            counts=1,
            verbose=False
        )
        synthetic_data[summary_metric] = data_points
        
    return synthetic_data

# ============================================
# Main Simulation Function
# ============================================

def main():

    parser = argparse.ArgumentParser(description="IMLE Simulation Script")
    parser.add_argument("--N", type=int, required=True, help="Maximum number of observations in the generation sequence.")
    parser.add_argument("--N_values", type=int, nargs='+', required=True, help="A list of N values to save datasets for.")
    parser.add_argument("--use_n_per_task", action="store_true", help="Use N per task (activates span distribution logic).")
    parser.add_argument("--num_synthetic_runs", type=int, default=1, help="Number of synthetic runs per original run.")
    parser.add_argument("--ground_truth_param_file", type=str, default=None, help="Path to the ground truth parameters file.")
    
    args = parser.parse_args()

    args.max_N = args.N # just for backward compatibility

    
    # --- Argument validation and path loading ---
    if any(n > args.max_N for n in args.N_values):
        logger.error("All --N_values must be less than or equal to --max_N.")
        sys.exit(1)
    

    ground_truth_params_path = os.fspath(resolve_ground_truth_params_path(args.ground_truth_param_file))
    
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    if not os.path.exists(ground_truth_params_path):
        logger.error(f"Ground truth parameters file not found at {ground_truth_params_path}")
        sys.exit(1)

    try:
        ground_truth_params = torch.load(ground_truth_params_path, map_location=COMPUTE_DEVICE)
        logger.info(f"Ground truth parameters loaded successfully from '{ground_truth_params_path}'.")
    except Exception as e:
        logger.error(f"Failed to load ground truth parameters from '{ground_truth_params_path}': {e}")
        sys.exit(1)
    # --- End setup ---

    all_synthetic_data_by_N = {n: {} for n in args.N_values}
    
    simulation_runs = list(ground_truth_params.keys())
    for sim_run in simulation_runs:
        logger.info(f"Running simulation for original run: {sim_run}")
        sim_run_params = ground_truth_params.get(sim_run)
        if sim_run_params is None: continue

        for sim_num in range(1, args.num_synthetic_runs + 1):
            set_seed(RANDOM_SEED + sim_num)
            synthetic_run_id = f"{sim_run}_sim{sim_num}"
            logger.debug(f"Generating data for synthetic run: {synthetic_run_id}")

            # =================================================================
            # === STEP 1: Generate the FULL dataset for max_N ONCE          ===
            # =================================================================
            full_dataset = {}
            master_sequence = []

            if args.use_n_per_task:
                # === Part A: Handle the distributed span tasks ===
                all_lengths = list(range(2, 11))
                first_len, last_len = 2, 10
                middle_lengths = [l for l in all_lengths if l not in [first_len, last_len]]
                random.shuffle(middle_lengths)
                base_allocation_order = [first_len, last_len] + middle_lengths
                for i in range(args.max_N):
                    master_sequence.append(base_allocation_order[i % len(base_allocation_order)])

                span_tasks = [task for task in SUMMARIZED_METRICS if SUMMARIZED_METRICS_METRIC_TYPES.get(task) == 'binarySpan']
                for task in span_tasks:
                    for l in all_lengths:
                        full_dataset[f"{task}_correct_w_len_{l}"] = []
                
                for target_length in master_sequence:
                    for task in span_tasks:
                        metric = f"{task}_correct_w_len_{target_length}"
                        metric_label = CURR_METRICS_DICT[metric].get("summary_metric_label")
                        metric_params = sim_run_params.get(metric_label) if metric_label else None
                        if metric_params is None: continue
                        data_point, _ = generate_synthetic_dataset(metric, 'binarySpan', 1, metric_params)
                        if data_point:
                            full_dataset[metric].append(data_point[0])
                
                # === Part B: Handle all other standard (non-span) tasks ===
                span_metric_names = list(full_dataset.keys())
                for metric in RELEVANT_METRICS:
                    if metric in span_metric_names: continue # Skip metrics already handled

                    metric_type = metric_types.get(metric)
                    if metric_type is None: continue
                    
                    metric_params = sim_run_params.get(metric, None)
                    if metric_params is None:
                        full_dataset[metric] = []
                        continue
                    
                    data_points, _ = generate_synthetic_dataset(metric, metric_type, args.max_N, metric_params)
                    full_dataset[metric] = data_points

            else:
                # Logic for the standard (non-distributed) case remains correct
                for metric in RELEVANT_METRICS:
                    metric_type = metric_types.get(metric)
                    if metric_type is None: continue
                    metric_params = sim_run_params.get(metric, None)
                    if metric_params is None:
                        full_dataset[metric] = []
                        continue
                    data_points, _ = generate_synthetic_dataset(metric, metric_type, args.max_N, metric_params)
                    full_dataset[metric] = data_points

            # =================================================================
            # === STEP 2: Distribute the full dataset into N-value buckets  ===
            # =================================================================
            for n_val in args.N_values:
                synthetic_data_for_n = {}
                if args.use_n_per_task:
                    allocation_for_n = Counter(master_sequence[:n_val])
                    for metric, full_data_list in full_dataset.items():
                        # Check if it's a distributed span metric
                        if metric.startswith(tuple(span_tasks)) and "correct_w_len" in metric:
                            length = int(metric.split('_')[-1])
                            num_points_for_n = allocation_for_n.get(length, 0)
                            synthetic_data_for_n[metric] = full_data_list[:num_points_for_n]
                        else: # Otherwise, it's a standard metric
                            synthetic_data_for_n[metric] = full_data_list[:n_val]
                else:
                    for metric, full_data_list in full_dataset.items():
                        synthetic_data_for_n[metric] = full_data_list[:n_val]
                
                all_synthetic_data_by_N[n_val][synthetic_run_id] = synthetic_data_for_n


    # --- Saving logic remains the same ---
    synthetic_output_dir = os.fspath(ensure_dir(SYNTHETIC_DATA_ROOT / dataset_name))
    for n_val, data_to_save in all_synthetic_data_by_N.items():
        synthetic_output_path = os.path.join(synthetic_output_dir, f'all_synthetic_data_N{n_val}.pt')
        try:
            torch.save(data_to_save, synthetic_output_path)
            logger.info(f"Synthetic data for N={n_val} saved successfully to '{synthetic_output_path}'")
        except Exception as e:
            logger.error(f"Failed to save synthetic data for N={n_val}: {e}")

    logger.info("===== IMLE Simulation Script Completed Successfully =====")

# ============================================
# Execute Main Function
# ============================================

if __name__ == "__main__":
    main()
