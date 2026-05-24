#!/usr/bin/env python3
"""
create_custom_dataframe.py

This script generates a custom DataFrame from synthetic data, ground truth parameters, and IMLE parameters.
The DataFrame has the following structure:
user_session, task_label, data_type, metric, result, presentation_time

Usage:
    python create_custom_dataframe.py --N 1 \
        --synthetic_data_path "path/to/all_synthetic_data_N1.pt" \
        --output_csv "path/to/custom_dataframe.csv" \
        --log_dir "path/to/log_dir"
"""

import os
import sys
import argparse
import logging
import torch
import pandas as pd
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from nmil_dlvm.utils.data_distribution_utils import (
    RANDOM_SEED,
    RELEVANT_METRICS,
    CURR_METRICS_DICT,
    SUMMARIZED_METRICS_METRIC_TYPES,
    SUMMARIZED_METRICS
)
from nmil_dlvm.utils.set_seed import set_seed

def setup_logging(log_file):
    """
    Configure logging to output to both console and a file.

    Args:
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger.
    """
    logging.basicConfig(
        level=logging.INFO,  # Set to INFO for general logs
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )
    logger = logging.getLogger(__name__)
    return logger

def load_pt_file(file_path, logger, description=""):
    """
    Loads a .pt file using torch.load.

    Args:
        file_path (str): Path to the .pt file.
        logger (logging.Logger): Logger for logging messages.
        description (str): Description of the file being loaded.

    Returns:
        dict or list or other: Loaded data from the .pt file.
    """
    if not os.path.exists(file_path):
        logger.error(f"{description} file not found at {file_path}")
        return None
    try:
        data = torch.load(file_path, map_location='cpu')
        logger.info(f"Successfully loaded {description} from '{file_path}'.")
        return data
    except Exception as e:
        logger.error(f"Error loading {description} from '{file_path}': {e}")
        return None

def create_custom_dataframe(synthetic_data, logger):
    """
    Creates a custom DataFrame from synthetic data.

    Args:
        synthetic_data (dict): Synthetic data loaded from a .pt file.
        logger (logging.Logger): Logger for logging messages.

    Returns:
        pd.DataFrame: Custom DataFrame with columns ['user_session', 'task_label', 'data_type', 'metric', 'result', 'presentation_time'].
    """
    data = []
    for run_id, metrics_data in synthetic_data.items():
        logger.debug(f"Processing run '{run_id}'.")

        for metric_key, results in metrics_data.items():
            if metric_key not in RELEVANT_METRICS:
                logger.warning(f"Metric '{metric_key}' is not in RELEVANT_METRICS. Skipping.")
                continue

            # Extract task_label from metric_key (assuming it's the first part before '_')
            task_label = metric_key.split('_')[0]

            # Get data_type from CURR_METRICS_DICT
            data_type = CURR_METRICS_DICT.get(metric_key, {}).get("type", "Unknown")

            for result in results:

                data.append({
                    "user_session": run_id,
                    "task_label": task_label,
                    "data_type": data_type,
                    "metric": metric_key,
                    "result": result,
                })

    # Convert the data list into a DataFrame
    custom_df = pd.DataFrame(data)
    logger.info(f"Constructed custom DataFrame with {len(custom_df)} rows.")

    return custom_df

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Create Custom DataFrame from Synthetic Data")
    parser.add_argument(
        "--N", type=int, required=True, help="Number of synthetic observations per metric."
    )
    parser.add_argument(
        "--synthetic_data_path", type=str, required=True, help="Path to synthetic data .pt file."
    )
    parser.add_argument(
        "--output_csv", type=str, required=True, help="Path to save the custom DataFrame as CSV."
    )
    parser.add_argument(
        "--log_dir", type=str, required=True, help="Directory to save log files."
    )
    args = parser.parse_args()

    # Set up logging
    os.makedirs(args.log_dir, exist_ok=True)  # Ensure log directory exists
    log_file = os.path.join(args.log_dir, f'create_custom_dataframe_N{args.N}.log')
    logger = setup_logging(log_file)
    logger.info("===== Starting Custom DataFrame Creation =====")

    # Set random seed for reproducibility
    set_seed(RANDOM_SEED)

    # Load synthetic data
    synthetic_data = load_pt_file(
        file_path=args.synthetic_data_path,
        logger=logger,
        description="Synthetic data"
    )
    if synthetic_data is None:
        logger.error("Synthetic data loading failed. Exiting.")
        sys.exit(1)

    # Create the custom DataFrame
    custom_df = create_custom_dataframe(
        synthetic_data=synthetic_data,
        logger=logger
    )

    # Save the DataFrame to CSV
    try:
        custom_df.to_csv(args.output_csv, index=False)
        logger.info(f"Custom DataFrame saved to '{args.output_csv}'.")
    except Exception as e:
        logger.error(f"Failed to save DataFrame to '{args.output_csv}': {e}")
        sys.exit(1)

    logger.info("===== Custom DataFrame Creation Completed Successfully =====")

if __name__ == '__main__':
    main()
