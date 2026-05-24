# visualize_rmse.py

"""
visualize_rmse.py

This script computes the differences between meu_Z and meu_Z_prime across varying N
and generates visualizations to assess the grid search's correctness. It specifically examines the model's
behavior as N varies.
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import glob
import sys
import re
from pathlib import Path

# Adjust the system path to import from utils
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

# Import utility functions from data_distribution_utils.py
from nmil_dlvm.paths import ANALYSIS_ARTIFACTS_ROOT, MODEL_TRAINING_ANALYSIS_ROOT, SAVED_MODELS_ROOT, ensure_dir
from nmil_dlvm.utils.data_distribution_utils import (
    COMPUTE_DEVICE,
    prepare_data
)

# Import model definition
from nmil_dlvm.utils.variational_NN import variationalNN  # Adjust import path as necessary

# Import the built-in load_trained_model from active_learning_utils
from nmil_dlvm.utils.active_learning_utils import load_trained_model

# Configure Seaborn for better aesthetics
sns.set(style="whitegrid")


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


def generate_meu_z_prime(model, synthetic_data, logger):
    """
    Generates meu_Z_prime by passing synthetic data through the trained model.

    Args:
        model: Trained model.
        synthetic_data (torch.Tensor): Synthetic input data tensor of shape (synthetic_N, Features).
        logger: Logger for logging messages.

    Returns:
        torch.Tensor: meu_Z_prime tensor of shape (synthetic_N, D).
    """
    try:
        synthetic_data = synthetic_data.to(COMPUTE_DEVICE)
        with torch.no_grad():
            meu_z_prime = model.encode(synthetic_data)
        meu_z_prime = meu_z_prime.cpu()
        logger.info(f"Generated meu_Z_prime with shape {meu_z_prime.shape}.")
        return meu_z_prime
    except Exception as e:
        logger.error(f"Failed to generate meu_Z_prime: {e}")
        return None


def align_tensors(meu_Z, meu_Z_prime, participant_ids, logger, model_participant_ids):
    """
    Align meu_Z and meu_Z_prime tensors based on participant IDs.

    Args:
        meu_Z (torch.Tensor): Tensor of shape (M, D) from the model.
        meu_Z_prime (torch.Tensor): Tensor of shape (synthetic_N, D) generated from the model.
        participant_ids (list): List of participant IDs corresponding to rows in meu_Z_prime.
        logger (logging.Logger): Logger for logging messages.
        model_participant_ids (list): List of participant IDs corresponding to rows in meu_Z.

    Returns:
        tuple: (aligned_meu_Z, aligned_meu_Z_prime)
    """
    # Since synthetic_N and model_N are different, decide on the alignment strategy.
    # For demonstration, we'll compute the mean of meu_z_prime and compare it to the mean of meu_z.

    try:
        mean_meu_Z = meu_Z.mean(dim=0)
        mean_meu_Z_prime = meu_Z_prime.mean(dim=0)
        logger.info(f"Computed mean meu_Z: {mean_meu_Z}")
        logger.info(f"Computed mean meu_Z_prime: {mean_meu_Z_prime}")
        return mean_meu_Z, mean_meu_Z_prime
    except Exception as e:
        logger.error(f"Failed to align tensors: {e}")
        return None, None


def extract_latent_dim(model_filename):
    """
    Extracts the latent dimension (d) from the model filename.

    Args:
        model_filename (str): The filename of the model.

    Returns:
        int: The latent dimension.
    """
    pattern = re.compile(r'latentdim(\d+)')
    match = pattern.search(model_filename)
    if match:
        return int(match.group(1))
    else:
        return None


def extract_N(synthetic_filename):
    """
    Extracts the number of observations per metric (N) from the synthetic DLVM filename.

    Args:
        synthetic_filename (str): The filename of the synthetic DLVM parameters.

    Returns:
        int: The number of observations per metric.
    """
    pattern = re.compile(r'synthetic_dlvm_params_D\d+_N(\d+)\.pt')
    match = pattern.search(synthetic_filename)
    if match:
        return int(match.group(1))
    else:
        return None


def plot_difference_vs_n(n_values, difference_values, difference_se_values, output_path):
    """
    Plot the differences between meu_Z and meu_Z_prime vs N with error bars and highlight specific N values.

    Args:
        n_values (list): List of N values.
        difference_values (list): Corresponding mean difference values.
        difference_se_values (list): Corresponding SE of difference values.
        output_path (str): Path to save the plot.
    """
    plt.figure(figsize=(12, 8))
    sns.lineplot(x=n_values, y=difference_values, marker='o', label='Mean Difference')
    plt.errorbar(n_values, difference_values, yerr=difference_se_values, fmt='none', ecolor='lightgray', alpha=0.7)

    # Highlight specific N values if needed (e.g., N=300 and N=500)
    for highlight_n in [300, 500]:
        if highlight_n in n_values:
            idx = n_values.index(highlight_n)
            diff = difference_values[idx]
            se = difference_se_values[idx]
            plt.scatter(highlight_n, diff, color='red', s=150, zorder=5,
                        label=f'N={highlight_n}' if highlight_n == 300 else "")
            plt.errorbar(highlight_n, diff, yerr=se, fmt='none',
                         ecolor='red', capsize=5, linewidth=2)
            plt.text(highlight_n, diff, f'  N={highlight_n}', color='red',
                     fontsize=12, verticalalignment='bottom')

    plt.title('Difference between Training meu_Z and Predicted meu_Z_prime as N Increases', fontsize=20)
    plt.xlabel('Number of Observations per Metric (N)', fontsize=16)
    plt.ylabel('Mean Difference (meu_Z - meu_Z_prime)', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Difference vs N plot saved to '{output_path}'.")


def main():
    """
    Main function to compute differences between meu_Z and meu_Z_prime across varying N and visualize the results.
    """
    # Define dataset name and directories
    dataset_name = "COLL10_SIM"  # Replace with your actual dataset name

    # Define base directories
    base_dir = REPO_ROOT
    saved_models_dir = SAVED_MODELS_ROOT / dataset_name / "heldout_obsmulti"
    synthetic_params_dir = base_dir / "analysis" / "dlvm_imle_comparison"
    csv_dir = MODEL_TRAINING_ANALYSIS_ROOT / dataset_name
    meu_z_root = ANALYSIS_ARTIFACTS_ROOT / "meu_z"
    plots_dir = ensure_dir(meu_z_root / "plots")
    logs_dir = ensure_dir(meu_z_root / "logs")

    # Create directories if they don't exist
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Define logging
    log_file = os.path.join(logs_dir, f'visualize_difference_{dataset_name}.log')
    logger = setup_logging(log_file)
    logger.info("===== Starting Difference Visualization Script =====")

    # Define the four specific models with their filenames and filepaths
    models = [
        {
            "name": "LD1",
            "filename": "/variationalNN_relevant_only_latentdim1_fearless-tree-373.pt", 
            "filepath": saved_models_dir / "variationalNN_relevant_only_latentdim1_fearless-tree-373.pt"
        },
        {
            "name": "LD2",
            "filename": "/variationalNN_relevant_only_latentdim2_devoted-valley-280.pt",  
            "filepath": saved_models_dir / "variationalNN_relevant_only_latentdim2_devoted-valley-280.pt"
        },
        {
            "name": "LD3",
            "filename": "/variationalNN_relevant_only_latentdim3_honest-frost-2316.pt", 
            "filepath": saved_models_dir / "variationalNN_relevant_only_latentdim3_honest-frost-2316.pt"
        },
        {
            "name": "LD4",
            "filename": "/variationalNN_relevant_only_latentdim4_fast-breeze-372.pt", 
            "filepath": saved_models_dir / "variationalNN_relevant_only_latentdim4_fast-breeze-372.pt"
        }
    ]

    # Initialize list to store difference results
    difference_results = []

    # Iterate over each specified model
    for model_info in models:
        model_name = model_info["name"]
        model_filename = model_info["filename"]
        model_path = model_info["filepath"]

        logger.info(f"Processing model '{model_name}': {model_filename}")

        # Check if the model file exists
        if not os.path.isfile(model_path):
            logger.error(f"Model file '{model_path}' does not exist. Skipping '{model_name}'.")
            continue

        # Extract latent dimension 'd' from the model filename
        d = extract_latent_dim(model_filename)
        if d is None:
            logger.warning(f"Could not extract latent dimension 'd' from filename '{model_filename}'. Skipping '{model_name}'.")
            continue

        # Find the corresponding synthetic parameter file based on 'd' and synthetic N
        synthetic_files = glob.glob(os.path.join(synthetic_params_dir, f"synthetic_dlvm_params_D{d}_N*.pt"))
        if not synthetic_files:
            logger.error(f"No synthetic parameter files found for latent dimension 'd={d}' in '{synthetic_params_dir}'. Skipping '{model_name}'.")
            continue
        elif len(synthetic_files) > 1:
            logger.warning(f"Multiple synthetic parameter files found for latent dimension 'd={d}'. Using the first one: '{synthetic_files[0]}'.")
        
        synthetic_file = synthetic_files[0]
        synthetic_filename = os.path.basename(synthetic_file)
        synthetic_N = extract_N(synthetic_filename)
        if synthetic_N is None:
            logger.warning(f"Could not extract 'N' from synthetic filename '{synthetic_filename}'. Skipping '{model_name}'.")
            continue

        logger.info(f"Found synthetic parameter file '{synthetic_filename}' with d={d} and N={synthetic_N} for model '{model_name}'.")

        # Load the model using the built-in load_trained_model
        # Adjust 'model_output_dim' if necessary. Here, it's assumed to be 12 as per previous code.
        model = load_trained_model(d, model_path) 
        if model is None:
            logger.error(f"Failed to load model '{model_filename}'. Skipping '{model_name}'.")
            continue

        # Access meu_z from the model
        if not hasattr(model, 'meu_z'):
            logger.error(f"Model '{model_filename}' does not contain 'meu_z'. Skipping '{model_name}'.")
            continue

        meu_Z = model.meu_z  # Shape: (M, d)
        if meu_Z is None:
            logger.error(f"Model '{model_filename}' has 'meu_z' as None. Skipping '{model_name}'.")
            continue

        logger.info(f"Extracted meu_Z from model '{model_filename}' with shape {meu_Z.shape}.")

        # Ensure the model has 'participant_ids' attribute for alignment
        if not hasattr(model, 'participant_ids'):
            logger.error(f"Model '{model_filename}' does not have 'participant_ids' attribute. Skipping '{model_name}'.")
            continue

        # Retrieve held_out_session_ids from the corresponding CSV files
        heldout_session_ids = []
        csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
        for csv_file in csv_files:
            # Exclude malformed CSV files like 'debugs_data.csv'
            if os.path.basename(csv_file) == "debugs_data.csv":
                logger.warning(f"Excluding malformed CSV file '{csv_file}' from processing.")
                continue  # Skip processing this file

            try:
                logger.info(f"Reading CSV file '{csv_file}'...")
                runs_df = pd.read_csv(csv_file)

            except Exception as e:
                logger.error(f"Failed to read CSV file '{csv_file}': {e}. Skipping this file.")
                continue

            # Check if 'heldout_session_ids' or 'held_out_session_ids' column exists
            expected_column = 'heldout_session_ids'
            alternative_column = 'held_out_session_ids'
            if expected_column not in runs_df.columns and alternative_column not in runs_df.columns:
                logger.warning(f"CSV file '{csv_file}' does not contain '{expected_column}' or '{alternative_column}' column. Skipping.")
                continue

            # Determine the actual column name
            column_name = expected_column if expected_column in runs_df.columns else alternative_column

            # Find the row corresponding to the current model
            # Assuming 'file_name' in CSV matches the model's filename without path
            model_row = runs_df[runs_df['file_name'] == model_filename]
            if not model_row.empty:
                logger.info(f"Found model '{model_name}' in CSV file '{csv_file}'.")
                heldout_session_ids_raw = model_row.iloc[0][column_name]
                
                # Handle different formats of heldout_session_ids
                if isinstance(heldout_session_ids_raw, str):
                    heldout_session_ids = heldout_session_ids_raw.strip("[]").replace("'", "").split(", ")
                elif isinstance(heldout_session_ids_raw, list):
                    heldout_session_ids = heldout_session_ids_raw
                else:
                    logger.warning(f"Unrecognized format for held_out_session_ids in CSV file '{csv_file}' for model '{model_name}'. Skipping.")
                    heldout_session_ids = []
                break  # Assuming each model appears only once across CSVs

        if not heldout_session_ids:
            logger.warning(f"No held_out_session_ids found for model '{model_name}' in CSV files. Skipping.")
            continue

        logger.info(f"Retrieved {len(heldout_session_ids)} held_out_session_ids for model '{model_name}' and synthetic N={synthetic_N}.")

        # Use prepare_data to get participant IDs
        data_dict, metrics, participant_ids = prepare_data(
            heldout_obs_ids=heldout_session_ids,
            get_heldout_instead=False,
            normalize_times=True,
            remove_outliers=True
        )

        if not participant_ids:
            logger.warning(f"No participant IDs retrieved for model '{model_name}' and synthetic N={synthetic_N}. Skipping.")
            continue

        logger.info(f"Retrieved {len(participant_ids)} participant IDs for model '{model_name}' and synthetic N={synthetic_N}.")

        # Prepare synthetic data inputs directly without DataLoader
        try:
            # Initialize list to store feature tensors
            feature_tensors = []
            
            for metric in metrics:
                metric_data = data_dict[metric][0]  # data_dict[metric][0] contains 'data'
                
                if isinstance(metric_data, list):
                    metric_data = np.array(metric_data)
                
                if len(metric_data.shape) == 1:
                    # Single feature per observation point
                    tensor = torch.tensor(metric_data, dtype=torch.float32).unsqueeze(1)  # Shape: (synthetic_N, 1)
                elif len(metric_data.shape) == 2:
                    # Multiple features per observation point
                    tensor = torch.tensor(metric_data, dtype=torch.float32)  # Shape: (synthetic_N, Features)
                else:
                    logger.error(f"Unexpected shape for metric '{metric}': {metric_data.shape}. Skipping.")
                    raise ValueError(f"Invalid data shape for metric '{metric}'.")
                
                # Log the shape of each tensor
                logger.debug(f"Metric '{metric}' tensor shape: {tensor.shape}")
                
                # Handle different synthetic_N values
                if tensor.shape[0] != synthetic_N:
                    logger.warning(f"Metric '{metric}' has synthetic_N={tensor.shape[0]}, expected {synthetic_N}. Adjusting.")
                    if tensor.shape[0] > synthetic_N:
                        # Sample the first synthetic_N observations
                        tensor = tensor[:synthetic_N, :]
                        logger.info(f"Metric '{metric}' tensor adjusted to shape: {tensor.shape}")
                    else:
                        # Pad with zeros to match synthetic_N
                        padding = torch.zeros(synthetic_N - tensor.shape[0], tensor.shape[1])
                        tensor = torch.cat([tensor, padding], dim=0)
                        logger.info(f"Metric '{metric}' tensor padded to shape: {tensor.shape}")
                
                feature_tensors.append(tensor)
            
            # Concatenate along the feature dimension
            synthetic_inputs = torch.cat(feature_tensors, dim=1)  # Shape: (synthetic_N, Total Features)
            
            # Move inputs to the appropriate device
            synthetic_inputs = synthetic_inputs.to(COMPUTE_DEVICE)
            
            logger.info(f"Prepared synthetic inputs with shape {synthetic_inputs.shape} for model '{model_name}'.")
            
        except Exception as e:
            logger.error(f"Failed to prepare synthetic inputs for model '{model_name}': {e}. Skipping.")
            continue

        # Generate meu_Z_prime using the model
        meu_z_prime = generate_meu_z_prime(model, synthetic_inputs, logger)
        if meu_z_prime is None:
            logger.error(f"Failed to generate meu_Z_prime for model '{model_name}' and synthetic N={synthetic_N}'. Skipping.")
            continue

        # Align meu_Z and meu_Z_prime based on participant_ids
        # Since synthetic_N and model_N are different, we'll use an aggregate comparison
        aligned_meu_z, aligned_meu_z_prime = align_tensors(
            meu_Z=meu_Z,
            meu_Z_prime=meu_z_prime,
            participant_ids=participant_ids,
            logger=logger,
            model_participant_ids=model.participant_ids  # Ensure model has 'participant_ids'
        )
        if aligned_meu_z is None or aligned_meu_z_prime is None:
            logger.error(f"Failed to align tensors for model '{model_name}' and synthetic N={synthetic_N}'. Skipping.")
            continue

        # Compute Differences
        differences = aligned_meu_z - aligned_meu_z_prime  # Shape: (D, )

        # Compute statistics for differences
        mean_diff = differences.mean().item()
        se_diff = differences.std().item() / np.sqrt(differences.shape[0])

        logger.info(f"Model '{model_name}', synthetic N={synthetic_N}: Mean Difference={mean_diff:.4f}, SE_Difference={se_diff:.4f}")

        # Append results
        difference_results.append({
            'Model': model_name,
            'd': d,
            'Synthetic_N': synthetic_N,
            'Mean_Difference': mean_diff,
            'SE_Difference': se_diff
        })

    # After all models are processed
    # Convert difference_results to DataFrame and plot
    df_diff = pd.DataFrame(difference_results)
    if not df_diff.empty:
        # Save Difference results to CSV
        diff_csv_path = os.path.join(plots_dir, f"difference_results_{dataset_name}.csv")
        df_diff.to_csv(diff_csv_path, index=False)
        logger.info(f"Difference results saved to '{diff_csv_path}'.")

        # Generate Difference vs Synthetic N plot
        plot_output_path = os.path.join(plots_dir, f"difference_vs_synthetic_n_{dataset_name}.png")
        plot_difference_vs_n(
            n_values=df_diff['Synthetic_N'].tolist(),
            difference_values=df_diff['Mean_Difference'].tolist(),
            difference_se_values=df_diff['SE_Difference'].tolist(),
            output_path=plot_output_path
        )
    else:
        logger.error("No difference results to plot.")

    logger.info("===== Difference Visualization Script Completed Successfully =====")


if __name__ == "__main__":
    main()
