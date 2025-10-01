# plot_rmse.py

"""
plot_rmse.py

This script computes the Root Mean Squared Error (RMSE) between IMLE-fitted parameters from synthetic data
and ground truth parameters across varying N (number of observations per metric) and specific DLVM models.
It then plots RMSE against N for IMLE and all DLVM models on the same graph to visualize performance improvements.
Additionally, it generates separate RMSE vs N plots for each metric to provide detailed insights into each metric's performance.
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import re  # For regex operations
import argparse
import scipy.stats as stats  # Add this import for KL divergence calculations
# Configure Seaborn for better aesthetics (optional)
sns.set_theme(style="whitegrid")

# Determine the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory to sys.path to access 'utils' module
parent_dir = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.append(parent_dir)

from utils.data_distribution_utils import (OUTLIER_HELDOUT_SESSIONS, DATASET,SUMMARIZED_METRICS_MAIN_MOMENTS_LABEL_DICT, 
                                           SUMMARIZED_METRICS, SUMMARIZED_METRICS_METRIC_TYPES, mle_params_to_dist, CURR_METRICS_DICT) # Ensure this is defined in your utils
from utils.active_learning_utils import calculate_kld_given_metric, move_distribution_to_cuda

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
# Existing Functions
# ============================================

def compute_rmse(estimated_params, ground_truth_params, metrics, parameters, logger, num_bootstrap=1000,main_param_only =False, normalize_errors=False, rmse_metric_normalizer={}):
    """
    Compute RMSE and standard error between estimated parameters and ground truth parameters.

    Args:
        estimated_params (dict): Estimated parameters (IMLE or DLVM).
        ground_truth_params (dict): Ground truth parameters.
        metrics (list): List of metrics to compare.
        parameters (dict): Dictionary mapping metrics to parameter indices.
        logger (logging.Logger): Logger for logging messages.
        num_bootstrap (int): Number of bootstrap samples for estimating SE.

    Returns:
        tuple: (rmse, se_rmse)
    """
    errors = []
    count = 0
    for run_id, est_metrics in estimated_params.items():
        # Handle different run ID structures
        original_run_id = '_'.join(run_id.split('_')[:-1])
        gt_metrics = ground_truth_params.get(original_run_id, None)
        if gt_metrics is None:
            logger.warning(f"Original Run ID '{original_run_id}' not found in ground truth data. Skipping synthetic run '{run_id}'.")
            continue
        run_total_errors = []
        for metric in metrics:
            # print(run_id,metric,gt_metrics, parameters.get(metric, None),parameters.keys())
            metric_errors = []
            for param_idx in parameters.get(metric, []):
                if main_param_only and param_idx != 0:
                    continue
                est_param_list = est_metrics.get(metric, [])
                gt_param_list = gt_metrics.get(metric, [])

                if len(est_param_list) <= param_idx or len(gt_param_list) <= param_idx:
                    logger.warning(f"Missing parameter index {param_idx} for Run ID '{run_id}', Metric '{metric}'. Skipping this parameter.")
                    errors.append(np.nan)
                    count += 1
                    continue
            
                est_value = est_param_list[param_idx]
                gt_value = gt_param_list[param_idx]

                if est_value is None or gt_value is None:
                    logger.warning(f"Missing values for Run ID '{run_id}', Metric '{metric}', Parameter index '{param_idx}'. Skipping this parameter.")
                    errors.append(np.nan)
                    count += 1
                    continue

                # Ensure values are numbers
                try:
                    est_value = float(est_value)
                    gt_value = float(gt_value)
                    error = est_value - gt_value # compute error
                    normalizer = rmse_metric_normalizer.get(metric, [1])
                    if normalize_errors:
                        error /= normalizer[param_idx]
                        metric_errors.append(error)
                    else:
                        metric_errors.append(error)
                    count += 1
                except ValueError:
                    logger.warning(f"Non-numeric values for Run ID '{run_id}', Metric '{metric}', Parameter index '{param_idx}'. Skipping this parameter.")
                    continue
            run_total_errors.append(np.mean(metric_errors))
        errors.append(np.mean(run_total_errors))

    if count == 0:
        logger.error("No valid parameter comparisons found. RMSE and SE are undefined.")
        return np.nan, np.nan

    errors = np.array(errors)
    squared_errors = errors[~np.isnan(errors)] ** 2  # Exclude NaNs

    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)

    # Bootstrapping to estimate SE of RMSE
    errors = errors[~np.isnan(errors)]  # Remove NaNs from errors
    n = len(errors)
    if n == 0:
        logger.error("No valid errors after removing NaNs. SE is undefined.")
        return rmse, np.nan

    bootstrap_rmses = []
    for _ in range(num_bootstrap):
        sample_indices = np.random.choice(n, n, replace=True)
        sample_errors = errors[sample_indices]
        sample_rmse = np.sqrt(np.mean(sample_errors ** 2))
        bootstrap_rmses.append(sample_rmse)

    se_rmse = np.std(bootstrap_rmses, ddof=1)

    logger.info(f"Computed RMSE: {rmse:.4f} with SE: {se_rmse:.4f} based on {count} parameter comparisons.")
    return rmse, se_rmse

def compute_rmse_per_metric(estimated_params, ground_truth_params, metrics, parameters, logger, main_param_only = False, normalize_errors=False,rmse_metric_normalizer={}):
    """
    Compute RMSE for each metric individually between estimated parameters and ground truth parameters.

    Args:
        estimated_params (dict): Estimated parameters (IMLE or DLVM).
        ground_truth_params (dict): Ground truth parameters.
        metrics (list): List of metrics to compare.
        parameters (dict): Dictionary mapping metrics to parameter indices.
        logger (logging.Logger): Logger for logging messages.

    Returns:
        dict: Dictionary with RMSE values for each metric.
    """
    rmse_per_metric = {}
    se_per_metric = {}

    for metric in metrics:
        errors = []
        for run_id, est_metrics in estimated_params.items():
            # Handle different run ID structures
            original_run_id = '_'.join(run_id.split('_')[:-1])
            gt_metrics = ground_truth_params.get(original_run_id, None)
            if gt_metrics is None:
                logger.warning(f"Original Run ID '{original_run_id}' not found in ground truth data. Skipping synthetic run '{run_id}'.")
                errors.append(np.nan)
                continue
            metric_errors = []
            for param_idx in parameters.get(metric, []):
                if main_param_only and param_idx != 0:
                    continue
                #param_idx = 0  # Only consider the first parameter for each metric
                est_param_list = est_metrics.get(metric, [])
                gt_param_list = gt_metrics.get(metric, [])

                if len(est_param_list) <= param_idx or len(gt_param_list) <= param_idx:
                    logger.warning(f"Missing parameter index {param_idx} for Run ID '{run_id}', Metric '{metric}'. Skipping this parameter.")
                    errors.append(np.nan)
                    continue

                est_value = est_param_list[param_idx]
                gt_value = gt_param_list[param_idx]

                if est_value is None or gt_value is None:
                    logger.warning(f"Missing values for Run ID '{run_id}', Metric '{metric}', Parameter index '{param_idx}'. Skipping this parameter.")
                    errors.append(np.nan)
                    continue

                # Ensure values are numbers
                try:
                    metric_type = SUMMARIZED_METRICS_METRIC_TYPES[metric]
                    est_value = float(est_value)
                    gt_value = float(gt_value)
                    error = est_value - gt_value
                    # Normalize the error based on the metric type
                    normalizer = rmse_metric_normalizer.get(metric, [1])

                    
                    if normalize_errors:
                        error /= normalizer[param_idx]
                        metric_errors.append(error)
                    else:
                        metric_errors.append(error)
                    
                    logger.debug(f"Run ID: {run_id} | Metric {metric} | Param Index {param_idx} | Comparing values: Estimated={est_value:.4f}, Ground Truth={gt_value:.4f}")
                except ValueError:
                    logger.warning(f"Non-numeric values for Run ID '{run_id}', Metric '{metric}', Parameter index '{param_idx}'. Skipping this parameter.")
                    continue
            errors.append(np.mean(metric_errors))  # Append the mean error for this metric

                # Calculate error and add to list

        # Calculate RMSE for this metric if there are valid errors
        if errors: # Check if errors list is not empty
            # mse = (np.array(errors) ** 2)
            # rmse = np.sqrt(mse)
            # rmse = np.mean(rmse)
            # rmse_per_metric[metric] = rmse
            # logger.info(f"Computed RMSE for metric '{metric}': {rmse:.4f}")
            errors = np.array(errors)
            squared_errors = errors[~np.isnan(errors)] ** 2  # Exclude NaNs

            mse = np.mean(squared_errors)
            rmse = np.sqrt(mse)
            rmse_per_metric[metric] = rmse
            se = np.std(errors[~np.isnan(errors)], ddof=1) / np.sqrt(len(errors[~np.isnan(errors)]))  # Standard error of RMSE
            se_per_metric[metric] = se
            logger.info(f"Computed RMSE for metric '{metric}': {rmse:.4f}")
        else:
            # import pdb; pdb.set_trace()
            rmse_per_metric[metric] = np.nan
            se_per_metric[metric] = np.nan
            se_per_metric[metric] = np.nan
            logger.warning(f"No valid parameter comparisons found for metric '{metric}'. RMSE is undefined.")

    # import pdb; pdb.set_trace()
    return rmse_per_metric, se_per_metric

def compute_kld_for_metric(est_param_list, gt_param_list, metric, metric_type, return_log_kld=True):
    """
    Compute the KLD for a given metric and its type.

    Args:
        est_param_list (list): Estimated parameters for the metric.
        gt_param_list (list): Ground truth parameters for the metric.
        metric (str): The name of the metric.
        metric_type (str): The type of the metric.

    Returns:
        float: The computed KLD value.
    """
    if metric_type == 'binarySpan':
        # Compute KLD for each length from 2 to 10
        cum_kld = 0
        for length in range(2, 11):
            length_metric = f"{metric}_correct_w_len_{length}"
            base_dist = mle_params_to_dist(length_metric, gt_param_list, metric_type, metrics_dict = CURR_METRICS_DICT)
            predicted_dist = mle_params_to_dist(length_metric, est_param_list, metric_type, metrics_dict = CURR_METRICS_DICT)
            # Move distributions to the appropriate device
            base_dist = move_distribution_to_cuda(base_dist)
            predicted_dist = move_distribution_to_cuda(predicted_dist)

            # Calculate KLD
            kld = calculate_kld_given_metric(predicted_dist,base_dist,  length_metric)
            cum_kld += kld
        return torch.mean(cum_kld)
    else:
        
        base_dist = mle_params_to_dist(metric, gt_param_list, metric_type, metrics_dict = CURR_METRICS_DICT)
        predicted_dist = mle_params_to_dist(metric, est_param_list, metric_type, metrics_dict = CURR_METRICS_DICT)
        # Move distributions to the appropriate device
        base_dist = move_distribution_to_cuda(base_dist)
        predicted_dist = move_distribution_to_cuda(predicted_dist)

    kld = calculate_kld_given_metric(predicted_dist, base_dist, metric)
    # Calculate KLD
    if return_log_kld:
        return torch.log(kld)
    else:
        return kld

def compute_kld(estimated_params, ground_truth_params, metrics, logger, num_bootstrap=1000, normalize_errors=False, rmse_metric_normalizer={}):
    """
    Compute KL Divergence between estimated parameters and ground truth parameters.

    Args:
        estimated_params (dict): Estimated parameters (IMLE or DLVM).
        ground_truth_params (dict): Ground truth parameters.
        metrics (list): List of metrics to compare.
        parameters (dict): Dictionary mapping metrics to parameter indices.
        logger (logging.Logger): Logger for logging messages.
        num_bootstrap (int): Number of bootstrap samples for estimating SE.
        main_param_only (bool): Whether to only consider the main parameter for each metric.
        normalize_errors (bool): Whether to normalize errors based on metric type.
        rmse_metric_normalizer (dict): Dictionary of normalizers for each metric.

    Returns:
        tuple: (mean_kld, se_kld)
    """
    klds = []
    count = 0
    for run_id, est_metrics in estimated_params.items():
        original_run_id = '_'.join(run_id.split('_')[:-1])
        gt_metrics = ground_truth_params.get(original_run_id, None)
        if gt_metrics is None:
            logger.warning(f"Original Run ID '{original_run_id}' not found in ground truth data. Skipping synthetic run '{run_id}'.")
            continue

        run_total_klds = []
        for metric in metrics:

            est_param_list = est_metrics.get(metric, [])
            gt_param_list = gt_metrics.get(metric, [])

            if not est_param_list or not gt_param_list:
                logger.warning(f"Missing parameters for Run ID '{run_id}', Metric '{metric}'. Skipping.")
                klds.append(np.nan)
                count += 1
                continue

            metric_type = SUMMARIZED_METRICS_METRIC_TYPES.get(metric, '')
            kld = compute_kld_for_metric(est_param_list, gt_param_list, metric, metric_type, return_log_kld=False)

            if normalize_errors:
                kld /= rmse_metric_normalizer.get(metric, 1)


            

            count += 1


            run_total_klds.append(kld.item())

        if run_total_klds:
            klds.append(np.mean(run_total_klds))

    klds = np.array(klds)
    valid_klds = klds[~np.isnan(klds)]

    if len(valid_klds) == 0:
        logger.error("No valid KL Divergence values after removing NaNs. Mean and SE are undefined.")
        return np.nan, np.nan

    mean_kld = float(np.mean(valid_klds))

    bootstrap_klds = []
    for _ in range(num_bootstrap):
        sample_indices = np.random.choice(len(valid_klds), len(valid_klds), replace=True)
        sample_klds = valid_klds[sample_indices]
        bootstrap_klds.append(np.mean(sample_klds))

    se_kld = float(np.std(bootstrap_klds, ddof=1))

    logger.info(f"Computed mean KL Divergence: {mean_kld:.4f} with SE: {se_kld:.4f} based on {count} parameter comparisons.")
    return mean_kld, se_kld

def compute_kld_per_metric(estimated_params, ground_truth_params, metrics, logger, normalize_errors=False, rmse_metric_normalizer={}):
    """
    Compute KL Divergence for each metric individually between estimated parameters and ground truth parameters.

    Args:
        estimated_params (dict): Estimated parameters (IMLE or DLVM).
        ground_truth_params (dict): Ground truth parameters.
        metrics (list): List of metrics to compare.
        parameters (dict): Dictionary mapping metrics to parameter indices.
        logger (logging.Logger): Logger for logging messages.
        main_param_only (bool): Whether to only consider the main parameter for each metric.
        normalize_errors (bool): Whether to normalize errors based on metric type.
        rmse_metric_normalizer (dict): Dictionary of normalizers for each metric.

    Returns:
        dict: Dictionary with KL Divergence values for each metric.
    """
    kld_per_metric = {}
    se_per_metric = {}

    for metric in metrics:
        klds = []
        for run_id, est_metrics in estimated_params.items():
            original_run_id = '_'.join(run_id.split('_')[:-1])
            gt_metrics = ground_truth_params.get(original_run_id, None)
            if gt_metrics is None:
                logger.warning(f"Original Run ID '{original_run_id}' not found in ground truth data. Skipping synthetic run '{run_id}'.")
                klds.append(np.nan)
                continue

            est_param_list = est_metrics.get(metric, [])
            gt_param_list = gt_metrics.get(metric, [])

            if not est_param_list or not gt_param_list:
                logger.warning(f"Missing parameters for Run ID '{run_id}', Metric '{metric}'. Skipping this metric.")
                klds.append(np.nan)
                continue

            if None in est_param_list or None in gt_param_list:
                logger.warning(f"Missing values for Run ID '{run_id}', Metric '{metric}'. Skipping this metric.")
                klds.append(np.nan) 
                continue

            metric_type = SUMMARIZED_METRICS_METRIC_TYPES.get(metric, '')
            kld = compute_kld_for_metric(est_param_list, gt_param_list, metric, metric_type, return_log_kld=False)
            if normalize_errors:
                kld /= rmse_metric_normalizer.get(metric, 1)

            klds.append(kld.item())

        if klds:
            klds = np.array(klds)
            valid_klds = klds[~np.isnan(klds)]

            if len(valid_klds) > 0:
                mean_kld = np.mean(valid_klds)
                kld_per_metric[metric] = mean_kld
                se = np.std(valid_klds, ddof=1) / np.sqrt(len(valid_klds))
                se_per_metric[metric] = se
                logger.info(f"Computed KL Divergence for metric '{metric}': {mean_kld:.4f}")
            else:
                kld_per_metric[metric] = np.nan
                se_per_metric[metric] = np.nan
                logger.warning(f"No valid KL Divergence values for metric '{metric}'. Mean and SE are undefined.")
        else:
            kld_per_metric[metric] = np.nan
            se_per_metric[metric] = np.nan
            logger.warning(f"No valid parameter comparisons found for metric '{metric}'. KL Divergence is undefined.")

    return kld_per_metric, se_per_metric

def plot_rmse_combined(df_plot, results_dict, rmse_plot_path, all_n_values, ylabel = 'Root Mean Squared Error (RMSE)'):
    # Plot each DLVM model
    dlvm_models = df_plot[df_plot['Method'] == 'DLVM']['ModelID'].unique()
    # Plot RMSE vs N for IMLE and all DLVM models on the same graph, in both linear and log scales
    palette = sns.color_palette("tab10", n_colors=len(dlvm_models)+1)
    palette = palette[:len(dlvm_models)]  # Ensure enough colors for all models
    dlvm_models = sorted(dlvm_models)  # Sort for consistent color mapping
    # create a dict of colors for each model
    model_colors = {model_id: palette[idx] for idx, model_id in enumerate(dlvm_models)}
    # Add IMLE color
    model_colors['IMLE'] = 'black'  # IMLE color

    for scale in ['linear', 'log']:
        plt.figure(figsize=(14, 10))
        sns.set_theme(style="whitegrid")

        # Plot IMLE if available
        if 'IMLE' in results_dict and not df_plot[df_plot['Method'] == 'IMLE'].empty:
            df_imle = df_plot[df_plot['Method'] == 'IMLE']
            plt.errorbar(
                df_imle['N'] if scale == "linear" else np.log(df_imle['N']),
                df_imle['Value'],
                yerr=df_imle['SE'],
                fmt='-o',
                color=model_colors['IMLE'],
                ecolor='lightgray', 
                elinewidth=2,
                capsize=6,
                capthick=2,
                markersize=8,
                markerfacecolor='black',
                markeredgecolor='black',
                markeredgewidth=1.5,
                label='IMLE RMSE ± SE'
            )

        
        

        for idx, model_id in enumerate(dlvm_models):
            df_model = df_plot[(df_plot['Method'] == 'DLVM') & (df_plot['ModelID'] == model_id)]
            plt.errorbar(
                df_model['N'] if scale == "linear" else np.log(df_model['N']), # Always plot actual N values
                df_model['Value'],
                yerr=df_model['SE'],
                fmt='-s',
                color=model_colors[model_id],
                ecolor='gray',
                elinewidth=2,
                capsize=6,
                capthick=2,
                markersize=8,
                markerfacecolor=model_colors[model_id],
                markeredgecolor='black',
                markeredgewidth=1.5,
                label=f'DLVM {model_id} RMSE ± SE'
            )

        plt.title(f'IMLE and DLVM Models ({DATASET}) Predictive Performance on Simulated data', fontsize=20, fontweight='bold')
        plt.xlabel(f'Number of Observations per Metric', fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        
        # Set scales after plotting data
        # if scale == 'log':
        #     plt.xscale('log')  # This will handle the x-axis log transformation
        #     plt.yscale('log')  # This will handle the y-axis log transformation
            
        # Set ticks - use actual N values
        if scale == 'log':
            plt.xticks(np.log(all_n_values), all_n_values, fontsize=12)
        else:
            plt.xticks(all_n_values, all_n_values, fontsize=12)
        plt.yticks(fontsize=12)
        
        plt.ylim(bottom=0)
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.legend(fontsize=14, loc='upper right', ncol=2)
        plt.tight_layout()
        
        # Save with scale indicator in filename
        scale_path = rmse_plot_path.replace('.pdf', f'_{scale}_scale.pdf')
        plt.savefig(scale_path, dpi=300)
        plt.close()
        print(f"Combined RMSE vs N plot ({scale} scale) saved to '{scale_path}'.")

def plot_rmse_per_metric(metrics, results_per_metric, n_values, output_dir, file_name = "",ylabel = 'Root Mean Squared Error (RMSE)'):
    """
    Plot RMSE vs N for each metric separately, comparing IMLE and DLVM models.

    Args:
        metrics (list): List of metric names.
        results_per_metric (dict): Nested dictionary with structure {
            'metric1': {
                'IMLE': {N1: rmse, N2: rmse, ...},
                'D1': {N1: rmse, N2: rmse, ...},
                ...
            },
            'metric2': { ... },
            ...
        }
        n_values (list): List of N values.
        output_dir (str): Directory to save the plots.
    """
    # Find any metric and use it to get the models
    if results_per_metric:
        first_metric = next(iter(results_per_metric))
        models = sorted(results_per_metric[first_metric].keys())
    else:
        models = []
    palette = sns.color_palette("tab10", n_colors=len(models)+1)
    palette = palette[:len(models)]  # Ensure enough colors for all models
    dlvm_models = [model for model in models if model != 'IMLE']  # Exclude IMLE from DLVM models
    # Sort models for consistent color mapping
    dlvm_models.sort()
    model_colors = {model_id: palette[idx] for idx, model_id in enumerate(dlvm_models)}
    # Add IMLE color
    model_colors['IMLE'] = 'black'  # IMLE color

    for metric in metrics:
        for scale in ['linear', 'log']:
            plt.figure(figsize=(14, 10))
            sns.set_theme(style="whitegrid")
            
            palette = sns.color_palette("tab10")
            
            has_positive_values = False  # Flag to check for positive values
            
            for idx, model_id in enumerate(models):
                rmse_values = [results_per_metric[metric][model_id].get(N, np.nan) for N in n_values]
                
                plt.plot(
                    n_values if scale == 'linear' else np.log(n_values),
                    rmse_values,
                    marker='o',
                    linestyle='-',
                    color=model_colors[model_id],
                    label=f'{"" if model_id == "IMLE" else "DLVM "}{model_id}',
                )
            
            plt.title(f'{SUMMARIZED_METRICS_MAIN_MOMENTS_LABEL_DICT[metric]}', fontsize=20, fontweight='bold')
            plt.xlabel(f'Number of Observations per Metric', fontsize=16)
            plt.ylabel(ylabel, fontsize=16)
            if scale == 'log':
                plt.xticks(np.log(n_values), n_values, fontsize=12)
            else:
                plt.xticks(n_values, range(0,len(n_values)) if scale == 'log' else n_values, fontsize=12)
            plt.yticks(fontsize=12)
            plt.ylim(bottom=0)
            plt.grid(True, linestyle='--', linewidth=0.5)
            plt.legend(fontsize=14, loc='upper right')
            
            plt.tight_layout()
            
            # Save the plot
            sanitized_metric = re.sub(r'[^A-Za-z0-9]+', '_', metric)
            scale_suffix = 'log' if scale == 'log' else 'linear'
            plot_path = os.path.join(output_dir, f"{metric}_{file_name}.pdf")
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"RMSE vs N plot for metric '{metric}' ({scale} scale) saved to '{plot_path}'.")

def load_ground_truth_params(ground_truth_path, logger):
    """
    Load ground truth parameters from a .pt file.

    Args:
        ground_truth_path (str): Path to the ground truth .pt file.
        logger (logging.Logger): Logger for logging messages.

    Returns:
        dict: Ground truth parameters.
    """
    if not os.path.exists(ground_truth_path):
        logger.error(f"Ground truth parameters file not found at {ground_truth_path}")
        sys.exit(1)

    try:
        ground_truth_params = torch.load(ground_truth_path, map_location='cpu')
        logger.info(f"Ground truth parameters loaded successfully from '{ground_truth_path}' with {len(ground_truth_params)} runs.")
        return ground_truth_params
    except Exception as e:
        logger.error(f"Failed to load ground truth parameters from '{ground_truth_path}': {e}")
        sys.exit(1)

def load_estimated_params(params_path, logger, param_type="IMLE"):
    """
    Load estimated parameters (IMLE or DLVM) from a .pt file.

    Args:
        params_path (str): Path to the estimated parameters .pt file.
        logger (logging.Logger): Logger for logging messages.
        param_type (str): Type of parameters ("IMLE" or "DLVM").

    Returns:
        dict: Estimated parameters.
    """
    if not os.path.exists(params_path):
        logger.warning(f"{param_type} parameters file not found at {params_path}")
        return None  # Instead of exiting, return None to allow flexibility

    try:
        estimated_params = torch.load(params_path, map_location='cpu')
        logger.info(f"Loaded {param_type} parameters from '{params_path}' with {len(estimated_params)} runs.")
        return estimated_params
    except Exception as e:
        logger.error(f"Failed to load {param_type} parameters from '{params_path}': {e}")
        return None

# ============================================
# Main Function
# ============================================

def main():
    """
    Main function to compute RMSE or KL Divergence for different N and plot the results on a single graph.
    Additionally, plots the chosen metric per metric separately for each metric across all N.
    """
    parser = argparse.ArgumentParser(description="Plot RMSE or KL Divergence curves for all models")
    parser.add_argument("--ground_truth_param_file", type=str, default=None, help="Path to the ground truth parameters file")
    parser.add_argument("--main_param_only", action='store_true', help="Only compute metrics for the main parameter of each metric")
    parser.add_argument("--normalize_errors", action='store_true', help="Normalize errors based on metric type")
    parser.add_argument("--metric", type=str, choices=["rmse", "kld"], default="rmse", 
                        help="Metric to compute and plot: Root Mean Squared Error (rmse) or KL Divergence (kld)")
    args = parser.parse_args()
    
    # Define dataset name
    dataset_name = DATASET  # Replace with your actual dataset name

    # Define directories using relative paths
    # Ground truth is in '../../data/COL10/all_data-best_mle_params_mpf100.pt'
    if args.ground_truth_param_file is not None:
        ground_truth_params_path = args.ground_truth_param_file
        ground_truth_params_path = os.path.abspath(
            os.path.join(script_dir, '..', '..', 'data', dataset_name, args.ground_truth_param_file)
        )
    else:
        ground_truth_params_path = os.path.abspath(
            os.path.join(script_dir, '..', '..', 'data', dataset_name, 'all_data-best_mle_params_mpf100.pt')
        )
    
    params_dir = os.path.join(script_dir, 'synthetic_data', dataset_name,"param_fits")

    plots_dir = os.path.join(script_dir,'synthetic_data', dataset_name, 'plots')
    # Ensure plots_dir exists
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"Created directory: {plots_dir}")

    # Create directories if they don't exist (they should exist since it's the script_dir)
    for directory in [params_dir, plots_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

    if args.metric == "kld":
        normalizer_params = {
            'CorsiComplex': 6.5,
            'Stroop_reaction_time': 2.5,
            'RunningSpan_correct_w_len_2': 0.5,
            'Countermanding_reaction_time': 2.5,
            'SimpleSpan': 6.5,
            'RunningSpan_correct_w_len_3': 0.5,
            'D2_hit_accuracy': 0.5,
            'PasatPlus_correctly_answered': 0.5,
        }
    else:
        normalizer_params = {'CorsiComplex': [12.234659463167189, 1.9972472935914993],
                        'Stroop_reaction_time': [1.0243954658508292, 0.5401461943984033],
                        'RunningSpan_correct_w_len_2': [0.9800000097602607],
                        'Countermanding_reaction_time': [0.7413206100463867, 0.38473375886678696],
                        'SimpleSpan': [12.079994469881058, 1.9972472935914993],
                        'RunningSpan_correct_w_len_3': [0.9800000097602607],
                        'D2_hit_accuracy': [0.6566666662693024],
                        'PasatPlus_correctly_answered': [0.8899999763816596]}
    # Define logging
    log_file = os.path.join(plots_dir, f'plot_rmse_python3 plot_rmse_curves_all_models.py --ground_truth_param_file mle_params_4_evaluation_purposes.pt{dataset_name}.log')
    logger = setup_logging(log_file)
    logger.info("===== Starting RMSE Plotting Script =====")

    # Load ground truth parameters
    ground_truth_params = load_ground_truth_params(ground_truth_params_path, logger)
    # remove ID from ground truth params if they are in OUTLIER_HELDOUT_SESSIONS
    ground_truth_params = {k: v for k, v in ground_truth_params.items() if k not in OUTLIER_HELDOUT_SESSIONS}
    logger.info(f"Removed {len(OUTLIER_HELDOUT_SESSIONS)} outlier heldout sessions from ground truth parameters, {len(ground_truth_params)} runs remaining")

    # Extract metrics and parameter indices from ground truth
    first_run_id = next(iter(ground_truth_params))
    first_run = ground_truth_params[first_run_id]
    # metrics = list(first_run.keys())
    metrics = SUMMARIZED_METRICS
    parameters = {}
    for metric in metrics:
        metric_type = SUMMARIZED_METRICS_METRIC_TYPES[metric]
        # Set number of parameters based on metric type
        if metric_type == 'binarySpan' or metric_type == 'timing':
            parameters[metric] = [0,1]  # mean and std or psiAlpha and psiSigma
        else:
            parameters[metric] = [0]  # default to single parameter for binary

    logger.debug(f"Metrics: {metrics}")
    logger.debug(f"Parameters: {parameters}")

    # Identify all DLVM parameter files and extract model_ids
    synthetic_dlvm_files = [f for f in os.listdir(params_dir)
                            if f.startswith("synthetic_dlvm_params_D") and f.endswith(".pt")]

    if not synthetic_dlvm_files:
        logger.error(f"No DLVM parameter files found in '{params_dir}'. Ensure DLVM fitting has been performed.")
        sys.exit(1)

    # Extract unique model_ids from filenames
    model_ids = set()
    pattern = re.compile(r'synthetic_dlvm_params_(D\d+)_N(\d+)\.pt')

    for dlvm_file in synthetic_dlvm_files:
        match = pattern.match(dlvm_file)
        if match:
            model_id = match.group(1)
            model_ids.add(model_id)
        else:
            logger.warning(f"DLVM filename '{dlvm_file}' does not match expected pattern. Skipping.")


    if not model_ids:
        logger.error("No valid model_ids extracted from DLVM parameter filenames.")
        sys.exit(1)

    model_ids = sorted(model_ids)  # Sort for consistency
    logger.info(f"Identified DLVM models: {model_ids}")

    # Initialize a dictionary to hold results for each model and N
    # Also include 'IMLE' as a separate key
    results_dict = {'IMLE': {}}  # Initialize IMLE separately

    # Add this initialization for each model ID
    for model_id in model_ids:
        results_dict[model_id] = {}

    # Initialize a nested dictionary to hold results per metric
    # Structure: {metric: {model_id: {N: value, ...}, ...}, ...}
    results_per_metric = {metric: {model_id: {} for model_id in ['IMLE'] + model_ids} for metric in metrics}
    

    # Extract all N values from IMLE parameter files
    synthetic_mle_files = [f for f in os.listdir(params_dir)
                           if f.startswith("synthetic_mle_params_N") and f.endswith(".pt")]

    if not synthetic_mle_files:
        logger.warning(f"No IMLE parameter files found in '{params_dir}'. Proceeding with DLVM models only.")
        N_values_imle = []  # Initialize as empty list to continue execution
    else:
        # Extract N values from filenames
        N_values_imle = []
        for mle_file in synthetic_mle_files:
            try:
                N_str = mle_file.replace("synthetic_mle_params_N", "").replace(".pt", "")
                N = int(N_str)
                N_values_imle.append(N)
            except ValueError:
                logger.warning(f"Filename '{mle_file}' does not conform to 'synthetic_mle_params_N{{N}}.pt' format. Skipping.")

    if not N_values_imle:
        logger.warning("No valid N values extracted from IMLE parameter filenames. Proceeding with DLVM models only.")

    N_values_imle = sorted(list(set(N_values_imle)))  # Remove duplicates and sort
    logger.info(f"IMLE: Processing {args.metric.upper()} for N values: {N_values_imle}")

    # Loop over each N to compute the chosen metric for IMLE
    for N in N_values_imle:
        logger.info(f"Computing {args.metric.upper()} for IMLE and N={N}")

        # Define path to IMLE parameter file
        imle_params_path = os.path.join(params_dir, f"synthetic_mle_params_N{N}.pt")

        # Load IMLE parameters
        imle_params = load_estimated_params(imle_params_path, logger, param_type="IMLE")

        if imle_params is None:
            logger.warning(f"IMLE parameters for N={N} could not be loaded. Skipping computation for this N.")
            results_dict['IMLE'][N] = {'Value': np.nan, 'SE': np.nan}
            # Also skip per metric computation
            for metric in metrics:
                results_per_metric[metric]['IMLE'][N] = np.nan
            continue

        # Compute overall metric and SE between IMLE parameters and ground truth
        if args.metric == "rmse":
            value, se = compute_rmse(imle_params, ground_truth_params, metrics, parameters, 
                                    logger, main_param_only=args.main_param_only, normalize_errors=args.normalize_errors,
                                    rmse_metric_normalizer=normalizer_params)
            # Compute per metric for IMLE
            per_metric_values, _ = compute_rmse_per_metric(imle_params, ground_truth_params, metrics, parameters, logger, 
                                                           main_param_only=args.main_param_only, normalize_errors=args.normalize_errors,
                                                           rmse_metric_normalizer=normalizer_params)
        else:  # kld
            value, se = compute_kld(imle_params, ground_truth_params, metrics, logger, 
                                   normalize_errors=args.normalize_errors,
                                   rmse_metric_normalizer=normalizer_params)
            # Compute per metric for IMLE
            per_metric_values, _ = compute_kld_per_metric(imle_params, ground_truth_params, metrics, logger, 
                                                          normalize_errors=args.normalize_errors,
                                                          rmse_metric_normalizer=normalizer_params)
        
        results_dict['IMLE'][N] = {'Value': value, 'SE': se}
        logger.info(f"IMLE {args.metric.upper()} for N={N}: {value:.4f} with SE: {se:.4f}")

        # Store per metric values
        for metric, metric_value in per_metric_values.items():
            results_per_metric[metric]['IMLE'][N] = metric_value

    # Now, loop over each DLVM model and compute the chosen metric
    for D in model_ids:
        # import pdb; pdb.set_trace()
        logger.info(f"Processing Model: {D}")

        # Find all N values for this model
        dlvm_files = [f for f in synthetic_dlvm_files if f.startswith(f"synthetic_dlvm_params_{D}_N")]

        if not dlvm_files:
            logger.warning(f"No DLVM parameter files found for Model '{D}'. Skipping.")
            continue

        # Extract N values from filenames
        N_values = []
        for dlvm_file in dlvm_files:
            match = pattern.match(dlvm_file)
            if match:
                N = int(match.group(2))
                N_values.append(N)
            else:
                logger.warning(f"DLVM filename '{dlvm_file}' does not match expected pattern. Skipping.")

        if not N_values:
            logger.warning(f"No valid N values extracted for Model '{D}'. Skipping.")
            continue

        N_values = sorted(list(set(N_values)))  # Remove duplicates and sort
        logger.info(f"Model '{D}': Processing {args.metric.upper()} for N values: {N_values}")

        # Loop over each N
        for N in N_values:
            logger.info(f"Computing {args.metric.upper()} for Model '{D}' and N={N}")

            # Define paths to IMLE and DLVM parameter files
            imle_params_path = os.path.join(params_dir, f"synthetic_mle_params_N{N}.pt")
            dlvm_params_path = os.path.join(params_dir, f"synthetic_dlvm_params_{D}_N{N}.pt")

            # Load estimated parameters
            imle_params = load_estimated_params(imle_params_path, logger, param_type="IMLE")
            dlvm_params = load_estimated_params(dlvm_params_path, logger, param_type="DLVM")

            if dlvm_params is None:
                logger.warning(f"DLVM parameters for Model '{D}', N={N} could not be loaded. Skipping computation for this N.")
                results_dict[D][N] = {'Value': np.nan, 'SE': np.nan}
                # Also skip per metric computation
                for metric in metrics:
                    results_per_metric[metric][D][N] = np.nan
                continue

            # Compute overall metric and SE between DLVM parameters and ground truth
            if args.metric == "rmse":
                value, se = compute_rmse(dlvm_params, ground_truth_params, metrics, parameters, logger, 
                                        main_param_only=args.main_param_only, normalize_errors=args.normalize_errors,
                                        rmse_metric_normalizer=normalizer_params)
                # Compute per metric for DLVM
                per_metric_values, se_per_metric = compute_rmse_per_metric(dlvm_params, ground_truth_params, metrics, parameters, logger, 
                                                               main_param_only=args.main_param_only, normalize_errors=args.normalize_errors,
                                                               rmse_metric_normalizer=normalizer_params)
            else:  # kld
                value, se = compute_kld(dlvm_params, ground_truth_params, metrics, logger, 
                                     normalize_errors=args.normalize_errors,
                                       rmse_metric_normalizer=normalizer_params)
                # Compute per metric for DLVM
                per_metric_values, se_per_metric = compute_kld_per_metric(dlvm_params, ground_truth_params, metrics, logger, 
                                                               normalize_errors=args.normalize_errors,
                                                              rmse_metric_normalizer=normalizer_params)
            
            # Debugging: Check the types of value and se
            logger.debug(f"Type of value: {type(value)}, Type of se: {type(se)}")

            # Store results in results_dict
            
            results_dict[D][N] = {'Value': value, 'SE': se}
            logger.info(f"DLVM {args.metric.upper()} for Model '{D}', N={N}: {value:.4f} with SE: {se:.4f}")
            

            # Store per metric values
            for metric, metric_value in per_metric_values.items():
                results_per_metric[metric][D][N] = metric_value

    # Now, prepare data for combined plotting
    # Collect all N values across IMLE and DLVM models
    all_n_values = sorted(set(N_values_imle))
    # Initialize a DataFrame
    plot_data = []

    # Add IMLE data if available
    if 'IMLE' in results_dict and results_dict['IMLE']:
        for N in all_n_values:
            value = results_dict['IMLE'].get(N, {}).get('Value', np.nan)
            se = results_dict['IMLE'].get(N, {}).get('SE', np.nan)
            plot_data.append({
                'Method': 'IMLE',
                'ModelID': 'IMLE',
                'N': N,
                'Value': value,
                'SE': se
            })

    # Add DLVM data
    for D in model_ids:
        if D in results_dict and results_dict[D]:
            for N in all_n_values:
                value = results_dict.get(D, {}).get(N, {}).get('Value', np.nan)
                se = results_dict.get(D, {}).get(N, {}).get('SE', np.nan)
                if not np.isnan(value) and not np.isnan(se):
                    plot_data.append({
                        'Method': 'DLVM',
                        'ModelID': D,
                        'N': N,
                        'Value': value,
                        'SE': se
                    })

    # Create DataFrame from plot_data
    df_plot = pd.DataFrame(plot_data)

    # Check if df_plot is empty
    if df_plot.empty:
        logger.warning("No data available for plotting. Ensure that the parameter files are correctly generated.")
    else:
        # Proceed with plotting
        # Define output path for the combined plot
        metric_name = "RMSE" if args.metric == "rmse" else "KLD"
        plot_path = os.path.join(plots_dir, f"{metric_name}_combined_{dataset_name}_{'main_params_only' if args.main_param_only else 'all_params'}_{'normalized' if args.normalize_errors else 'unnormalized'}.pdf")

        # Update plot_rmse_combined to handle both RMSE and KLD
        ylabel = f"{'Normalized ' if args.normalize_errors else ''}{'Root Mean Squared Error (RMSE)' if args.metric == 'rmse' else 'KL Divergence (KLD)'}"
        
        # Rename RMSE column to Value for plotting
        if 'RMSE' in df_plot.columns:
            df_plot = df_plot.rename(columns={'RMSE': 'Value'})
        
        plot_rmse_combined(df_plot, results_dict, plot_path, all_n_values, ylabel=ylabel)

    # Save results to CSV
    csv_path = os.path.join(plots_dir, f"{args.metric}_results_combined_{dataset_name}.csv")
    df_plot.to_csv(csv_path, index=False)
    logger.info(f"{args.metric.upper()} results saved to '{csv_path}'.")

    # Generate plots for each metric
    logger.info(f"Generating {args.metric.upper()} vs N plots for each individual metric.")
    file_name = f"{args.metric}_per_metric_{dataset_name}_{'main_params_only' if args.main_param_only else 'all_params'}_{'normalized' if args.normalize_errors else 'unnormalized'}"
    
    # Update plot_rmse_per_metric to handle both RMSE and KLD
    ylabel = f"{'Normalized ' if args.normalize_errors else ''}{'Root Mean Squared Error (RMSE)' if args.metric == 'rmse' else 'KL Divergence (KLD)'}"
    plot_rmse_per_metric(metrics, results_per_metric, all_n_values, plots_dir, file_name=file_name, ylabel=ylabel)
    
    logger.info(f"===== {args.metric.upper()} Plotting Script Completed Successfully =====")

# ============================================
# Execute Main Function
# ============================================

if __name__ == "__main__":
    main()
