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
from pathlib import Path

# Configure Seaborn for better aesthetics (optional)
sns.set_theme(style="whitegrid")

SCRIPT_DIR = os.fspath(Path(__file__).resolve().parent)
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from nmil_dlvm.paths import data_dir
from nmil_dlvm.utils.data_distribution_utils import (OUTLIER_HELDOUT_SESSIONS, DATASET,
                                                     SUMMARIZED_METRICS_MAIN_MOMENTS_LABEL_DICT,
                                                     SUMMARIZED_METRICS, SUMMARIZED_METRICS_METRIC_TYPES, mle_params_to_dist,
                                                     CURR_METRICS_DICT)  # Ensure this is defined in your utils
from nmil_dlvm.utils.active_learning_utils import calculate_kld_given_metric, move_distribution_to_cuda


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

def compute_rmse(estimated_params, ground_truth_params, metrics, parameters, logger, num_bootstrap=1000,
                 main_param_only=False, normalize_errors=False, rmse_metric_normalizer={}, use_std_dev=False):
    """
    Compute RMSE and standard error/deviation between estimated parameters and ground truth parameters.

    Args:
        estimated_params (dict): Estimated parameters (IMLE or DLVM).
        ground_truth_params (dict): Ground truth parameters.
        metrics (list): List of metrics to compare.
        parameters (dict): Dictionary mapping metrics to parameter indices.
        logger (logging.Logger): Logger for logging messages.
        num_bootstrap (int): Number of bootstrap samples for estimating SE.
        use_std_dev (bool): If True, compute standard deviation; otherwise, compute standard error.

    Returns:
        tuple: (rmse, error_metric) where error_metric is either SE or SD.
    """
    errors = []
    count = 0
    for run_id, est_metrics in estimated_params.items():
        # Handle different run ID structures
        original_run_id = '_'.join(run_id.split('_')[:-1])
        gt_metrics = ground_truth_params.get(original_run_id, None)
        if gt_metrics is None:
            logger.warning(
                f"Original Run ID '{original_run_id}' not found in ground truth data. Skipping synthetic run '{run_id}'.")
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
                    logger.warning(
                        f"Missing parameter index {param_idx} for Run ID '{run_id}', Metric '{metric}'. Skipping this parameter.")
                    errors.append(np.nan)
                    count += 1
                    continue

                est_value = est_param_list[param_idx]
                gt_value = gt_param_list[param_idx]

                if est_value is None or gt_value is None:
                    logger.warning(
                        f"Missing values for Run ID '{run_id}', Metric '{metric}', Parameter index '{param_idx}'. Skipping this parameter.")
                    errors.append(np.nan)
                    count += 1
                    continue

                # Ensure values are numbers
                try:
                    est_value = float(est_value)
                    gt_value = float(gt_value)
                    error = est_value - gt_value  # compute error
                    normalizer = rmse_metric_normalizer.get(metric, [1])
                    if normalize_errors:
                        error /= normalizer[param_idx]
                        metric_errors.append(error)
                    else:
                        metric_errors.append(error)
                    count += 1
                except ValueError:
                    logger.warning(
                        f"Non-numeric values for Run ID '{run_id}', Metric '{metric}', Parameter index '{param_idx}'. Skipping this parameter.")
                    continue
            if metric_errors:
                run_total_errors.append(np.mean(metric_errors))
        if run_total_errors:
            errors.append(np.mean(run_total_errors))

    if count == 0:
        logger.error("No valid parameter comparisons found. RMSE and SE are undefined.")
        return np.nan, np.nan

    errors = np.array(errors)
    valid_errors = errors[~np.isnan(errors)]

    if len(valid_errors) == 0:
        logger.error("No valid errors after removing NaNs. Error metric is undefined.")
        return np.nan, np.nan

    squared_errors = valid_errors ** 2
    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)

    if use_std_dev:
        # Calculate standard deviation of errors
        error_metric = np.std(valid_errors, ddof=1)
        log_msg = f"Computed RMSE: {rmse:.4f} with SD: {error_metric:.4f} based on {count} parameter comparisons."
    else:
        # Bootstrapping to estimate SE of RMSE
        bootstrap_rmses = []
        for _ in range(num_bootstrap):
            sample_indices = np.random.choice(len(valid_errors), len(valid_errors), replace=True)
            sample_errors = valid_errors[sample_indices]
            sample_rmse = np.sqrt(np.mean(sample_errors ** 2))
            bootstrap_rmses.append(sample_rmse)

        error_metric = np.std(bootstrap_rmses, ddof=1)
        log_msg = f"Computed RMSE: {rmse:.4f} with SE: {error_metric:.4f} based on {count} parameter comparisons."

    logger.info(log_msg)
    return rmse, error_metric


def compute_rmse_per_metric(estimated_params, ground_truth_params, metrics, parameters, logger, main_param_only=False,
                            normalize_errors=False, rmse_metric_normalizer={}, use_std_dev=False):
    """
    Compute RMSE for each metric individually between estimated parameters and ground truth parameters.
    """
    rmse_per_metric = {}
    error_per_metric = {}

    for metric in metrics:
        errors = []
        for run_id, est_metrics in estimated_params.items():
            # Handle different run ID structures
            original_run_id = '_'.join(run_id.split('_')[:-1])
            gt_metrics = ground_truth_params.get(original_run_id, None)
            if gt_metrics is None:
                logger.warning(
                    f"Original Run ID '{original_run_id}' not found in ground truth data. Skipping synthetic run '{run_id}'.")
                errors.append(np.nan)
                continue
            metric_errors = []
            for param_idx in parameters.get(metric, []):
                if main_param_only and param_idx != 0:
                    continue
                # param_idx = 0  # Only consider the first parameter for each metric
                est_param_list = est_metrics.get(metric, [])
                gt_param_list = gt_metrics.get(metric, [])

                if len(est_param_list) <= param_idx or len(gt_param_list) <= param_idx:
                    logger.warning(
                        f"Missing parameter index {param_idx} for Run ID '{run_id}', Metric '{metric}'. Skipping this parameter.")
                    errors.append(np.nan)
                    continue

                est_value = est_param_list[param_idx]
                gt_value = gt_param_list[param_idx]

                if est_value is None or gt_value is None:
                    logger.warning(
                        f"Missing values for Run ID '{run_id}', Metric '{metric}', Parameter index '{param_idx}'. Skipping this parameter.")
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

                    logger.debug(
                        f"Run ID: {run_id} | Metric {metric} | Param Index {param_idx} | Comparing values: Estimated={est_value:.4f}, Ground Truth={gt_value:.4f}")
                except ValueError:
                    logger.warning(
                        f"Non-numeric values for Run ID '{run_id}', Metric '{metric}', Parameter index '{param_idx}'. Skipping this parameter.")
                    continue
            if metric_errors:
                errors.append(np.mean(metric_errors))

        # Calculate RMSE for this metric if there are valid errors
        valid_errors = np.array(errors)[~np.isnan(errors)]
        if len(valid_errors) > 0:
            squared_errors = valid_errors ** 2
            mse = np.mean(squared_errors)
            rmse = np.sqrt(mse)
            rmse_per_metric[metric] = rmse

            if use_std_dev:
                error_per_metric[metric] = np.std(valid_errors, ddof=1)
            else:
                error_per_metric[metric] = np.std(valid_errors, ddof=1) / np.sqrt(len(valid_errors))

            logger.info(f"Computed RMSE for metric '{metric}': {rmse:.4f}")
        else:
            rmse_per_metric[metric] = np.nan
            error_per_metric[metric] = np.nan
            logger.warning(f"No valid parameter comparisons found for metric '{metric}'. RMSE is undefined.")

    return rmse_per_metric, error_per_metric


def compute_kld_for_metric(est_param_list, gt_param_list, metric, metric_type, return_log_kld=True):
    """
    Compute the KLD for a given metric and its type.
    """
    if metric_type == 'binarySpan':
        # Compute KLD for each length from 2 to 10
        cum_kld = 0
        for length in range(2, 11):
            length_metric = f"{metric}_correct_w_len_{length}"
            base_dist = mle_params_to_dist(length_metric, gt_param_list, metric_type, metrics_dict=CURR_METRICS_DICT)
            predicted_dist = mle_params_to_dist(length_metric, est_param_list, metric_type,
                                                metrics_dict=CURR_METRICS_DICT)
            # Move distributions to the appropriate device
            base_dist = move_distribution_to_cuda(base_dist)
            predicted_dist = move_distribution_to_cuda(predicted_dist)

            # Calculate KLD
            kld = calculate_kld_given_metric(predicted_dist, base_dist, length_metric)
            cum_kld += kld
        return torch.mean(cum_kld)
    else:

        base_dist = mle_params_to_dist(metric, gt_param_list, metric_type, metrics_dict=CURR_METRICS_DICT)
        predicted_dist = mle_params_to_dist(metric, est_param_list, metric_type, metrics_dict=CURR_METRICS_DICT)
        # Move distributions to the appropriate device
        base_dist = move_distribution_to_cuda(base_dist)
        predicted_dist = move_distribution_to_cuda(predicted_dist)

    kld = calculate_kld_given_metric(predicted_dist, base_dist, metric)
    # Calculate KLD
    if return_log_kld:
        return torch.log(kld)
    else:
        return kld


def compute_kld(estimated_params, ground_truth_params, metrics, logger, num_bootstrap=1000, normalize_errors=False,
                rmse_metric_normalizer={}, use_std_dev=False):
    """
    Compute KL Divergence and either Standard Error (SE) or Standard Deviation (SD).

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
        use_std_dev (bool): Whether to use SE or SD
    Returns:
        tuple: (mean_kld, error_metric)
    """
    klds = []
    count = 0
    for run_id, est_metrics in estimated_params.items():
        original_run_id = '_'.join(run_id.split('_')[:-1])
        gt_metrics = ground_truth_params.get(original_run_id, None)
        if gt_metrics is None:
            logger.warning(
                f"Original Run ID '{original_run_id}' not found in ground truth data. Skipping synthetic run '{run_id}'.")
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

    if use_std_dev:
        error_metric = np.std(valid_klds, ddof=1)
        log_msg = f"Computed mean KL Divergence: {mean_kld:.4f} with SD: {error_metric:.4f} based on {count} comparisons."
    else:
        bootstrap_klds = []
        for _ in range(num_bootstrap):
            sample_indices = np.random.choice(len(valid_klds), len(valid_klds), replace=True)
            sample_klds = valid_klds[sample_indices]
            bootstrap_klds.append(np.mean(sample_klds))
        error_metric = float(np.std(bootstrap_klds, ddof=1))
        log_msg = f"Computed mean KL Divergence: {mean_kld:.4f} with SE: {error_metric:.4f} based on {count} comparisons."

    logger.info(log_msg)
    return mean_kld, error_metric


def compute_kld_per_metric(estimated_params, ground_truth_params, metrics, logger, normalize_errors=False,
                           rmse_metric_normalizer={}, use_std_dev=False):
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
    error_per_metric = {}

    for metric in metrics:
        klds = []
        for run_id, est_metrics in estimated_params.items():
            original_run_id = '_'.join(run_id.split('_')[:-1])
            gt_metrics = ground_truth_params.get(original_run_id, None)
            if gt_metrics is None:
                logger.warning(
                    f"Original Run ID '{original_run_id}' not found in ground truth data. Skipping synthetic run '{run_id}'.")
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

        valid_klds = np.array(klds)[~np.isnan(klds)]
        if len(valid_klds) > 0:
            mean_kld = np.mean(valid_klds)
            kld_per_metric[metric] = mean_kld

            if use_std_dev:
                error_per_metric[metric] = np.std(valid_klds, ddof=1)
            else:
                error_per_metric[metric] = np.std(valid_klds, ddof=1) / np.sqrt(len(valid_klds))

            logger.info(f"Computed KL Divergence for metric '{metric}': {mean_kld:.4f}")
        else:
            kld_per_metric[metric] = np.nan
            error_per_metric[metric] = np.nan
            logger.warning(f"No valid KL Divergence values for metric '{metric}'. Mean and error metric are undefined.")

    return kld_per_metric, error_per_metric


def plot_rmse_combined(df_plot, rmse_plot_path, all_n_values, ylabel, combined_plot_title, error_label='SD'):
    # Plot each DLVM model
    dlvm_models = df_plot[df_plot['Method'] == 'DLVM']['ModelID'].unique()
    # Plot RMSE vs N for IMLE and all DLVM models on the same graph, in both linear and log scales
    palette = sns.color_palette("tab10", n_colors=len(dlvm_models) + 1)
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
        if not df_plot[df_plot['Method'] == 'IMLE'].empty:
            df_imle = df_plot[df_plot['Method'] == 'IMLE'].sort_values('N')
            x_values = df_imle['N'] if scale == "linear" else np.log(df_imle['N'])
            mean_values = df_imle['Value']
            error_values = df_imle['Error']

            if error_label == 'SD':
                plt.plot(x_values, mean_values, '-o', color=model_colors['IMLE'], label='IMLE',
                         markersize=8, markerfacecolor='black', markeredgecolor='black', markeredgewidth=1.5)
                plt.fill_between(x_values, mean_values - error_values, mean_values + error_values,
                                 color=model_colors['IMLE'], alpha=0.2)

            else:
                plt.errorbar(
                    x_values,
                    mean_values,
                    yerr=error_values,
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
                    label=f'IMLE ± {error_label}'
                )

        for idx, model_id in enumerate(dlvm_models):
            df_model = df_plot[(df_plot['Method'] == 'DLVM') & (df_plot['ModelID'] == model_id)].sort_values('N')
            if not df_model.empty:
                x_values = df_model['N'] if scale == "linear" else np.log(df_model['N'])
                mean_values = df_model['Value']
                error_values = df_model['Error']

                if model_id.startswith('D2'):
                    plot_label = 'DLVM-2'
                elif model_id.startswith('D3'):
                    plot_label = 'DLVM-3'
                else:
                    plot_label = f'DLVM {model_id}'

                if error_label == 'SD':
                    plt.plot(x_values, mean_values, '-s', color=model_colors[model_id], label=plot_label,
                             markersize=8, markerfacecolor=model_colors[model_id], markeredgecolor='black',
                             markeredgewidth=1.5)
                    plt.fill_between(x_values, mean_values - error_values, mean_values + error_values,
                                     color=model_colors[model_id], alpha=0.2)
                else:
                    plt.errorbar(
                        x_values,
                        mean_values,
                        yerr=error_values,
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
                        label=f'{plot_label} ± {error_label}'
                    )

        # Use the fully formatted title passed from main()
        plt.title(combined_plot_title, fontsize=20, fontweight='bold')

        plt.xlabel(f'Number of Observations per Task', fontsize=16)
        plt.ylabel(ylabel, fontsize=16)

        # Set scales after plotting data
        # if scale == 'log':
        #     plt.xscale('log')  # This will handle the x-axis log transformation
        #     plt.yscale('log')  # This will handle the y-axis log transformation

        # Set ticks - use actual N values
        if scale == 'log' and all_n_values:
            plt.xticks(np.log(all_n_values), all_n_values, fontsize=12)
        elif all_n_values:
            plt.xticks(all_n_values, all_n_values, fontsize=12)
        plt.yticks(fontsize=12)

        plt.ylim(bottom=0)
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.legend(fontsize=14, loc='upper right', ncol=2)
        plt.tight_layout()

        # Save with scale indicator in filename
        scale_path = rmse_plot_path.replace('.pdf', f'_{scale}_scale.pdf')
        png_path = scale_path.replace('.pdf', '.png')
        plt.savefig(scale_path, dpi=300)
        plt.savefig(png_path, dpi=300)
        plt.close()
        print(f"Combined plot ({scale} scale) saved to '{scale_path}' and '{png_path}'.")


def plot_rmse_per_metric(metrics, results_per_metric, n_values, output_dir, file_name="",
                         ylabel='Root Mean Squared Error (RMSE)', title_subtitle='', error_label='SE'):
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
    palette = sns.color_palette("tab10", n_colors=len(models))
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

            for idx, model_id in enumerate(models):
                metric_data = results_per_metric.get(metric, {}).get(model_id, {})
                if not metric_data:
                    continue

                plot_n = sorted(metric_data.keys())
                values = np.array([metric_data[N]['Value'] for N in plot_n])
                errors = np.array([metric_data[N]['Error'] for N in plot_n])
                x_values = np.array(plot_n) if scale == 'linear' else np.log(np.array(plot_n))

                if model_id == "IMLE":
                    label = "IMLE"
                elif model_id.startswith('D2'):
                    label = 'DLVM-2'
                elif model_id.startswith('D3'):
                    label = 'DLVM-3'
                else:
                    label = f'DLVM {model_id}'

                if error_label == 'SD':
                    plt.plot(x_values, values, marker='o', linestyle='-', color=model_colors[model_id], label=label)
                    plt.fill_between(x_values, values - errors, values + errors, color=model_colors[model_id],
                                     alpha=0.2)
                else:
                    plt.errorbar(
                        x_values,
                        values,
                        yerr=errors,
                        marker='o',
                        linestyle='-',
                        color=model_colors[model_id],
                        label=label,
                        ecolor='lightgray',
                        elinewidth=2,
                        capsize=6
                    )

            # Create a main title and a subtitle from the passed info
            main_title = SUMMARIZED_METRICS_MAIN_MOMENTS_LABEL_DICT.get(metric, metric)
            if error_label == 'SD':
                full_title = f'{main_title}\n{title_subtitle}'
            else:
                full_title = f'{main_title}\n{title_subtitle} (± {error_label})'

            plt.title(full_title, fontsize=20, fontweight='bold')

            plt.xlabel(f'Number of Observations per Metric', fontsize=16)
            plt.ylabel(ylabel, fontsize=16)
            if scale == 'log' and n_values:
                plt.xticks(np.log(n_values), n_values, fontsize=12)
            elif n_values:
                plt.xticks(n_values, n_values, fontsize=12)
            plt.yticks(fontsize=12)
            plt.ylim(bottom=0)
            plt.grid(True, linestyle='--', linewidth=0.5)
            plt.legend(fontsize=14, loc='upper right')

            plt.tight_layout()

            # Save the plot
            sanitized_metric = re.sub(r'[^A-Za-z0-9]+', '_', metric)
            plot_path = os.path.join(output_dir, f"{sanitized_metric}_{file_name}_{scale}_scale.pdf")
            png_path = plot_path.replace('.pdf', '.png')
            plt.savefig(plot_path, dpi=300)
            plt.savefig(png_path, dpi=300)
            plt.close()
            print(f"Plot for metric '{metric}' ({scale} scale) saved to '{plot_path}' and '{png_path}'.")


def load_ground_truth_params(ground_truth_path, logger):
    """
    Load ground truth parameters from a .pt file.
    """
    if not os.path.exists(ground_truth_path):
        logger.error(f"Ground truth parameters file not found at {ground_truth_path}")
        sys.exit(1)

    try:
        ground_truth_params = torch.load(ground_truth_path, map_location='cpu')
        logger.info(
            f"Ground truth parameters loaded successfully from '{ground_truth_path}' with {len(ground_truth_params)} runs.")
        return ground_truth_params
    except Exception as e:
        logger.error(f"Failed to load ground truth parameters from '{ground_truth_path}': {e}")
        sys.exit(1)


def load_estimated_params(params_path, logger, param_type="IMLE"):
    """
    Load estimated parameters (IMLE or DLVM) from a .pt file.
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
    parser.add_argument("--ground_truth_param_file", type=str, default=None,
                        help="Path to the ground truth parameters file")
    parser.add_argument("--params_dir_name", type=str, default="param_fits",
                        help="Name of the directory containing the estimated parameter files (e.g., '5_sim_10_restarts_param_fits')")
    parser.add_argument("--main_param_only", action='store_true',
                        help="Only compute metrics for the main parameter of each metric")
    parser.add_argument("--normalize_errors", action='store_true', help="Normalize errors based on metric type")
    parser.add_argument("--metric", type=str, choices=["rmse", "kld"], default="rmse",
                        help="Metric to compute and plot: Root Mean Squared Error (rmse) or KL Divergence (kld)")
    parser.add_argument("--plot_std_dev", action='store_true',
                        help="Plot Standard Deviation instead of Standard Error as error bars.")
    args = parser.parse_args()

    # Define dataset name
    dataset_name = DATASET  # Replace with your actual dataset name

    # Define directories using relative paths
    # Ground truth is in '../../data/COL10/all_data-best_mle_params_mpf100.pt'
    if args.ground_truth_param_file is not None:
        ground_truth_params_path = args.ground_truth_param_file
        ground_truth_params_path = os.fspath(data_dir(dataset_name) / args.ground_truth_param_file)
    else:
        ground_truth_params_path = os.fspath(data_dir(dataset_name) / "all_data-best_mle_params_mpf100.pt")

    # Use the provided directory name for parameter fits
    params_dir = os.path.join(SCRIPT_DIR, 'synthetic_data', dataset_name, args.params_dir_name)

    # Create a dynamic plot directory name based on the params_dir_name
    plot_dir_name = f"{args.params_dir_name}_plots"
    plots_dir = os.path.join(SCRIPT_DIR, 'synthetic_data', dataset_name, plot_dir_name)

    # Ensure plots_dir exists
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Created/ensured directory exists: {plots_dir}")

    # Check for params_dir existence now that it's dynamic
    if not os.path.exists(params_dir):
        print(f"Warning: The specified parameters directory '{params_dir}' does not exist. The script might fail.")

    # Parse params_dir_name to create dynamic, descriptive titles
    title_pattern = re.compile(r'(\d+)_sim_(\d+)_restarts')
    match = title_pattern.search(args.params_dir_name)

    if match:
        sim_count = match.group(1)
        restart_count = match.group(2)
        combined_plot_title = (
            f"IMLE & {restart_count} restarts DLVM ({dataset_name}) Predictive Performance\n"
            f"on Simulation ({sim_count} times) Data"
        )
        per_metric_subtitle = f"({sim_count} sims, {restart_count} restarts)"
    else:
        # Fallback to the previous format if the directory name doesn't match
        combined_plot_title = f"IMLE & DLVM ({dataset_name}) Predictive Performance"
        per_metric_subtitle = f""

    if args.metric == "kld":
        normalizer_params = {
            'CorsiComplex': 6.5, 'Stroop_reaction_time': 2.5,
            'RunningSpan_correct_w_len_2': 0.5, 'Countermanding_reaction_time': 2.5,
            'SimpleSpan': 6.5, 'RunningSpan_correct_w_len_3': 0.5,
            'D2_hit_accuracy': 0.5, 'PasatPlus_correctly_answered': 0.5,
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
    log_file = os.path.join(plots_dir, 'script_run.log')
    logger = setup_logging(log_file)
    logger.info(f"===== Starting Plotting Script (Error Metric: {'SD' if args.plot_std_dev else 'SE'}) =====")
    logger.info(f"Reading estimated parameters from: {params_dir}")
    logger.info(f"Saving plots and logs to: {plots_dir}")

    # Load ground truth parameters
    ground_truth_params = load_ground_truth_params(ground_truth_params_path, logger)
    # remove ID from ground truth params if they are in OUTLIER_HELDOUT_SESSIONS
    ground_truth_params = {k: v for k, v in ground_truth_params.items() if k not in OUTLIER_HELDOUT_SESSIONS}
    logger.info(
        f"Removed {len(OUTLIER_HELDOUT_SESSIONS)} outlier heldout sessions from ground truth parameters, {len(ground_truth_params)} runs remaining")

    # Extract metrics and parameter indices from ground truth
    metrics = SUMMARIZED_METRICS
    parameters = {}
    for metric in metrics:
        metric_type = SUMMARIZED_METRICS_METRIC_TYPES.get(metric, 'binary')
        parameters[metric] = [0, 1] if metric_type in ['binarySpan', 'timing'] else [0]

    logger.debug(f"Metrics: {metrics}")
    logger.debug(f"Parameters: {parameters}")

    # Identify all DLVM parameter files and extract model_ids
    try:
        synthetic_dlvm_files = [f for f in os.listdir(params_dir)
                                if f.startswith("synthetic_dlvm_params_") and f.endswith(".pt")]
    except FileNotFoundError:
        logger.error(f"Parameters directory '{params_dir}' not found.")
        sys.exit(1)

    # Extract unique model_ids from filenames
    model_ids = set()
    pattern = re.compile(r'synthetic_dlvm_params_((?:[a-zA-Z_]+_)*)(D\d+)_N(\d+)\.pt')
    model_n_values = {}

    for dlvm_file in synthetic_dlvm_files:
        match = pattern.match(dlvm_file)
        if match:
            method_prefix, dimension_id, n_value_str = match.groups()
            n_value = int(n_value_str)
            model_id = f"{dimension_id}_{method_prefix.strip('_')}" if method_prefix else dimension_id
            model_ids.add(model_id)
            model_n_values.setdefault(model_id, []).append(n_value)

    model_ids = sorted(list(model_ids))
    logger.info(f"Identified DLVM models: {model_ids}")

    all_models = ['IMLE'] + model_ids
    results_dict = {model_id: {} for model_id in all_models}
    results_per_metric = {m: {model_id: {} for model_id in all_models} for m in metrics}

    # Extract all N values
    all_n_values = set()
    try:
        synthetic_mle_files = [f for f in os.listdir(params_dir)
                               if f.startswith("synthetic_mle_params_N") and f.endswith(".pt")]
        N_values_imle = sorted(
            [int(f.replace("synthetic_mle_params_N", "").replace(".pt", "")) for f in synthetic_mle_files])
        all_n_values.update(N_values_imle)
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"Could not process IMLE files: {e}")
        N_values_imle = []

    for n_list in model_n_values.values():
        all_n_values.update(n_list)
    all_n_values = sorted(list(all_n_values))

    # Create parameter dictionaries for each computation type to avoid passing unexpected arguments

    rmse_compute_params = {
        "ground_truth_params": ground_truth_params, "metrics": metrics, "parameters": parameters,
        "logger": logger, "main_param_only": args.main_param_only, "normalize_errors": args.normalize_errors,
        "rmse_metric_normalizer": normalizer_params, "use_std_dev": args.plot_std_dev
    }

    kld_compute_params = {
        "ground_truth_params": ground_truth_params, "metrics": metrics,
        "logger": logger, "normalize_errors": args.normalize_errors,
        "rmse_metric_normalizer": normalizer_params, "use_std_dev": args.plot_std_dev
    }

    # Loop over each N to compute the chosen metric for IMLE
    for N in N_values_imle:
        imle_params_path = os.path.join(params_dir, f"synthetic_mle_params_N{N}.pt")
        imle_params = load_estimated_params(imle_params_path, logger, param_type="IMLE")
        if imle_params:
            if args.metric == "rmse":
                value, error = compute_rmse(imle_params, **rmse_compute_params)
                per_metric_values, per_metric_errors = compute_rmse_per_metric(imle_params, **rmse_compute_params)
            else:  # kld
                value, error = compute_kld(imle_params, **kld_compute_params)
                per_metric_values, per_metric_errors = compute_kld_per_metric(imle_params, **kld_compute_params)

            results_dict['IMLE'][N] = {'Value': value, 'Error': error}
            for metric, metric_value in per_metric_values.items():
                results_per_metric[metric]['IMLE'][N] = {'Value': metric_value, 'Error': per_metric_errors.get(metric)}

    # Now, loop over each DLVM model and compute the chosen metric
    for D in model_ids:
        logger.info(f"Processing Model: {D}")
        for N in sorted(list(set(model_n_values.get(D, [])))):
            parts = D.split('_', 1)
            dlvm_file_name = f"synthetic_dlvm_params_{parts[1]}_{parts[0]}_N{N}.pt" if len(
                parts) > 1 else f"synthetic_dlvm_params_{D}_N{N}.pt"
            dlvm_params_path = os.path.join(params_dir, dlvm_file_name)
            dlvm_params = load_estimated_params(dlvm_params_path, logger, param_type="DLVM")

            if dlvm_params:
                if args.metric == "rmse":
                    value, error = compute_rmse(dlvm_params, **rmse_compute_params)
                    per_metric_values, per_metric_errors = compute_rmse_per_metric(dlvm_params, **rmse_compute_params)
                else:  # kld
                    value, error = compute_kld(dlvm_params, **kld_compute_params)
                    per_metric_values, per_metric_errors = compute_kld_per_metric(dlvm_params, **kld_compute_params)

                results_dict[D][N] = {'Value': value, 'Error': error}
                for metric, metric_value in per_metric_values.items():
                    results_per_metric[metric][D][N] = {'Value': metric_value, 'Error': per_metric_errors.get(metric)}

    # Now, prepare data for combined plotting
    plot_data = []
    for model_id, res_by_n in results_dict.items():
        for N, data in res_by_n.items():
            plot_data.append({
                'Method': 'IMLE' if model_id == 'IMLE' else 'DLVM', 'ModelID': model_id,
                'N': N, 'Value': data['Value'], 'Error': data['Error']
            })
    df_plot = pd.DataFrame(plot_data)

    if not df_plot.empty:
        # Proceed with plotting
        metric_name = "RMSE" if args.metric == "rmse" else "KLD"
        error_label = "SD" if args.plot_std_dev else "SE"

        error_metric_suffix = 'stddev' if args.plot_std_dev else 'stderr'
        norm_suffix  = 'normalized' if args.normalize_errors else 'unnormalized'
        final_suffix = f"{error_metric_suffix}_{norm_suffix}"
        plot_path = os.path.join(plots_dir, f"{metric_name}_combined_{dataset_name}_{final_suffix}.pdf")
        ylabel = f"{'Normalized ' if args.normalize_errors else ''}{'Root Mean Squared Error (RMSE)' if args.metric == 'rmse' else 'KL Divergence (KLD)'}"

        plot_rmse_combined(df_plot, plot_path, all_n_values, ylabel=ylabel,
                           combined_plot_title=combined_plot_title, error_label=error_label)

        # Save results to CSV
        csv_path = os.path.join(plots_dir, f"{args.metric}_results_combined_{dataset_name}_{final_suffix}.csv")
        df_plot.to_csv(csv_path, index=False)
        logger.info(f"{args.metric.upper()} results saved to '{csv_path}'.")

        # Generate plots for each metric
        logger.info(f"Generating {args.metric.upper()} vs N plots for each individual metric.")
        file_name = f"{args.metric}_per_metric_{dataset_name}_{final_suffix}"

        plot_rmse_per_metric(metrics, results_per_metric, all_n_values, plots_dir, file_name=file_name,
                             ylabel=ylabel, title_subtitle=per_metric_subtitle, error_label=error_label)
    else:
        logger.warning("No data available for plotting. Ensure that the parameter files are correctly generated.")

    logger.info(f"===== {args.metric.upper()} Plotting Script Completed Successfully =====")


# ============================================
# Execute Main Function
# ============================================

if __name__ == "__main__":
    main()
