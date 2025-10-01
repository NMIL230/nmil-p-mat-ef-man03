# plot_rmse_modernized.py

"""
plot_rmse_modernized.py

This script computes the Root Mean Squared Error (RMSE) between IMLE-fitted parameters from synthetic data
and ground truth parameters across varying N (number of observations per metric) and specific DLVM models.
It then plots RMSE against N for IMLE and all DLVM models on the same graph to visualize performance improvements.
Additionally, it generates separate RMSE vs N plots for each metric to provide detailed insights into each metric's performance.

MODERNIZED VERSION: Uses plot_generic_comparison for consistent plotting across the codebase.
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
from typing import Dict, List

# Configure Seaborn for better aesthetics (optional)
sns.set_theme(style="whitegrid")

# Determine the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory to sys.path to access 'utils' module
parent_dir = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.append(parent_dir)

from utils.data_distribution_utils import (OUTLIER_HELDOUT_SESSIONS, DATASET,
                                           SUMMARIZED_METRICS_MAIN_MOMENTS_LABEL_DICT,
                                           SUMMARIZED_METRICS, SUMMARIZED_METRICS_METRIC_TYPES, mle_params_to_dist,
                                           CURR_METRICS_DICT)  # Ensure this is defined in your utils
from utils.active_learning_utils import calculate_kld_given_metric, move_distribution_to_cuda

# Import the new plotting utilities
from analysis.analysis_utils.utils_plot import plot_generic_comparison


# ============================================
# Method Properties for Styling
# ============================================

def build_method_properties(all_methods: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Build method properties for IMLE and DLVM models used in this script.
    """
    method_config = {
        # IMLE methods - use circles with dimgray color
        "IMLE": {
            "color": "dimgray",
            "marker": "o",
            "label": "IMLE"
        },
        "IMLE_D1": {
            "color": "dimgray",
            "marker": "o", 
            "label": "IMLE (D1)"
        },
        "IMLE_D2": {
            "color": "dimgray",
            "marker": "o",
            "label": "IMLE (D2)"
        },
        "IMLE_D3": {
            "color": "dimgray",
            "marker": "o",
            "label": "IMLE (D3)"
        },
        
        # DLVM methods - use squares with tab10 colors
        "D1_gradient_descent": {
            "color": "#d62728",  # Red
            "marker": "s",
            "label": "DLVM-1"
        },
        "D2_gradient_descent": {
            "color": "#1f77b4",  # Blue
            "marker": "s",
            "label": "DLVM-2"
        },
        "D3_gradient_descent": {
            "color": "#ff7f0e",  # Orange
            "marker": "s",
            "label": "DLVM-3"
        },
        "D1_grid_search": {
            "color": "#d62728",  # Red
            "marker": "^",
            "label": "DLVM-1 (Grid)"
        },
        "D2_grid_search": {
            "color": "#1f77b4",  # Blue
            "marker": "^",
            "label": "DLVM-2 (Grid)"
        },
        "D3_grid_search": {
            "color": "#ff7f0e",  # Orange
            "marker": "^",
            "label": "DLVM-3 (Grid)"
        }
    }
    
    props = {}
    for method in all_methods:
        if method in method_config:
            props[method] = method_config[method]
        else:
            # For unmapped methods, provide defaults based on patterns
            if method.startswith('IMLE'):
                props[method] = {
                    "color": "dimgray", 
                    "marker": "o", 
                    "label": method.replace('_', ' ')
                }
            elif method.startswith('D1'):
                props[method] = {
                    "color": "#d62728", 
                    "marker": "s", 
                    "label": f"DLVM-1 {method.split('_', 1)[1] if '_' in method else ''}"
                }
            elif method.startswith('D2'):
                props[method] = {
                    "color": "#1f77b4", 
                    "marker": "s", 
                    "label": f"DLVM-2 {method.split('_', 1)[1] if '_' in method else ''}"
                }
            elif method.startswith('D3'):
                props[method] = {
                    "color": "#ff7f0e", 
                    "marker": "s", 
                    "label": f"DLVM-3 {method.split('_', 1)[1] if '_' in method else ''}"
                }
            else:
                # Generic fallback
                props[method] = {
                    "color": "gray", 
                    "marker": "x", 
                    "label": method
                }
    
    return props


# ============================================
# Logging and Utility Functions
# ============================================

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )
    logger = logging.getLogger()
    return logger


def get_dimension_from_model_id(model_id):
    """
    Extracts the dimension (e.g., 'D2', 'D3') from a model ID string.
    """
    if not model_id:
        return None
    match = re.search(r'(D\d+)', model_id)
    if match:
        return match.group(1)
    return None


# ============================================
# Core Computation Functions
# ============================================

def compute_rmse(estimated_params, ground_truth_params, metrics, parameters, logger, num_bootstrap=1000,
                 main_param_only=False, normalize_errors=False, rmse_metric_normalizer={}, use_std_dev=False):
    errors = []
    count = 0
    for run_id, est_metrics in estimated_params.items():
        original_run_id = '_'.join(run_id.split('_')[:-1])
        gt_metrics = ground_truth_params.get(original_run_id, None)
        if gt_metrics is None:
            logger.warning(
                f"Original Run ID '{original_run_id}' not found in ground truth data. Skipping synthetic run '{run_id}'.")
            continue
        run_total_errors = []
        for metric in metrics:
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
                try:
                    est_value = float(est_value)
                    gt_value = float(gt_value)
                    error = est_value - gt_value
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
        return np.nan, np.nan
    errors = np.array(errors)
    valid_errors = errors[~np.isnan(errors)]
    if len(valid_errors) == 0:
        return np.nan, np.nan
    squared_errors = valid_errors ** 2
    rmse = np.sqrt(np.mean(squared_errors))
    if use_std_dev:
        error_metric = np.std(valid_errors, ddof=1)
    else:
        bootstrap_rmses = [np.sqrt(np.mean(np.random.choice(valid_errors, len(valid_errors), replace=True) ** 2)) for _
                           in range(num_bootstrap)]
        error_metric = np.std(bootstrap_rmses, ddof=1)
    return rmse, error_metric


def compute_rmse_per_metric(estimated_params, ground_truth_params, metrics, parameters, logger, main_param_only=False,
                            normalize_errors=False, rmse_metric_normalizer={}, use_std_dev=False):
    rmse_per_metric = {}
    error_per_metric = {}
    for metric in metrics:
        errors = []
        for run_id, est_metrics in estimated_params.items():
            original_run_id = '_'.join(run_id.split('_')[:-1])
            gt_metrics = ground_truth_params.get(original_run_id, None)
            if gt_metrics is None:
                errors.append(np.nan)
                continue
            metric_errors = []
            for param_idx in parameters.get(metric, []):
                if main_param_only and param_idx != 0:
                    continue
                est_param_list = est_metrics.get(metric, [])
                gt_param_list = gt_metrics.get(metric, [])
                if len(est_param_list) <= param_idx or len(gt_param_list) <= param_idx:
                    errors.append(np.nan)
                    continue
                est_value = est_param_list[param_idx]
                gt_value = gt_param_list[param_idx]
                if est_value is None or gt_value is None:
                    errors.append(np.nan)
                    continue
                try:
                    est_value = float(est_value)
                    gt_value = float(gt_value)
                    error = est_value - gt_value
                    normalizer = rmse_metric_normalizer.get(metric, [1])
                    if normalize_errors:
                        error /= normalizer[param_idx]
                        metric_errors.append(error)
                    else:
                        metric_errors.append(error)
                except ValueError:
                    continue
            if metric_errors:
                errors.append(np.mean(metric_errors))
        valid_errors = np.array(errors)[~np.isnan(errors)]
        if len(valid_errors) > 0:
            rmse = np.sqrt(np.mean(valid_errors ** 2))
            rmse_per_metric[metric] = rmse
            if use_std_dev:
                error_per_metric[metric] = np.std(valid_errors, ddof=1)
            else:
                error_per_metric[metric] = np.std(valid_errors, ddof=1) / np.sqrt(len(valid_errors))
        else:
            rmse_per_metric[metric] = np.nan
            error_per_metric[metric] = np.nan
    return rmse_per_metric, error_per_metric


def compute_kld_for_metric(est_param_list, gt_param_list, metric, metric_type, return_log_kld=True):
    if metric_type == 'binarySpan':
        cum_kld = 0
        for length in range(2, 11):
            length_metric = f"{metric}_correct_w_len_{length}"
            base_dist = mle_params_to_dist(length_metric, gt_param_list, metric_type, metrics_dict=CURR_METRICS_DICT)
            predicted_dist = mle_params_to_dist(length_metric, est_param_list, metric_type,
                                                metrics_dict=CURR_METRICS_DICT)
            base_dist = move_distribution_to_cuda(base_dist)
            predicted_dist = move_distribution_to_cuda(predicted_dist)
            kld = calculate_kld_given_metric(predicted_dist, base_dist, length_metric)
            cum_kld += kld
        return torch.mean(cum_kld)
    else:
        base_dist = mle_params_to_dist(metric, gt_param_list, metric_type, metrics_dict=CURR_METRICS_DICT)
        predicted_dist = mle_params_to_dist(metric, est_param_list, metric_type, metrics_dict=CURR_METRICS_DICT)
        base_dist = move_distribution_to_cuda(base_dist)
        predicted_dist = move_distribution_to_cuda(predicted_dist)
        kld = calculate_kld_given_metric(predicted_dist, base_dist, metric)
        return torch.log(kld) if return_log_kld else kld


def compute_kld(estimated_params, ground_truth_params, metrics, logger, num_bootstrap=1000, normalize_errors=False,
                rmse_metric_normalizer={}, use_std_dev=False):
    klds = []
    count = 0
    for run_id, est_metrics in estimated_params.items():
        original_run_id = '_'.join(run_id.split('_')[:-1])
        gt_metrics = ground_truth_params.get(original_run_id, None)
        if gt_metrics is None:
            continue
        run_total_klds = []
        for metric in metrics:
            est_param_list = est_metrics.get(metric, [])
            gt_param_list = gt_metrics.get(metric, [])
            if not est_param_list or not gt_param_list:
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
    valid_klds = np.array(klds)[~np.isnan(klds)]
    if len(valid_klds) == 0:
        return np.nan, np.nan
    mean_kld = float(np.mean(valid_klds))
    if use_std_dev:
        error_metric = np.std(valid_klds, ddof=1)
    else:
        bootstrap_klds = [np.mean(np.random.choice(valid_klds, len(valid_klds), replace=True)) for _ in
                          range(num_bootstrap)]
        error_metric = float(np.std(bootstrap_klds, ddof=1))
    return mean_kld, error_metric


def compute_kld_per_metric(estimated_params, ground_truth_params, metrics, logger, normalize_errors=False,
                           rmse_metric_normalizer={}, use_std_dev=False):
    kld_per_metric = {}
    error_per_metric = {}
    for metric in metrics:
        klds = []
        for run_id, est_metrics in estimated_params.items():
            original_run_id = '_'.join(run_id.split('_')[:-1])
            gt_metrics = ground_truth_params.get(original_run_id, None)
            if gt_metrics is None:
                klds.append(np.nan)
                continue
            est_param_list = est_metrics.get(metric, [])
            gt_param_list = gt_metrics.get(metric, [])
            if not est_param_list or not gt_param_list or None in est_param_list or None in gt_param_list:
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
        else:
            kld_per_metric[metric] = np.nan
            error_per_metric[metric] = np.nan
    return kld_per_metric, error_per_metric


# ============================================
# MODERNIZED Plotting Functions
# ============================================

def plot_combined_comparison(df_plot, plots_dir, dataset_name, final_suffix, ylabel, combined_plot_title, error_type='SD'):
    """
    Create combined plots using plot_generic_comparison for both linear and log scales.
    """
    method_properties = build_method_properties(df_plot['ModelID'].unique())
    
    # Convert ModelID to Method for the generic plotter 
    df_plot_copy = df_plot.copy()
    df_plot_copy['Method'] = df_plot_copy['ModelID']
    
    for scale in ['linear', 'log']:
        plot_path = os.path.join(plots_dir, f"{dataset_name}_combined_{final_suffix}_{scale}_scale.pdf")
        
        # Determine y_scale based on metric type
        y_scale = 'log' if 'KLD' in ylabel or 'kld' in final_suffix.lower() else 'linear'
        
        plot_generic_comparison(
            df_plot=df_plot_copy,
            output_path=plot_path,
            x_scale=scale,
            y_scale='linear',
            show_markers=True,
            error_type=error_type,
            title=combined_plot_title,
            xlabel='Number of Observations per Task',
            ylabel=ylabel,
            method_properties=method_properties,
            xlim=None, ylim=[0.0, 0.4],  # Use auto limits
            x_margin=0.05, y_margin=0.08, min_decades=0.15,
            x_ticks=[1,2,5,10,20,50,100,200] if scale == 'log' else None,
            show_grid=False,
            legend_ncol=1
        )
        
        # Also create PNG version
        png_path = plot_path.replace('.pdf', '.png')
        plot_generic_comparison(
            df_plot=df_plot_copy,
            output_path=png_path,
            x_scale=scale,
            y_scale='linear',
            show_markers=True,
            error_type=error_type,
            title=combined_plot_title,
            xlabel='Number of Observations per Task',
            ylabel=ylabel,
            method_properties=method_properties,
            xlim=[0, 250], ylim=[0.0, 0.4],  # Use auto limits
            x_margin=0.05, y_margin=0.08, min_decades=0.15,
            x_ticks=[1,2,5,10,20,50,100,200] if scale == 'log' else None,
            show_grid=False,
            legend_ncol=1
        )


def plot_per_metric_comparison(metrics, results_per_metric, plots_dir, file_name, ylabel, title_subtitle='', error_type='SE'):
    """
    Create per-metric plots using plot_generic_comparison for both linear and log scales.
    """
    if not results_per_metric:
        return
    
    first_metric = next(iter(results_per_metric))
    all_methods = sorted(results_per_metric[first_metric].keys())
    method_properties = build_method_properties(all_methods)
    
    for metric in metrics:
        # Prepare data for this metric
        plot_data = []
        for method_id in all_methods:
            metric_data = results_per_metric.get(metric, {}).get(method_id, {})
            for N, data in metric_data.items():
                plot_data.append({
                    'Method': method_id,
                    'N': N,
                    'Value': data['Value'],
                    'Error': data['Error']
                })
        
        if not plot_data:
            continue
            
        df_metric = pd.DataFrame(plot_data)
        
        for scale in ['linear', 'log']:
            main_title = SUMMARIZED_METRICS_MAIN_MOMENTS_LABEL_DICT.get(metric, metric)
            full_title = f'{main_title}\n{title_subtitle}' if title_subtitle else main_title
            
            sanitized_metric = re.sub(r'[^A-Za-z0-9]+', '_', metric)
            plot_path = os.path.join(plots_dir,"metric_specific", f"{sanitized_metric}_{file_name}_{scale}_scale.pdf")
            
            # Determine y_scale based on metric type
            y_scale = 'log' if 'KLD' in ylabel or 'kld' in file_name.lower() else 'linear'
            
            plot_generic_comparison(
                df_plot=df_metric,
                output_path=plot_path,
                x_scale=scale,
                y_scale=y_scale,
                show_markers=True,
                error_type=error_type,
                title=full_title,
                xlabel='Number of Observations per Metric',
                ylabel=ylabel,
                method_properties=method_properties,
                xlim=None, ylim=None,  # Use auto limits
                x_margin=0.05, y_margin=0.08, min_decades=0.15
            )
            
            # Also create PNG version
            png_path = plot_path.replace('.pdf', '.png')
            plot_generic_comparison(
                df_plot=df_metric,
                output_path=png_path,
                x_scale=scale,
                y_scale=y_scale,
                show_markers=True,
                error_type=error_type,
                title=full_title,
                xlabel='Number of Observations per Metric',
                ylabel=ylabel,
                method_properties=method_properties,
                xlim=None, ylim=None,  # Use auto limits
                x_margin=0.05, y_margin=0.08, min_decades=0.15
            )


# ============================================
# File Loading Function
# ============================================

def load_estimated_params(params_path, logger, param_type="IMLE"):
    if not os.path.exists(params_path):
        logger.warning(f"{param_type} parameters file not found at {params_path}")
        return None
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
    parser = argparse.ArgumentParser(description="Plot RMSE or KL Divergence curves for all models using modernized plotting")
    parser.add_argument("--ground_truth_dir", type=str, default=None,
                        help="Path to the directory containing ground truth parameter files.")
    parser.add_argument("--params_dir", type=str, default="param_fits",
                        help="Path of the directory containing the estimated parameter files.")
    parser.add_argument("--main_param_only", action='store_true',
                        help="Only compute metrics for the main parameter of each metric")
    parser.add_argument("--normalize_errors", action='store_true', help="Normalize errors based on metric type")
    parser.add_argument("--metric", type=str, choices=["rmse", "kld"], default="rmse",
                        help="Metric to compute and plot: RMSE or KLD")
    parser.add_argument("--plot_std_dev", action='store_true',
                        help="Plot Standard Deviation instead of Standard Error as error bars.")
    parser.add_argument("--dlvm_search_method", type=str, choices=["grid_search", "gradient_descent"], default="gradient_descent",
                        help="The method used to fit the DLVM parameters.")
    parser.add_argument("--eval_dataset_type", type=str, default="validation_simulated", choices=["validation_simulated", "training_set"], help="Type of experiment (validation_simulated or training_simulated)")
    parser.add_argument("--show_single_imle_plot", action='store_true', default=False, help="Show a single IMLE plot")
    parser.add_argument("--show_single_dlvm_plot", action='store_true', default=False, help="Show a single DLVM plot")
    args = parser.parse_args()
    N_MAX_PLOT = 200

    dataset_name = DATASET
    params_dir = args.params_dir
    plot_dir_name = f"plots"
    vis_type = "main_figure" if args.show_single_dlvm_plot else "appendix_figure"
    plots_dir = os.path.join(params_dir, plot_dir_name, args.eval_dataset_type, vis_type)
    os.makedirs(plots_dir, exist_ok=True)

    log_file = os.path.join(plots_dir, 'script_run_modernized.log')
    logger = setup_logging(log_file)
    logger.info(f"===== Starting MODERNIZED Plotting Script (Error Metric: {'SD' if args.plot_std_dev else 'SE'}) =====")

    ground_truth_base_dir = args.ground_truth_dir
    if ground_truth_base_dir:
        ground_truth_base_dir = os.path.expanduser(ground_truth_base_dir)
        logger.info(f"User provided ground truth directory, expanded path: {ground_truth_base_dir}")
    else:
        ground_truth_base_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', dataset_name))
        logger.info(f"--ground_truth_dir not specified, defaulting to {ground_truth_base_dir}")

    all_ground_truths = {}
    if args.eval_dataset_type == "training_set":
        gt_pattern = re.compile(r'all_data-best_mle_params_mpf(\d+)\.pt')
        # check if this file exists
        if not os.path.exists(os.path.join(ground_truth_base_dir, 'all_data-best_mle_params_mpf100.pt')):
            logger.error(f"Ground truth file not found: {os.path.join(ground_truth_base_dir, 'all_data-best_mle_params_mpf100.pt')}")
            sys.exit(1)
        gt_path = os.path.join(ground_truth_base_dir, 'all_data-best_mle_params_mpf100.pt')
        params = load_estimated_params(gt_path, logger, param_type="Ground Truth")
        if params:
            params = {k: v for k, v in params.items() if k not in OUTLIER_HELDOUT_SESSIONS}
            all_ground_truths["D1"] = params
            all_ground_truths["D2"] = params
            all_ground_truths["D3"] = params
        else:
            logger.error(f"Failed to load ground truth file: {gt_path}")
            sys.exit(1)
    else:
        gt_pattern = re.compile(r'(D\d+)_synthetic_ground_truth_parameters_.*\.pt') 
        try:
            for f in os.listdir(ground_truth_base_dir):
                match = gt_pattern.match(f)
                if match:
                    dimension = match.group(1)
                    gt_path = os.path.join(ground_truth_base_dir, f)
                    logger.info(f"Found ground truth file for dimension {dimension}: {f}")
                    params = load_estimated_params(gt_path, logger, param_type=f"Ground Truth {dimension}")
                    if params:
                        params = {k: v for k, v in params.items() if k not in OUTLIER_HELDOUT_SESSIONS}
                        all_ground_truths[dimension] = params
        except FileNotFoundError:
            logger.error(f"Ground truth directory not found: {ground_truth_base_dir}")
            sys.exit(1)

    if not all_ground_truths:
        logger.error(f"No ground truth files found in {ground_truth_base_dir} matching the pattern 'D*_synthetic_...'.")
        sys.exit(1)
    logger.info(f"Successfully loaded ground truths for dimensions: {list(all_ground_truths.keys())}")

    title_pattern = re.compile(r'(\d+)_sim_(\d+)_restarts')
    match = title_pattern.search(args.params_dir)
    if match:
        sim_count, restart_count = match.groups()
        combined_plot_title = f"IMLE & {restart_count} restarts DLVM ({dataset_name}) Predictive Performance\non Simulation ({sim_count} times) Data"
        per_metric_subtitle = f"({sim_count} sims, {restart_count} restarts)"
    else:
        combined_plot_title = f"IMLE and DLVM Fits on the Same Data"
        per_metric_subtitle = ""

    if args.metric == "kld":
        normalizer_params = {'CorsiComplex': 6.5, 'Stroop_reaction_time': 2.5, 'RunningSpan_correct_w_len_2': 0.5,
                             'Countermanding_reaction_time': 2.5, 'SimpleSpan': 6.5, 'RunningSpan_correct_w_len_3': 0.5,
                             'D2_hit_accuracy': 0.5, 'PasatPlus_correctly_answered': 0.5}
    else:
        normalizer_params = {'CorsiComplex': [12.234659463167189, 1.9972472935914993],
                             'Stroop_reaction_time': [1.0243954658508292, 0.5401461943984033],
                             'RunningSpan_correct_w_len_2': [0.9800000097602607],
                             'Countermanding_reaction_time': [0.7413206100463867, 0.38473375886678696],
                             'SimpleSpan': [12.079994469881058, 1.9972472935914993],
                             'RunningSpan_correct_w_len_3': [0.9800000097602607],
                             'D2_hit_accuracy': [0.6566666662693024],
                             'PasatPlus_correctly_answered': [0.8899999763816596]}

    metrics = SUMMARIZED_METRICS
    parameters = {m: ([0, 1] if SUMMARIZED_METRICS_METRIC_TYPES.get(m) in ['binarySpan', 'timing'] else [0]) for m in
                  metrics}

    try:
        all_folders_in_dir = os.listdir(params_dir)
    except FileNotFoundError:
        logger.error(f"Parameters directory '{params_dir}' not found.")
        sys.exit(1)
    # Filter folders based on eval_dataset_type using proper regex patterns
    if args.eval_dataset_type == "training_set":
        # For training_set, include only folders that start with D followed by digits and _all_data
        all_data_pattern = re.compile(r'^D\d+_all_data')
        all_folders_in_dir = [folder for folder in all_folders_in_dir if all_data_pattern.match(folder)]
    else:
        # For validation_simulated, exclude folders that start with D followed by digits and _synthetic_
        synthetic_pattern = re.compile(r'^D\d+_synthetic_')
        all_folders_in_dir = [folder for folder in all_folders_in_dir if synthetic_pattern.match(folder)]
    
    # Read the folders and identify the DLVM models
    dlvm_model_ids = set()
    model_n_values = {}
    model_id_to_folder = {}
    
    for folder in all_folders_in_dir:
        if folder.startswith('D'):
            for f in os.listdir(os.path.join(params_dir, folder)):
                dlvm_pattern = re.compile(r'synthetic_dlvm_params_((?:[a-zA-Z_]+_)*)(D\d+)_N(\d+)\.pt')
                match = dlvm_pattern.match(f)
                if match:
                    method_prefix, dimension_id, n_str = match.groups()
                    model_id = f"{dimension_id}_{args.dlvm_search_method}" if args.dlvm_search_method else dimension_id
                    if args.dlvm_search_method in method_prefix:
                        dlvm_model_ids.add(model_id)
                        model_id_to_folder[model_id] = folder
                        model_n_values.setdefault(model_id, []).append(int(n_str))

    dlvm_model_ids = sorted(list(dlvm_model_ids))
    logger.info(f"Identified DLVM models: {dlvm_model_ids}")

    all_n_values = set()
    imle_files_info = []
    imle_pattern = re.compile(r'synthetic_mle_params_(?:on_(D\d+)_)?N(\d+)\.pt')
    
    # Process IMLE files in each dimension folder
    for folder in all_folders_in_dir:
        if folder.startswith('D'):
            folder_path = os.path.join(params_dir, folder)
            for f in os.listdir(folder_path):
                match = imle_pattern.match(f)
                if match:
                    imle_model_id, n_str = match.groups()
                    n_val = int(n_str)
                    imle_files_info.append({
                        'N': n_val,
                        'model': imle_model_id if imle_model_id else folder.split('_')[0],
                        'path': os.path.join(folder_path, f)
                    })
                    all_n_values.add(n_val)
            
    logger.info(f"Found {len(imle_files_info)} IMLE parameter files.")

    for n_list in model_n_values.values():
        all_n_values.update(n_list)
    all_n_values = sorted(list(all_n_values))

    results_dict = {}
    results_per_metric = {m: {} for m in metrics}

    # Parameter dictionaries for different metrics
    rmse_compute_params = {
        "metrics": metrics, "parameters": parameters, "logger": logger,
        "main_param_only": args.main_param_only, "normalize_errors": args.normalize_errors,
        "rmse_metric_normalizer": normalizer_params, "use_std_dev": args.plot_std_dev
    }

    kld_compute_params = {
        "metrics": metrics, "logger": logger, "normalize_errors": args.normalize_errors,
        "rmse_metric_normalizer": normalizer_params, "use_std_dev": args.plot_std_dev
    }

    # Process IMLE files
    for imle_info in imle_files_info:
        model_key = f"IMLE_{imle_info['model']}" if imle_info['model'] else "IMLE"

        if args.show_single_imle_plot and not "D2" in model_key: # lets use only D2 if we want to show a single IMLE plot
            continue
        
        dimension = get_dimension_from_model_id(model_key)
        if args.show_single_imle_plot:
            model_key = "IMLE"
        
        if not dimension or dimension not in all_ground_truths:
            logger.warning(
                f"Skipping {model_key} (N={imle_info['N']}) as no matching ground truth was found for dimension '{dimension}'.")
            continue

        
        current_gt = all_ground_truths[dimension]
        imle_params = load_estimated_params(imle_info['path'], logger, param_type="IMLE")
        if imle_params:
            if args.metric == "rmse":
                value, error = compute_rmse(imle_params, ground_truth_params=current_gt, **rmse_compute_params)
                per_metric_values, per_metric_errors = compute_rmse_per_metric(imle_params,
                                                                               ground_truth_params=current_gt,
                                                                               **rmse_compute_params)
            else:
                value, error = compute_kld(imle_params, ground_truth_params=current_gt, **kld_compute_params)
                per_metric_values, per_metric_errors = compute_kld_per_metric(imle_params,
                                                                              ground_truth_params=current_gt,
                                                                              **kld_compute_params)

            results_dict.setdefault(model_key, {})[imle_info['N']] = {'Value': value, 'Error': error}
            for metric, val in per_metric_values.items():
                results_per_metric[metric].setdefault(model_key, {})[imle_info['N']] = {'Value': val,
                                                                                        'Error': per_metric_errors.get(
                                                                                            metric)}

    # Process DLVM files
    for model_id in dlvm_model_ids:
        if not "D2" in model_id and args.show_single_dlvm_plot:
            continue
        dimension = get_dimension_from_model_id(model_id)
        if not dimension or dimension not in all_ground_truths:
            logger.warning(
                f"Skipping DLVM model {model_id} as no matching ground truth was found for dimension '{dimension}'.")
            continue

        current_gt = all_ground_truths[dimension]
        logger.info(f"Processing DLVM Model: {model_id} against '{dimension}' ground truth.")
        for N in sorted(list(set(model_n_values.get(model_id, [])))):
            parts = model_id.split('_', 1)
            dlvm_file_name = f"synthetic_dlvm_params_{parts[1]}_{parts[0]}_N{N}.pt" if len(
                parts) > 1 else f"synthetic_dlvm_params_{model_id}_N{N}.pt"
            dlvm_params = load_estimated_params(os.path.join(params_dir, model_id_to_folder[model_id], dlvm_file_name), logger, param_type="DLVM")
            if dlvm_params:
                if args.metric == "rmse":
                    value, error = compute_rmse(dlvm_params, ground_truth_params=current_gt, **rmse_compute_params)
                    per_metric_values, per_metric_errors = compute_rmse_per_metric(dlvm_params,
                                                                                   ground_truth_params=current_gt,
                                                                                   **rmse_compute_params)
                else:  # kld
                    value, error = compute_kld(dlvm_params, ground_truth_params=current_gt, **kld_compute_params)
                    per_metric_values, per_metric_errors = compute_kld_per_metric(dlvm_params,
                                                                                  ground_truth_params=current_gt,
                                                                                  **kld_compute_params)

                results_dict.setdefault(model_id, {})[N] = {'Value': value, 'Error': error}
                for metric, val in per_metric_values.items():
                    results_per_metric[metric].setdefault(model_id, {})[N] = {'Value': val,
                                                                              'Error': per_metric_errors.get(metric)}

    # Prepare plotting data
    plot_data = []
    for model_id, res_by_n in results_dict.items():
        for N, data in res_by_n.items():
            if N>N_MAX_PLOT:
                continue
            plot_data.append(
                {'ModelID': model_id, 'N': N, 'Value': data.get('Value'), 'Error': data.get('Error')})
    df_plot = pd.DataFrame(plot_data)

    if not df_plot.empty:
        metric_name = "RMSE" if args.metric == "rmse" else "KLD"
        error_type = "SD" if args.plot_std_dev else "SE"
        error_metric_suffix = 'stddev' if args.plot_std_dev else 'stderr'
        norm_suffix = 'normalized' if args.normalize_errors else 'unnormalized'
        final_suffix = f"{error_metric_suffix}_{norm_suffix}"
        ylabel = f"{'Normalized ' if args.normalize_errors else ''}{'Root Mean Squared Error (RMSE)' if args.metric == 'rmse' else 'KLD'}"

        # MODERNIZED: Use plot_generic_comparison for combined plots
        plot_combined_comparison(df_plot, plots_dir, f"{metric_name}_combined_{dataset_name}", final_suffix, 
                               ylabel, combined_plot_title, error_type=error_type)

        # Save CSV
        csv_path = os.path.join(plots_dir, f"{args.metric}_results_combined_{dataset_name}_{final_suffix}.csv")
        df_plot.to_csv(csv_path, index=False)
        logger.info(f"{args.metric.upper()} results saved to '{csv_path}'.")

        # MODERNIZED: Use plot_generic_comparison for per-metric plots
        file_name = f"{args.metric}_per_metric_{dataset_name}_{final_suffix}"
        plot_per_metric_comparison(metrics, results_per_metric, plots_dir, file_name,
                                 ylabel=ylabel, title_subtitle=per_metric_subtitle, error_type=error_type)
    else:
        logger.warning("No data available for plotting. Ensure that the parameter files are correctly generated.")
    
    logger.info(f"===== MODERNIZED {args.metric.upper()} Plotting Script Completed Successfully =====")


if __name__ == "__main__":
    main()