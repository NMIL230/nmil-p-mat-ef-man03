#!/usr/bin/env python3
"""
compare_marginal_median.py

This script compares ground truth IMLE parameters against synthetic IMLE parameters for N=2 and N=50
on the same plot, excluding D3 models and using distinct colors for each curve.

The sorting logic has been modified: It now uses the model performance at N=200 to rank and select
the best, middle, and worst trials for plotting. However, the final plots will still display
the curves for N=2 and N=50.

A subtitle is added to display the specific metrics for N=2 and N=50 fits.

Usage:
    python plot_marginal_median.py \
        --ground_truth_pt_file "all_data-best_mle_params_mpf100.pt" \
        --params_dir "param_fits" \
        --data_path "." \
        --metric "kld"
"""

import os
import sys
import argparse
import logging
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm, binom
import matplotlib.colors as mcolors
import tempfile
import shutil
from PyPDF2 import PdfMerger

# Add the parent directory to sys.path to access 'utils' and other modules
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(parent_dir)

# Import both RMSE and KLD computation functions
# Assuming the new file is in the same directory or accessible via path
from analysis.dlvm_imle_comparison.plot_rmse_curves_all_models import compute_rmse_per_metric, compute_kld_per_metric
from visualization.create_marginal_fits import visualize_marginal_fits_many_methods
from utils.set_seed import set_seed
from utils.data_distribution_utils import (
    RANDOM_SEED, RELEVANT_METRICS, CURR_METRICS_DICT, DATASET, SUMMARIZED_METRICS_METRIC_TYPES, SUMMARIZED_METRICS,
    VIS_ORDER_PREFERENCE_METRICS_MAIN_ONLY, SUMMARIZED_METRICS_MAIN_MOMENTS_LABEL_DICT, OUTLIER_HELDOUT_SESSIONS
)


def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )
    logger = logging.getLogger(__name__)
    return logger


def compare_imle_params(models_data, output_dir, logger, file_id, plot_title=None, per_task_metric_results=None,
                        metric_name='RMSE'):
    """
    Generates marginal fit plots comparing ground truth IMLE parameters and synthetic IMLE parameters.
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_session_name = file_id
    try:
        fig = visualize_marginal_fits_many_methods(
            models_data=models_data,
            show_raw_data=False,
            show_curves=True,
            # line_thickness=0.5,
            show_grid=False,
            verbose=False,
            plot_title=plot_title,
            # title_fontsize=10,
            # metric_font_size=6,
            # x_tick_font_size=6,
            # x_y_label_font_size=6,
            per_task_rmse_results=per_task_metric_results,
            metric_name=metric_name,
            # legend_font_size=4,
            # figure_width=12.8,
            # marker_size=1,
            show_legend_per_task=False
        )
        # Use tight_layout before adjusting title to avoid overlap
        plt.tight_layout(rect=[0, 0, 0.98, 1])  # Adjust layout to make space for suptitle if needed [left, bottom, right, top]
        
        # Save both PDF and PNG formats with 300 DPI
        plot_pdf_filename = f"{plot_session_name}.pdf"
        plot_png_filename = f"{plot_session_name}.png"
        plot_pdf_path = os.path.join(output_dir, plot_pdf_filename)
        plot_png_path = os.path.join(output_dir, plot_png_filename)
        
        plt.savefig(plot_pdf_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(plot_png_path, format='png', dpi=300, bbox_inches='tight')
        
        logger.info(f"Saved plot for session '{plot_session_name}' to '{output_dir}' in both PDF and PNG formats.")
        return {'pdf': plot_pdf_path, 'png': plot_png_path}
    except Exception as e:
        logger.error(f"Failed to generate plot for session '{plot_session_name}': {e}", exc_info=True)
        return None
    finally:
        plt.close()


def create_custom_dataframe(synthetic_data, logger):
    """
    Creates a custom DataFrame from synthetic data.
    """
    data = []
    for run_id, metrics_data in synthetic_data.items():
        if not run_id.endswith("_sim1"):
            continue
        for metric_key, results in metrics_data.items():
            if metric_key not in RELEVANT_METRICS:
                continue
            task_label = metric_key.split('_')[0]
            data_type = CURR_METRICS_DICT.get(metric_key, {}).get("type", "Unknown")
            for result in results:
                data.append({
                    "user_session": run_id,
                    "task_label": task_label,
                    "data_type": data_type,
                    "metric": metric_key,
                    "result": result,
                })
    custom_df = pd.DataFrame(data)
    logger.info(f"Constructed custom DataFrame with {len(custom_df)} rows.")
    return custom_df


def load_pt_file(file_path, logger, description=""):
    """
    Loads a .pt file using torch.load.
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


def combine_pdfs_in_folder(folder_path, output_path, logger):
    """Combine all PDF files in a folder into a single PDF, sorted by filename."""
    try:
        merger = PdfMerger()
        
        # Get all PDF files and sort them
        pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
        pdf_files.sort()  # This will sort alphabetically, which should maintain rank order
        
        logger.info(f"Combining {len(pdf_files)} PDF files from {folder_path}")
        
        for filename in pdf_files:
            file_path = os.path.join(folder_path, filename)
            if os.path.exists(file_path):
                with open(file_path, 'rb') as pdf_file:
                    merger.append(pdf_file)
                    logger.debug(f"Added {filename} to combined PDF")
            else:
                logger.warning(f"File {file_path} does not exist and will be skipped.")

        # Write the combined PDF to the output file
        with open(output_path, 'wb') as output_file:
            merger.write(output_file)
        merger.close()
        
        logger.info(f"Combined PDFs saved to: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error combining PDFs: {e}")
        return False


def filter_outlier_sessions(data_dict, session_type, logger, eval_dataset_type, ids_to_discard):
    """Filter out outlier sessions from parameter dictionaries (only for training_set)"""
    if not data_dict or eval_dataset_type != "training_set":
        if eval_dataset_type != "training_set":
            logger.info(f"Using {eval_dataset_type} data - no outlier filtering applied to {session_type}")
        return data_dict
    
    original_count = len(data_dict)
    # Filter based on base session ID (before _sim suffix)
    filtered_dict = {}
    for session_id, params in data_dict.items():
        base_session = session_id.split('_sim')[0] if '_sim' in session_id else session_id
        if base_session not in ids_to_discard:
            filtered_dict[session_id] = params
    
    filtered_count = len(filtered_dict)
    logger.info(f"Filtered out {original_count - filtered_count} outlier sessions from {session_type}, {filtered_count} sessions remaining")
    return filtered_dict


def generate_median_plot(median_session_name, median_base_session, ground_truth_params, 
                          all_data_frames, all_gradient_d2_params, all_imle_params, 
                          all_grid_d2_params, args, logger, Ns_for_plotting, 
                          model_colors, n_to_color_idx, parameters, normalizer_params):
    """
    Generates a single plot for the median session.
    """
    models_data = []
    all_model_per_task_metrics = {}

    for n in Ns_for_plotting:
        session_df_filtered = all_data_frames[n][all_data_frames[n]['user_session'] == median_session_name]
        if session_df_filtered.empty:
            logger.warning(f"No data for median session '{median_session_name}' for N={n}. Skipping this N for this plot.")
            continue

        params_gd_d2 = all_gradient_d2_params[n].get(median_session_name)
        params_imle = all_imle_params[n].get(median_session_name)

        if not all([params_gd_d2, params_imle]):
            logger.warning(
                f"Missing parameters for median session '{median_session_name}' for N={n}. Skipping this N for this plot.")
            continue

        models_to_evaluate = {f"DLVM-2 (n={n})": params_gd_d2, f"IMLE (n={n})": params_imle}
        if args.plot_grid_search and all_grid_d2_params.get(n, {}).get(median_session_name):
            models_to_evaluate[f"DLVM gs D2 (n={n})"] = all_grid_d2_params[n][median_session_name]

        for model_label, model_params in models_to_evaluate.items():
            if args.metric == 'kld':
                metric_per_task, _ = compute_kld_per_metric(
                    estimated_params={median_session_name: model_params},
                    ground_truth_params={median_base_session: ground_truth_params.get(median_base_session)}, metrics=SUMMARIZED_METRICS, logger=logger,
                    normalize_errors=args.normalize_errors, rmse_metric_normalizer=normalizer_params)
            else:
                metric_per_task, _ = compute_rmse_per_metric(
                    estimated_params={median_session_name: model_params},
                    ground_truth_params={median_base_session: ground_truth_params.get(median_base_session)}, metrics=SUMMARIZED_METRICS,
                    parameters=parameters, logger=logger,
                    normalize_errors=args.normalize_errors, rmse_metric_normalizer=normalizer_params)

            all_model_per_task_metrics[model_label] = metric_per_task

        color_idx = n_to_color_idx[n]
        models_data.extend([
            {'raw_data': session_df_filtered, 'params': params_gd_d2, 'label': f"DLVM-2",
             "color": model_colors["DLVM-2"][color_idx], "alpha": 0.7, "same_dataset": True, "use_solid_line": True,
             "line_thickness": 1, "show_raw_data": True, "show_raw_data_annotation": False},
            {'raw_data': session_df_filtered, 'params': params_imle, 'label': f"IMLE",
             "color": model_colors["IMLE"][color_idx], "alpha": 0.7, "same_dataset": True, "use_solid_line": False,
             "line_thickness": 1.5, "show_raw_data": True, "show_raw_data_annotation": False}
        ])
        if args.plot_grid_search and all_grid_d2_params.get(n, {}).get(median_session_name):
            models_data.append(
                {'raw_data': session_df_filtered, 'params': all_grid_d2_params[n][median_session_name],
                 'label': f'DLVM gs D2 (N={n})', 'color': model_colors["DLVM gs D2"][color_idx], 'alpha': 0.7,
                 'same_dataset': True, 'use_solid_line': True, 'line_thickness': 1, "show_raw_data": False, "show_raw_data_annotation": False})

    if models_data:
        gt_data_n = max(Ns_for_plotting)
        models_data.append({
            # 'raw_data': all_data_frames[gt_data_n][all_data_frames[gt_data_n]['user_session'] == median_session_name],
            'raw_data': None,
            'params': ground_truth_params.get(median_base_session), 'label': f"Ground Truth", "color": "gray",
              "alpha": 0.7, "same_dataset": True, "use_solid_line": True, 
            "line_thickness": 1
        })

        # Generate subtitle
        subtitle_parts = []
        for n in Ns_for_plotting:
            dlvm_key = f"DLVM-2 (n={n})"
            imle_key = f"IMLE (n={n})"

            dlvm_metrics = all_model_per_task_metrics.get(dlvm_key, {})
            imle_metrics = all_model_per_task_metrics.get(imle_key, {})

            dlvm_avg_metric = np.mean(list(dlvm_metrics.values())) if dlvm_metrics else float('nan')
            imle_avg_metric = np.mean(list(imle_metrics.values())) if imle_metrics else float('nan')

            part = f"n={n} {args.metric.upper()}: DLVM-2={dlvm_avg_metric:.3f}, IMLE={imle_avg_metric:.3f}"
            subtitle_parts.append(part)

        subtitle = " | ".join(subtitle_parts)

        # Create file ID with rank information
        file_id = f"Median_{median_session_name}_N{'_vs_'.join(map(str, Ns_for_plotting))}_{args.metric.upper()}"
        
        # Main title shows the sorting metric (from N=100)
        # plot_title = f"Rank {rank_idx + 1}/{num_sessions} (N={N_for_sorting} {args.metric.upper()}: {metric_val:.4f}) | Ppt:{base_session} | {', '.join(map(str, Ns_for_plotting))} obs/task"
        plot_title = f"Session ID:{median_base_session}"

        # Combine title and subtitle with a newline
        full_plot_title = f"{plot_title}\n{subtitle}"

        # Generate the plot
        plot_paths = compare_imle_params(
            models_data=models_data,
            output_dir=args.output_dir,  # Use the main output directory
            logger=logger,
            file_id=file_id,
            plot_title=full_plot_title,
            per_task_metric_results=all_model_per_task_metrics,
            metric_name=args.metric.upper()
        )
        return plot_paths
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Compare IMLE Synthetic Parameters Against Ground Truth IMLE Parameters for N=2 and N=50, sorted by N=200 performance.")
    parser.add_argument("--ground_truth_pt_file", type=str, required=True,
                        help="Path to the ground truth IMLE parameters .pt file.")
    parser.add_argument("--plot_grid_search", action="store_true", help="Include grid search models in the plots.")
    parser.add_argument("--params_dir", type=str, default="param_fits",
                        help="Directory containing the fitted synthetic parameter files.")
    parser.add_argument("--synthetic_data_dir", type=str, default=".",
                        help="Directory containing the synthetic data.")
    parser.add_argument("--data_path", type=str, default=".",
                        help="Path to the synthetic data.")
    parser.add_argument("--metric", type=str, choices=["rmse", "kld"], default="kld",
                        help="Metric to compute and display in the legend: 'rmse' or 'kld'.")
    parser.add_argument("--normalize_errors", action='store_true', default=True, help="Normalize errors based on metric type")
    parser.add_argument("--eval_dataset_type", type=str, default="validation_simulated", choices=["validation_simulated", "training_set"], help="Type of experiment (validation_simulated or training_set)")
    parser.add_argument("--generate_individual_plots", action='store_true', default=False, help="Generate individual plots for each session")
    args = parser.parse_args()

    IDS_TO_DISCARD = ['303run1', '306run6', '303run5', '307run6', '306run1', '303run0', '306run2', '408run0', '303run3', '303run4', '306run5', '403run7', '306run7']
    if args.eval_dataset_type == "training_set":
        IDS_TO_DISCARD = IDS_TO_DISCARD + OUTLIER_HELDOUT_SESSIONS
    else:
        IDS_TO_DISCARD = []

    # N values for plotting
    Ns_for_plotting = [2, 50]
    # N value for sorting model performance
    N_for_sorting = 200

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

    model_colors = {
        "IMLE": ('#ff7f0e', '#17becf'),
        "DLVM-2": ('#ff7f0e', '#17becf'),
        "DLVM gs D2": ('#ff9896', '#9467bd'),
    }
    n_to_color_idx = {n: i for i, n in enumerate(Ns_for_plotting)}

    log_file = os.path.join(script_dir, 'plot_marginal_median.log')
    logger = setup_logging(log_file)
    logger.info(
        f"===== Starting IMLE Parameters Comparison for N={Ns_for_plotting} (sorted by N={N_for_sorting}) using {args.metric.upper()} =====")

    set_seed(RANDOM_SEED)
    args.ground_truth_pt_path =  args.ground_truth_pt_file

    all_data_frames = {}
    all_gradient_d2_params = {}
    all_imle_params = {}
    all_grid_d2_params = {}

    param_folder_path = args.params_dir
    logger.info(f"Loading synthetic data and parameters from: '{param_folder_path}'")

    # Load data for plotting (N=2, 50)
    for n in Ns_for_plotting:
        data_file = os.path.join(args.synthetic_data_dir, f"all_synthetic_data_N{n}.pt")
        synthetic_data = load_pt_file(file_path=data_file, logger=logger, description=f"Synthetic data N={n}")
        if synthetic_data is None: sys.exit(1)
        all_data_frames[n] = create_custom_dataframe(synthetic_data=synthetic_data, logger=logger)

        all_gradient_d2_params[n] = load_pt_file(
            os.path.join(param_folder_path, f"synthetic_dlvm_params_gradient_descent_D2_N{n}.pt"), logger,
            f"GD D2 Params N={n}")
        all_gradient_d2_params[n] = filter_outlier_sessions(all_gradient_d2_params[n], f"GD D2 Params N={n}", logger, args.eval_dataset_type, ids_to_discard=IDS_TO_DISCARD)
        
        all_imle_params[n] = load_pt_file(os.path.join(param_folder_path, f"synthetic_mle_params_N{n}.pt"), logger,
                                          f"IMLE Params N={n}")
        all_imle_params[n] = filter_outlier_sessions(all_imle_params[n], f"IMLE Params N={n}", logger, args.eval_dataset_type, ids_to_discard=IDS_TO_DISCARD)
        if None in [all_gradient_d2_params[n], all_imle_params[n]]: sys.exit(1)

        if args.plot_grid_search:
            all_grid_d2_params[n] = load_pt_file(
                os.path.join(param_folder_path, f"synthetic_dlvm_params_grid_search_D2_N{n}.pt"), logger,
                f"GS D2 Params N={n}")
            if all_grid_d2_params[n] is None:
                logger.warning(
                    f"Could not load grid search params for N={n}. Disabling grid search plotting for N={n}.")

    # Load data for sorting (N=200)
    logger.info(f"Loading parameters for N={N_for_sorting} to be used for sorting...")
    params_gd_d2_sorting = load_pt_file(
        os.path.join(param_folder_path, f"synthetic_dlvm_params_gradient_descent_D2_N{N_for_sorting}.pt"), logger,
        f"GD D2 Params N={N_for_sorting} (for sorting)")
    params_imle_sorting = load_pt_file(
        os.path.join(param_folder_path, f"synthetic_mle_params_N{N_for_sorting}.pt"), logger,
        f"IMLE Params N={N_for_sorting} (for sorting)")

    if params_gd_d2_sorting is None or params_imle_sorting is None:
        logger.error(f"Cannot proceed without N={N_for_sorting} parameter files for sorting. Exiting.")
        sys.exit(1)

    params_grid_d2_sorting = {}
    if args.plot_grid_search:
        params_grid_d2_sorting = load_pt_file(
            os.path.join(param_folder_path, f"synthetic_dlvm_params_grid_search_D2_N{N_for_sorting}.pt"), logger,
            f"GS D2 Params N={N_for_sorting} (for sorting)")
        if params_grid_d2_sorting is None:
            logger.warning(
                f"Could not load grid search params for N={N_for_sorting}. Grid search models will not be included in sorting metric.")
            params_grid_d2_sorting = {}
    # extract the folder name from the params_dir
    params_dir_name = os.path.basename(args.params_dir)
    output_folder_name = f"{params_dir_name}_output_plots_N_comparison_D2_{args.metric.upper()}_SORTED_BY_N{N_for_sorting}"
    args.output_dir = os.path.join(script_dir, "fitted_parameters", DATASET,"plots","marginals", args.eval_dataset_type, output_folder_name)
    os.makedirs(args.output_dir, exist_ok=True)

    parameters = {metric: [0, 1] if SUMMARIZED_METRICS_METRIC_TYPES.get(metric) in ['binarySpan', 'timing'] else [0] for
                  metric in SUMMARIZED_METRICS}

    logger.info("Loading ground truth IMLE parameters.")
    ground_truth_params = load_pt_file(args.ground_truth_pt_path, logger, "Ground Truth IMLE params")
    if ground_truth_params is None: sys.exit(1)
    
    # Filter out outlier sessions from ground truth parameters (only for training_set)
    if args.eval_dataset_type == "training_set":
        original_gt_count = len(ground_truth_params)
        ground_truth_params = {k: v for k, v in ground_truth_params.items() if k not in IDS_TO_DISCARD}
        filtered_gt_count = len(ground_truth_params)
        logger.info(f"Filtered out {original_gt_count - filtered_gt_count} outlier sessions from ground truth parameters, {filtered_gt_count} sessions remaining")
    else:
        logger.info(f"Using validation_simulated data - no outlier filtering applied to ground truth parameters")

    reference_sessions = params_gd_d2_sorting
    logger.info(
        f"Calculating {args.metric.upper()} for all sessions using N={N_for_sorting} fits to determine performance ranks...")
    session_performance = {}
    for synthetic_session, _ in reference_sessions.items():

        if not synthetic_session.endswith('_sim1'):
            continue
        base_session = synthetic_session.split('_sim')[0]
        gt_params = ground_truth_params.get(base_session)
        if gt_params is None:
            continue

        session_metrics = []

        params_gd_d2 = params_gd_d2_sorting.get(synthetic_session)
        params_imle = params_imle_sorting.get(synthetic_session)

        if not all([params_gd_d2, params_imle]):
            continue

        models_to_evaluate = {f"DLVM-2 (N={N_for_sorting})": params_gd_d2, f"IMLE (N={N_for_sorting})": params_imle}
        if args.plot_grid_search and params_grid_d2_sorting.get(synthetic_session):
            models_to_evaluate[f"DLVM gs D2 (N={N_for_sorting})"] = params_grid_d2_sorting[synthetic_session]

        for model_params in models_to_evaluate.values():
            if args.metric == 'kld':
                metric_per_task, _ = compute_kld_per_metric(
                    estimated_params={synthetic_session: model_params},
                    ground_truth_params={base_session: gt_params},
                    metrics=SUMMARIZED_METRICS, logger=logger,
                    normalize_errors=args.normalize_errors, rmse_metric_normalizer=normalizer_params)
            else:
                metric_per_task, _ = compute_rmse_per_metric(
                    estimated_params={synthetic_session: model_params},
                    ground_truth_params={base_session: gt_params},
                    metrics=SUMMARIZED_METRICS, parameters=parameters, logger=logger,
                    normalize_errors=args.normalize_errors, rmse_metric_normalizer=normalizer_params)

            if metric_per_task:
                avg_model_metric = np.mean(list(metric_per_task.values()))
                session_metrics.append(avg_model_metric)

        if session_metrics:
            session_performance[synthetic_session] = np.mean(session_metrics)

    if not session_performance: #
        logger.error("Could not calculate performance for any session. Cannot find median. Exiting.")
        sys.exit(1)

    sorted_sessions = sorted(session_performance.items(), key=lambda item: item[1])
    num_sessions = len(sorted_sessions)
    logger.info(f"Total sessions evaluated and ranked based on N={N_for_sorting} performance: {num_sessions}")

    # Find median session (always generated)
    median_index = num_sessions // 2
    median_session_name, median_metric = sorted_sessions[median_index]
    median_base_session = median_session_name.split('_sim')[0]
    logger.info(f"Median session: {median_session_name} (rank {median_index + 1}/{num_sessions}, {args.metric.upper()}: {median_metric:.4f})")

    # Always generate median plot
    logger.info("Generating median plot...")
    median_plot_paths = generate_median_plot(median_session_name, median_base_session, ground_truth_params, 
                                          all_data_frames, all_gradient_d2_params, all_imle_params, 
                                          all_grid_d2_params, args, logger, Ns_for_plotting, 
                                          model_colors, n_to_color_idx, parameters, normalizer_params)
    
    if median_plot_paths:
        median_pdf_path = os.path.join(args.output_dir, f"median_session_{median_base_session}_{args.metric.upper()}.pdf")
        median_png_path = os.path.join(args.output_dir, f"median_session_{median_base_session}_{args.metric.upper()}.png")
        shutil.copy2(median_plot_paths['pdf'], median_pdf_path)
        shutil.copy2(median_plot_paths['png'], median_png_path)
        logger.info(f"Median plot saved to: {median_pdf_path} and {median_png_path}")
    else:
        logger.warning("Failed to generate median plot")

    if args.generate_individual_plots:
        # Create temporary directory for individual plots
        temp_dir = tempfile.mkdtemp(prefix="marginal_plots_")
        logger.info(f"Created temporary directory for individual plots: {temp_dir}")

        # Generate all individual plots
        individual_plot_paths = []
        
        logger.info(f"Generating plots for all {num_sessions} sessions...")
        
        for rank_idx, (synthetic_session, metric_val) in enumerate(sorted_sessions):
            base_session = synthetic_session.split('_sim')[0]
            gt_params = ground_truth_params.get(base_session)

            if gt_params is None:
                logger.error(f"Ground truth params for base session '{base_session}' not found. Skipping this session.")
                continue

            models_data = []
            all_model_per_task_metrics = {}

            for n in Ns_for_plotting:
                session_df_filtered = all_data_frames[n][all_data_frames[n]['user_session'] == synthetic_session]
                if session_df_filtered.empty:
                    logger.warning(f"No data for session '{synthetic_session}' for N={n}. Skipping this N for this plot.")
                    continue

                params_gd_d2 = all_gradient_d2_params[n].get(synthetic_session)
                params_imle = all_imle_params[n].get(synthetic_session)

                if not all([params_gd_d2, params_imle]):
                    logger.warning(
                        f"Missing parameters for session '{synthetic_session}' for N={n}. Skipping this N for this plot.")
                    continue

                models_to_evaluate = {f"DLVM-2 (n={n})": params_gd_d2, f"IMLE (n={n})": params_imle}
                if args.plot_grid_search and all_grid_d2_params.get(n, {}).get(synthetic_session):
                    models_to_evaluate[f"DLVM gs D2 (n={n})"] = all_grid_d2_params[n][synthetic_session]

                for model_label, model_params in models_to_evaluate.items():
                    if args.metric == 'kld':
                        metric_per_task, _ = compute_kld_per_metric(
                            estimated_params={synthetic_session: model_params},
                            ground_truth_params={base_session: gt_params}, metrics=SUMMARIZED_METRICS, logger=logger,
                            normalize_errors=args.normalize_errors, rmse_metric_normalizer=normalizer_params)
                    else:
                        metric_per_task, _ = compute_rmse_per_metric(
                            estimated_params={synthetic_session: model_params},
                            ground_truth_params={base_session: gt_params}, metrics=SUMMARIZED_METRICS,
                            parameters=parameters, logger=logger,
                            normalize_errors=args.normalize_errors, rmse_metric_normalizer=normalizer_params)

                    all_model_per_task_metrics[model_label] = metric_per_task

                color_idx = n_to_color_idx[n]
                models_data.extend([
                    {'raw_data': None, 'params': params_gd_d2, 'label': f"DLVM-2",
                     "color": model_colors["DLVM-2"][color_idx], "alpha": 0.9, "same_dataset": True, "use_solid_line": True,
                     "line_thickness": 3},
                    {'raw_data': session_df_filtered, 'params': params_imle, 'label': f"IMLE",
                     "color": model_colors["IMLE"][color_idx], "alpha": 0.9, "same_dataset": True, "use_solid_line": False,
                     "line_thickness": 2}
                ])
                if args.plot_grid_search and all_grid_d2_params.get(n, {}).get(synthetic_session):
                    models_data.append(
                        {'raw_data': session_df_filtered, 'params': all_grid_d2_params[n][synthetic_session],
                         'label': f'DLVM gs D2 (N={n})', 'color': model_colors["DLVM gs D2"][color_idx], 'alpha': 0.9,
                         'same_dataset': True, 'use_solid_line': True, 'line_thickness': 2})

            if models_data:
                gt_data_n = max(Ns_for_plotting)
                models_data.append({
                    'raw_data': all_data_frames[gt_data_n][all_data_frames[gt_data_n]['user_session'] == synthetic_session],
                    'params': gt_params, 'label': f"Ground Truth", "color": "gray", "alpha": 0.8, "same_dataset": True, "use_solid_line": True, 
                    "line_thickness": 2
                })

                # Generate subtitle
                subtitle_parts = []
                for n in Ns_for_plotting:
                    dlvm_key = f"DLVM-2 (n={n})"
                    imle_key = f"IMLE (n={n})"

                    dlvm_metrics = all_model_per_task_metrics.get(dlvm_key, {})
                    imle_metrics = all_model_per_task_metrics.get(imle_key, {})

                    dlvm_avg_metric = np.mean(list(dlvm_metrics.values())) if dlvm_metrics else float('nan')
                    imle_avg_metric = np.mean(list(imle_metrics.values())) if imle_metrics else float('nan')

                    part = f"N={n} {args.metric.upper()}: DLVM-2={dlvm_avg_metric:.3f}, IMLE={imle_avg_metric:.3f}"
                    subtitle_parts.append(part)

                subtitle = " | ".join(subtitle_parts)

                # Create file ID with rank information
                file_id = f"Rank{rank_idx + 1:03d}_{base_session}_N{'_vs_'.join(map(str, Ns_for_plotting))}_{args.metric.upper()}"
                
                # Main title shows the sorting metric (from N=100)
                # plot_title = f"Rank {rank_idx + 1}/{num_sessions} (N={N_for_sorting} {args.metric.upper()}: {metric_val:.4f}) | Ppt:{base_session} | {', '.join(map(str, Ns_for_plotting))} obs/task"
                plot_title = f"Session ID:{base_session}"

                # Combine title and subtitle with a newline
                full_plot_title = f"{plot_title}\n{subtitle}"

                # Generate the plot
                plot_paths = compare_imle_params(
                    models_data=models_data,
                    output_dir=temp_dir,
                    logger=logger,
                    file_id=file_id,
                    plot_title=full_plot_title,
                    per_task_metric_results=all_model_per_task_metrics,
                    metric_name=args.metric.upper()
                )
                
                if plot_paths:
                    individual_plot_paths.append(plot_paths)
                    
                    # Check if this is the median session
                    if synthetic_session == median_session_name:
                        median_plot_paths = plot_paths
                        logger.info(f"Generated median plot: {plot_paths}")

        logger.info(f"Generated {len(individual_plot_paths)} individual plots")

        # Create combined PDF with all plots (sorted by rank)
        combined_pdf_path = os.path.join(args.output_dir, f"all_sessions_combined_ranked_{args.metric.upper()}.pdf")
        logger.info(f"Combining all plots into: {combined_pdf_path}")
        
        if combine_pdfs_in_folder(temp_dir, combined_pdf_path, logger):
            logger.info(f"Successfully created combined PDF with all {len(individual_plot_paths)} sessions")
        else:
            logger.error("Failed to create combined PDF")

        # Create separate median-only PDF
        if median_plot_paths and os.path.exists(median_plot_paths['pdf']):
            median_pdf_path = os.path.join(args.output_dir, f"median_session_{median_base_session}_{args.metric.upper()}.pdf")
            median_png_path = os.path.join(args.output_dir, f"median_session_{median_base_session}_{args.metric.upper()}.png")
            shutil.copy2(median_plot_paths['pdf'], median_pdf_path)
            shutil.copy2(median_plot_paths['png'], median_png_path)
            logger.info(f"Created separate median PDF: {median_pdf_path}")
        else:
            logger.warning("Could not create separate median PDF - median plot not found")

        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")

        logger.info(
            f"===== IMLE Parameters Comparison Completed Successfully. Combined PDF saved to '{combined_pdf_path}'. Median PDF saved to '{median_pdf_path if median_plot_paths else 'N/A'}'. =====")
    else:
        logger.info("Individual plot generation disabled - skipping plot generation and PDF creation")


if __name__ == '__main__':
    main()
