import sys
# sys.path.append("../")
import torch
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..")) 
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import logging
import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
from utils.data_distribution_utils import (
    RELEVANT_METRICS,
    DATASET,
    COMPUTE_DEVICE,
    create_metrics_dict,
    mle_params_to_dist,
    prepare_data,
    activation_dict,
    load_trained_model,
    generate_model_name,
    SUMMARIZED_METRICS,
    SUMMARIZED_METRICS_METRIC_TYPES,
    CURR_METRICS_DICT,
    dist_dict,
)
import pandas as pd
from gpytorch.priors import NormalPrior
from collections import Counter
import random
import argparse
import numpy as np
from utils.set_seed import set_seed
from utils.active_learning_utils import (
    calculate_kld_given_metric,
    compute_correct_mle_ratio,
    compute_naive_mle_ratio,
    get_KL_per_cognitive_test,
    get_data_list_from_mle_data_generator,
    get_mle_ratio,
    evaluate_model_fit_performance,
    move_distribution_to_cuda,
    update_latent_dist_from_data,
    compute_total_n_trials_actual_session_data,
    perform_active_learning_update,
)
from utils.grid_search_utils import get_predictions_dicts_from_latent_points
from utils.data_distribution_utils import SUMMARIZED_METRICS_METRIC_TYPES
from visualization.create_marginal_fits import visualize_marginal_fits_many_methods, combine_pdfs_in_folder
import json


# Define simplified span metric groups for test selection
RELEVANT_METRICS_SPAN_CONDENSED = ["CorsiComplex_", "SimpleSpan_"]
RELEVANT_METRICS_COMPLEX_SPAN = [metric for metric in RELEVANT_METRICS if "CorsiComplex" in metric]
RELEVANT_METRICS_SIMPLE_SPAN = [metric for metric in RELEVANT_METRICS if "SimpleSpan" in metric]
import warnings

warnings.filterwarnings("ignore")
relu_torch = torch.nn.ReLU(inplace=False)
EPS = 1e-10


def relu(x):
    """
    Custom ReLU function with epsilon to prevent numerical issues.
    Adds a small epsilon value to avoid exact zeros which can cause gradient problems.
    
    Args:
        x: Input tensor
    Returns:
        ReLU activated tensor with small epsilon added
    """
    return relu_torch(x) + EPS


def setup_run_directory(args, run_name,session_id,base_results_dir):
    """
    Setup organized directory structure for DALE run outputs.
    Creates results/dale_runs/[run_name]/ directory structure.
    
    Args:
        args: Parsed command-line arguments
        run_name: Name of the run
    Returns:
        tuple: (run_dir, logger) - Run directory path and configured logger
    """
    # Create base results directory
    
    # dale_runs_dir = os.path.join(base_results_dir, "dale_runs")
    # Create run directory
    run_dir = os.path.join(base_results_dir, run_name, session_id)
    os.makedirs(run_dir, exist_ok=True)

    
    
    # Create subdirectories for organized outputs
    subdirs = [
        "logs",
        "models", 
        "data",
        "plots",
        "checkpoints",
        "analysis"
    ]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(run_dir, "logs", f"{run_name}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)  # Also log to console
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Log run configuration
    logger.info(f"=== DALE Run Started: {run_name} ===")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Run mode: {args.run_mode}")
    logger.info(f"Dataset: {DATASET}")
    logger.info(f"Test session ID: {args.test_session_id}")
    logger.info(f"Latent dimension: {args.latent_dim}")
    logger.info(f"Test budget: {args.test_budget}")
    logger.info(f"Compute device: {COMPUTE_DEVICE}")
    
    # Save run configuration
    config_file = os.path.join(run_dir, "run_config.json")
    import json
    config_dict = vars(args).copy()
    config_dict["run_name"] = run_name
    config_dict["run_dir"] = run_dir
    config_dict["dataset"] = DATASET
    config_dict["compute_device"] = str(COMPUTE_DEVICE)
    config_dict["timestamp"] = datetime.datetime.now().isoformat()
    
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    
    return run_dir, logger


def safe_to_scalar(value):
    """
    Safely convert a value (potentially a PyTorch tensor) to a Python scalar.
    
    Args:
        value: Value to convert (tensor, scalar, etc.)
    Returns:
        Python scalar (float/int)
    """
    if hasattr(value, 'item'):
        # It's a PyTorch tensor
        return float(value.item())
    elif hasattr(value, 'detach'):
        # It's a tensor with gradients, detach first
        return float(value.detach().item())
    else:
        # It's already a scalar
        return float(value)


def create_marginal_fits_visualization(curr_posterior_dist, model, update_w_data, 
                                     best_mle_params_obs, run_dir, args, test_num, task_ran = None):
    """
    Create and save marginal fits visualization comparing model predictions with ground truth.
    
    Args:
        curr_posterior_dist: Current posterior belief distribution
        model: The neural network model
        update_w_data: Current collected data
        best_mle_params_obs: Ground truth MLE parameters
        run_dir: Directory to save plots
        args: Command line arguments
        test_num: Current test number
    """
    
    try:
        plots_dir = os.path.join(run_dir, "plots", "marginal_fits")
        os.makedirs(plots_dir, exist_ok=True)
        
        # # Get current latent position from posterior
        latent_position = curr_posterior_dist.mean.unsqueeze(0)  # Add batch dimension
        
        model_params = get_predictions_dicts_from_latent_points(latent_position, model, model_type = "NN", with_activation = True)[0]
        
        # Create raw data DataFrame for visualization
        raw_data_rows = []
        for metric, data_list in update_w_data.items():
            for data_point in data_list:
                raw_data_rows.append({
                    'metric': metric,
                    'result': data_point
                })
        
        raw_data_df = pd.DataFrame(raw_data_rows)

        # Prepare models_data for visualization function
        models_data = [
            {
                'params': model_params,
                'raw_data': raw_data_df,
                'label': f'DLVM D{args.latent_dim}',
                "color": "#1f77b4",  # Muted blue
                'alpha': 0.8
            },
            {
                'params': best_mle_params_obs,
                'raw_data': raw_data_df,
                'label': 'Ground Truth MLE',
                "color": "gray",  # Ensuring Ground Truth is the most visible
                "alpha": 0.8,  # Higher transparency for clarity
            }
        ]
        
        # Create the visualization
        plot_title = f"Session {args.test_session_id}: Ran: {task_ran} (n = {test_num} observations)" if task_ran is not None else f"Session {args.test_session_id} - Test {test_num}"
        fig = visualize_marginal_fits_many_methods( 
            models_data=models_data,
            show_raw_data=True,
            show_curves=True,
            line_thickness=2,
            show_grid=True,
            verbose=False,
            plot_title=plot_title,
            title_fontsize=16
        )
        
        # Save the plot
        plot_filename = f"marginal_fits_test_{test_num:03d}_session_{args.test_session_id}.pdf"
        plot_path = os.path.join(plots_dir, plot_filename)
        plt.savefig(plot_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
        
    except Exception as e:
        print(f"Error creating marginal fits visualization: {e}")
        return None


def create_combined_marginal_fits_pdf(run_dir, args, logger):
    """
    Combine all intermediate marginal fit PDFs into a single comprehensive PDF.
    
    Args:
        run_dir: Directory containing the run outputs
        args: Command line arguments
        logger: Logger instance
    """
    try:
        marginal_fits_dir = os.path.join(run_dir, "plots", "marginal_fits")
        
        if not os.path.exists(marginal_fits_dir):
            logger.warning("No marginal fits directory found - skipping combined PDF creation")
            return None
        
        # Look for intermediate PDFs (created by batching process)
        intermediate_files = [f for f in os.listdir(marginal_fits_dir) 
                            if f.startswith('intermediate_marginal_fits_') and f.endswith('.pdf')]
        
        # If no intermediate files, look for individual files (fallback)
        if not intermediate_files:
            individual_files = [f for f in os.listdir(marginal_fits_dir) 
                              if f.startswith('marginal_fits_test_') and f.endswith('.pdf')]
            if individual_files:
                logger.info("Found individual PDFs instead of intermediate batches - combining directly")
                # Create output path for combined PDF
                combined_pdf_path = os.path.join(
                    run_dir, 
                    "plots", 
                    f"combined_marginal_fits_session_{args.test_session_id}_ld{args.latent_dim}.pdf"
                )
                combine_pdfs_in_folder(marginal_fits_dir, combined_pdf_path)
                return combined_pdf_path
            else:
                logger.warning("No marginal fit PDFs found - skipping combined PDF creation")
                return None
        
        # Create output path for final combined PDF
        combined_pdf_path = os.path.join(
            run_dir, 
            "plots", 
            f"combined_marginal_fits_session_{args.test_session_id}_ld{args.latent_dim}.pdf"
        )
        
        # Combine all intermediate PDFs
        from PyPDF2 import PdfMerger
        merger = PdfMerger()
        
        for filename in sorted(intermediate_files):
            file_path = os.path.join(marginal_fits_dir, filename)
            with open(file_path, 'rb') as pdf_file:
                merger.append(pdf_file)
        
        # Write final combined PDF
        with open(combined_pdf_path, 'wb') as output_file:
            merger.write(output_file)
        
        # Optionally delete intermediate files to save more space
        for filename in intermediate_files:
            file_path = os.path.join(marginal_fits_dir, filename)
            os.remove(file_path)
        
        logger.info(f"Created final combined marginal fits PDF: {combined_pdf_path}")
        logger.info(f"Combined {len(intermediate_files)} intermediate PDF batches")
        logger.info(f"Deleted {len(intermediate_files)} intermediate files to save space")
        
        return combined_pdf_path
        
    except Exception as e:
        logger.error(f"Error creating combined marginal fits PDF: {e}")
        return None


def create_performance_plots(performance_data, run_dir, args):
    """
    Create and save plots showing KLD and MLE ratio evolution over the number of tests.
    
    Args:
        performance_data (dict): Dictionary containing tracked performance metrics
        run_dir (str): Directory to save plots
        args: Command line arguments
    """
    try:
        plots_dir = os.path.join(run_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Check if we have data to plot
        if len(performance_data["num_tests_run"]) == 0:
            print(f"Warning: No performance data to plot for session {args.test_session_id}")
            return None, None, None, None
        
        print(f"Creating performance plots for session {args.test_session_id} with {len(performance_data['num_tests_run'])} data points")
        
        # Extract data for plotting
        num_tests = performance_data["num_tests_run"]
        kld_values = performance_data["total_kld"]
        mle_ratio_naive = performance_data["mle_ratio_naive"]
        mle_ratio_correct = performance_data["mle_ratio_correct"]
        
        print(f"KLD range: {min(kld_values):.2f} to {max(kld_values):.2f}")
        print(f"MLE ratio naive range: {min(mle_ratio_naive):.2f} to {max(mle_ratio_naive):.2f}")
        print(f"MLE ratio correct range: {min(mle_ratio_correct):.2f} to {max(mle_ratio_correct):.2f}")
        
        # Set up the plotting style
        plt.style.use('default')
        fig_width, fig_height = 10, 6
        
        # Create KLD plot
        plt.figure(figsize=(fig_width, fig_height))
        plt.plot(num_tests, kld_values, 'b-', linewidth=2, marker='o', markersize=4, alpha=0.7)
        plt.xlabel('Number of Tests Run', fontsize=12)
        plt.ylabel('Kullback-Leibler Divergence (KLD)', fontsize=12)
        plt.title(f'KLD Evolution - Session {args.test_session_id} (LD={args.latent_dim})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)  # Force y-axis limits
        plt.tight_layout()
        
        # Save KLD plot
        kld_plot_path = os.path.join(plots_dir, f"kld_evolution_session_{args.test_session_id}.pdf")
        plt.savefig(kld_plot_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved KLD plot: {kld_plot_path}")
        
        # Create MLE Ratio plot
        plt.figure(figsize=(fig_width, fig_height))
        plt.plot(num_tests, mle_ratio_naive, 'r-', linewidth=2, marker='s', markersize=4, alpha=0.7, label='MLE Ratio (Naive)')
        plt.plot(num_tests, mle_ratio_correct, 'g-', linewidth=2, marker='^', markersize=4, alpha=0.7, label='MLE Ratio (Correct)')
        plt.xlabel('Number of Tests Run', fontsize=12)
        plt.ylabel('MLE Ratio', fontsize=12)
        plt.title(f'MLE Ratio Evolution - Session {args.test_session_id} (LD={args.latent_dim})', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Save MLE Ratio plot
        mle_plot_path = os.path.join(plots_dir, f"mle_ratio_evolution_session_{args.test_session_id}.pdf")
        plt.savefig(mle_plot_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved MLE ratio plot: {mle_plot_path}")
        
        # Create combined plot with dual y-axes
        fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))
        
        # Plot KLD on left y-axis
        color1 = 'tab:blue'
        ax1.set_xlabel('Number of Tests Run', fontsize=12)
        ax1.set_ylabel('KLD', color=color1, fontsize=12)
        line1 = ax1.plot(num_tests, kld_values, color=color1, linewidth=2, marker='o', markersize=4, alpha=0.7, label='KLD')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim(0, 100)  # Force y-axis limits for KLD
        ax1.grid(True, alpha=0.3)
        
        # Create second y-axis for MLE ratio
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        color3 = 'tab:green'
        ax2.set_ylabel('MLE Ratio', color=color2, fontsize=12)
        line2 = ax2.plot(num_tests, mle_ratio_naive, color=color2, linewidth=2, marker='s', markersize=4, alpha=0.7, label='MLE Ratio (Naive)')
        line3 = ax2.plot(num_tests, mle_ratio_correct, color=color3, linewidth=2, marker='^', markersize=4, alpha=0.7, label='MLE Ratio (Correct)')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Add legend
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right', fontsize=10)
        
        plt.title(f'Performance Evolution - Session {args.test_session_id} (LD={args.latent_dim})', fontsize=14)
        plt.tight_layout()
        
        # Save combined plot
        combined_plot_path = os.path.join(plots_dir, f"performance_evolution_session_{args.test_session_id}.pdf")
        plt.savefig(combined_plot_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved combined plot: {combined_plot_path}")
        
        # Create additional plots for log probabilities and distance from prior
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: KLD evolution
        ax1.plot(num_tests, kld_values, 'b-', linewidth=2, marker='o', markersize=4, alpha=0.7)
        ax1.set_xlabel('Number of Tests', fontsize=10)
        ax1.set_ylabel('KLD', fontsize=10)
        ax1.set_title('KLD Evolution', fontsize=12)
        ax1.set_ylim(0, 100)  # Force y-axis limits
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: MLE Ratios
        ax2.plot(num_tests, mle_ratio_naive, 'r-', linewidth=2, marker='s', markersize=4, alpha=0.7, label='Naive')
        ax2.plot(num_tests, mle_ratio_correct, 'g-', linewidth=2, marker='^', markersize=4, alpha=0.7, label='Correct')
        ax2.set_xlabel('Number of Tests', fontsize=10)
        ax2.set_ylabel('MLE Ratio', fontsize=10)
        ax2.set_title('MLE Ratio Evolution', fontsize=12)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Log Probabilities
        model_log_prob = performance_data["model_log_prob_data"]
        mle_log_prob = performance_data["mle_log_prob_data"]
        ax3.plot(num_tests, model_log_prob, 'purple', linewidth=2, marker='d', markersize=4, alpha=0.7, label='Model')
        ax3.plot(num_tests, mle_log_prob, 'orange', linewidth=2, marker='v', markersize=4, alpha=0.7, label='MLE')
        ax3.set_xlabel('Number of Tests', fontsize=10)
        ax3.set_ylabel('Log Probability', fontsize=10)
        ax3.set_title('Log Probability Evolution', fontsize=12)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: MLE Ratio comparison (both naive and correct)
        ax4.plot(num_tests, mle_ratio_naive, 'r-', linewidth=2, marker='s', markersize=4, alpha=0.7, label='MLE Ratio (Naive)')
        ax4.plot(num_tests, mle_ratio_correct, 'g-', linewidth=2, marker='^', markersize=4, alpha=0.7, label='MLE Ratio (Correct)')
        ax4.set_xlabel('Number of Tests', fontsize=10)
        ax4.set_ylabel('MLE Ratio', fontsize=10)
        ax4.set_title('MLE Ratio Comparison', fontsize=12)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Comprehensive Performance Analysis - Session {args.test_session_id} (LD={args.latent_dim})', fontsize=14)
        plt.tight_layout()
        
        # Save comprehensive plot
        comprehensive_plot_path = os.path.join(plots_dir, f"comprehensive_analysis_session_{args.test_session_id}.pdf")
        plt.savefig(comprehensive_plot_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved comprehensive plot: {comprehensive_plot_path}")
        
        print(f"Successfully created all performance plots for session {args.test_session_id}")
        return kld_plot_path, mle_plot_path, combined_plot_path, comprehensive_plot_path
    
    except Exception as e:
        print(f"Error creating performance plots for session {args.test_session_id}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def create_aggregate_plots(all_session_data, run_dir, args):
    """
    Create aggregate plots combining data from multiple sessions.
    Shows individual session curves and aggregate statistics.
    
    Args:
        all_session_data (dict): Dictionary with session_id as keys and performance_tracking as values
        run_dir (str): Run directory where aggregate analysis should be placed
        args: Command line arguments
    """
    try:
        # Create aggregate plots directory within the run directory
        aggregate_dir = os.path.join(run_dir, "aggregate_analysis")
        os.makedirs(aggregate_dir, exist_ok=True)
        
        if len(all_session_data) == 0:
            print("No session data available for analysis")
            return None
        
        print(f"Creating aggregate plots for {len(all_session_data)} sessions: {list(all_session_data.keys())}")
        
        # Determine maximum number of tests across all sessions
        max_tests = max(len(data["num_tests_run"]) for data in all_session_data.values())
        print(f"Maximum number of tests across sessions: {max_tests}")
        
        # Prepare data for aggregation
        session_ids = list(all_session_data.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(session_ids)))
        
        # Define common test points for interpolation (used throughout the function)
        common_tests = np.arange(0, max_tests)
        
        # === Individual KLD curves with aggregate ===
        plt.figure(figsize=(12, 8))
        
        # Plot individual session curves
        all_kld_data = []
        for i, (session_id, data) in enumerate(all_session_data.items()):
            num_tests = data["num_tests_run"]
            kld_values = data["total_kld"]
            plt.plot(num_tests, kld_values, color=colors[i], alpha=0.6, linewidth=1.5, 
                    marker='o', markersize=3, label=f'Session {session_id}')
            all_kld_data.append((num_tests, kld_values))
        
        # Initialize interpolated arrays
        interpolated_kld = []
        mean_kld = np.array([])
        std_kld = np.array([])
        
        # Calculate and plot aggregate statistics
        if len(all_kld_data) > 1:
            # Interpolate all curves to common test points for aggregation
            for num_tests, kld_values in all_kld_data:
                if len(num_tests) > 1:
                    interp_kld = np.interp(common_tests, num_tests, kld_values)
                    interpolated_kld.append(interp_kld)
            
            if interpolated_kld:
                interpolated_kld = np.array(interpolated_kld)
                mean_kld = np.mean(interpolated_kld, axis=0)
                std_kld = np.std(interpolated_kld, axis=0)
                
                # Plot mean with confidence interval
                plt.plot(common_tests, mean_kld, 'k-', linewidth=3, alpha=0.9, label='Mean across sessions')
                plt.fill_between(common_tests, mean_kld - std_kld, mean_kld + std_kld, 
                               color='black', alpha=0.2, label='±1 std')
        
        plt.xlabel('Number of Tests Run', fontsize=12)
        plt.ylabel('Kullback-Leibler Divergence (KLD)', fontsize=12)
        
        # Dynamic title based on number of sessions
        if len(session_ids) == 1:
            plt.title(f'KLD Evolution - Session {session_ids[0]} (LD={args.latent_dim})', fontsize=14)
        else:
            plt.title(f'KLD Evolution - Aggregate Analysis ({len(session_ids)} Sessions, LD={args.latent_dim})', fontsize=14)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        kld_aggregate_path = os.path.join(aggregate_dir, f"kld_aggregate_ld{args.latent_dim}.pdf")
        plt.savefig(kld_aggregate_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved KLD aggregate plot: {kld_aggregate_path}")
        
        # === Individual MLE Ratio curves with aggregate ===
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # MLE Ratio Naive
        all_mle_naive_data = []
        for i, (session_id, data) in enumerate(all_session_data.items()):
            num_tests = data["num_tests_run"]
            mle_naive = data["mle_ratio_naive"]
            ax1.plot(num_tests, mle_naive, color=colors[i], alpha=0.6, linewidth=1.5,
                    marker='s', markersize=3, label=f'Session {session_id}')
            all_mle_naive_data.append((num_tests, mle_naive))
        
        # Initialize MLE naive variables
        interpolated_mle_naive = []
        mean_mle_naive = np.array([])
        std_mle_naive = np.array([])
        
        # Calculate aggregate for naive
        if len(all_mle_naive_data) > 1:
            for num_tests, mle_values in all_mle_naive_data:
                if len(num_tests) > 1:
                    interp_mle = np.interp(common_tests, num_tests, mle_values)
                    interpolated_mle_naive.append(interp_mle)
            
            if interpolated_mle_naive:
                interpolated_mle_naive = np.array(interpolated_mle_naive)
                mean_mle_naive = np.mean(interpolated_mle_naive, axis=0)
                std_mle_naive = np.std(interpolated_mle_naive, axis=0)
                
                ax1.plot(common_tests, mean_mle_naive, 'k-', linewidth=3, alpha=0.9, label='Mean')
                ax1.fill_between(common_tests, mean_mle_naive - std_mle_naive, 
                               mean_mle_naive + std_mle_naive, color='black', alpha=0.2, label='±1 std')
        
        ax1.set_xlabel('Number of Tests Run', fontsize=12)
        ax1.set_ylabel('MLE Ratio (Naive)', fontsize=12)
        ax1.set_title('MLE Ratio (Naive) - Aggregate', fontsize=12)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # MLE Ratio Correct
        all_mle_correct_data = []
        for i, (session_id, data) in enumerate(all_session_data.items()):
            num_tests = data["num_tests_run"]
            mle_correct = data["mle_ratio_correct"]
            ax2.plot(num_tests, mle_correct, color=colors[i], alpha=0.6, linewidth=1.5,
                    marker='^', markersize=3, label=f'Session {session_id}')
            all_mle_correct_data.append((num_tests, mle_correct))
        
        # Initialize MLE correct variables
        interpolated_mle_correct = []
        mean_mle_correct = np.array([])
        std_mle_correct = np.array([])
        
        # Calculate aggregate for correct
        if len(all_mle_correct_data) > 1:
            for num_tests, mle_values in all_mle_correct_data:
                if len(num_tests) > 1:
                    interp_mle = np.interp(common_tests, num_tests, mle_values)
                    interpolated_mle_correct.append(interp_mle)
            
            if interpolated_mle_correct:
                interpolated_mle_correct = np.array(interpolated_mle_correct)
                mean_mle_correct = np.mean(interpolated_mle_correct, axis=0)
                std_mle_correct = np.std(interpolated_mle_correct, axis=0)
                
                ax2.plot(common_tests, mean_mle_correct, 'k-', linewidth=3, alpha=0.9, label='Mean')
                ax2.fill_between(common_tests, mean_mle_correct - std_mle_correct, 
                               mean_mle_correct + std_mle_correct, color='black', alpha=0.2, label='±1 std')
        
        ax2.set_xlabel('Number of Tests Run', fontsize=12)
        ax2.set_ylabel('MLE Ratio (Correct)', fontsize=12)
        ax2.set_title('MLE Ratio (Correct) - Aggregate', fontsize=12)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Dynamic suptitle based on number of sessions
        if len(session_ids) == 1:
            plt.suptitle(f'MLE Ratio Evolution - Session {session_ids[0]} (LD={args.latent_dim})', fontsize=14)
        else:
            plt.suptitle(f'MLE Ratio Evolution - Aggregate Analysis ({len(session_ids)} Sessions, LD={args.latent_dim})', fontsize=14)
        
        plt.tight_layout()
        
        mle_aggregate_path = os.path.join(aggregate_dir, f"mle_ratio_aggregate_ld{args.latent_dim}.pdf")
        plt.savefig(mle_aggregate_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved MLE ratio aggregate plot: {mle_aggregate_path}")
        
        # === Comprehensive aggregate analysis ===
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: KLD with all individual curves and mean
        for i, (session_id, data) in enumerate(all_session_data.items()):
            ax1.plot(data["num_tests_run"], data["total_kld"], color=colors[i], alpha=0.5, 
                    linewidth=1, marker='o', markersize=2)
        if len(all_kld_data) > 1 and len(interpolated_kld) > 0 and mean_kld.size > 0:
            ax1.plot(common_tests, mean_kld, 'k-', linewidth=2, label='Mean')
            ax1.fill_between(common_tests, mean_kld - std_kld, mean_kld + std_kld, 
                           color='black', alpha=0.2)
        ax1.set_xlabel('Number of Tests', fontsize=10)
        ax1.set_ylabel('KLD', fontsize=10)
        ax1.set_title('KLD Evolution (All Sessions)', fontsize=11)
        ax1.set_ylim(0, 100)  # Force y-axis limits for KLD
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: MLE Ratio Naive with all curves
        for i, (session_id, data) in enumerate(all_session_data.items()):
            ax2.plot(data["num_tests_run"], data["mle_ratio_naive"], color=colors[i], alpha=0.5, 
                    linewidth=1, marker='s', markersize=2)
        if len(all_mle_naive_data) > 1 and len(interpolated_mle_naive) > 0 and mean_mle_naive.size > 0:
            ax2.plot(common_tests, mean_mle_naive, 'k-', linewidth=2, label='Mean')
            ax2.fill_between(common_tests, mean_mle_naive - std_mle_naive, 
                           mean_mle_naive + std_mle_naive, color='black', alpha=0.2)
        ax2.set_xlabel('Number of Tests', fontsize=10)
        ax2.set_ylabel('MLE Ratio (Naive)', fontsize=10)
        ax2.set_title('MLE Ratio Naive (All Sessions)', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: MLE Ratio Correct with all curves
        for i, (session_id, data) in enumerate(all_session_data.items()):
            ax3.plot(data["num_tests_run"], data["mle_ratio_correct"], color=colors[i], alpha=0.5, 
                    linewidth=1, marker='^', markersize=2)
        if len(all_mle_correct_data) > 1 and len(interpolated_mle_correct) > 0 and mean_mle_correct.size > 0:
            ax3.plot(common_tests, mean_mle_correct, 'k-', linewidth=2, label='Mean')
            ax3.fill_between(common_tests, mean_mle_correct - std_mle_correct, 
                           mean_mle_correct + std_mle_correct, color='black', alpha=0.2)
        ax3.set_xlabel('Number of Tests', fontsize=10)
        ax3.set_ylabel('MLE Ratio (Correct)', fontsize=10)
        ax3.set_title('MLE Ratio Correct (All Sessions)', fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Distance from Prior - Replace with MLE ratio comparison
        for i, (session_id, data) in enumerate(all_session_data.items()):
            ax4.plot(data["num_tests_run"], data["mle_ratio_naive"], color=colors[i], alpha=0.5, 
                    linewidth=1, marker='s', markersize=2, label=f'Session {session_id} (Naive)')
            ax4.plot(data["num_tests_run"], data["mle_ratio_correct"], color=colors[i], alpha=0.3, 
                    linewidth=1, marker='^', markersize=2, linestyle='--')
        ax4.set_xlabel('Number of Tests', fontsize=10)
        ax4.set_ylabel('MLE Ratio', fontsize=10)
        ax4.set_title('MLE Ratio Comparison (All Sessions)', fontsize=11)
        ax4.grid(True, alpha=0.3)
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Dynamic suptitle based on number of sessions
        if len(session_ids) == 1:
            plt.suptitle(f'Comprehensive Performance Analysis - Session {session_ids[0]} (LD={args.latent_dim})', fontsize=14)
        else:
            plt.suptitle(f'Comprehensive Aggregate Analysis - {len(session_ids)} Sessions (LD={args.latent_dim})', fontsize=14)
        
        plt.tight_layout()
        
        comprehensive_aggregate_path = os.path.join(aggregate_dir, f"comprehensive_aggregate_ld{args.latent_dim}.pdf")
        plt.savefig(comprehensive_aggregate_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved comprehensive aggregate plot: {comprehensive_aggregate_path}")
        
        # Save aggregate data as CSV
        aggregate_data = {
            'session_ids': session_ids,
            'num_sessions': len(session_ids),
            'latent_dim': args.latent_dim
        }
        
        # Add summary statistics if available
        if len(all_kld_data) > 1 and len(interpolated_kld) > 0 and mean_kld.size > 0:
            aggregate_data.update({
                'final_kld_mean': float(mean_kld[-1]),
                'final_kld_std': float(std_kld[-1]),
                'final_mle_naive_mean': float(mean_mle_naive[-1]) if len(interpolated_mle_naive) > 0 and mean_mle_naive.size > 0 else None,
                'final_mle_naive_std': float(std_mle_naive[-1]) if len(interpolated_mle_naive) > 0 and std_mle_naive.size > 0 else None,
                'final_mle_correct_mean': float(mean_mle_correct[-1]) if len(interpolated_mle_correct) > 0 and mean_mle_correct.size > 0 else None,
                'final_mle_correct_std': float(std_mle_correct[-1]) if len(interpolated_mle_correct) > 0 and std_mle_correct.size > 0 else None,
            })
        
        # Save detailed aggregate data
        detailed_aggregate_df = pd.DataFrame({
            'test_number': common_tests,
            'kld_mean': mean_kld if len(all_kld_data) > 1 and len(interpolated_kld) > 0 and mean_kld.size > 0 else [None] * len(common_tests),
            'kld_std': std_kld if len(all_kld_data) > 1 and len(interpolated_kld) > 0 and std_kld.size > 0 else [None] * len(common_tests),
            'mle_naive_mean': mean_mle_naive if len(all_mle_naive_data) > 1 and len(interpolated_mle_naive) > 0 and mean_mle_naive.size > 0 else [None] * len(common_tests),
            'mle_naive_std': std_mle_naive if len(all_mle_naive_data) > 1 and len(interpolated_mle_naive) > 0 and std_mle_naive.size > 0 else [None] * len(common_tests),
            'mle_correct_mean': mean_mle_correct if len(all_mle_correct_data) > 1 and len(interpolated_mle_correct) > 0 and mean_mle_correct.size > 0 else [None] * len(common_tests),
            'mle_correct_std': std_mle_correct if len(all_mle_correct_data) > 1 and len(interpolated_mle_correct) > 0 and std_mle_correct.size > 0 else [None] * len(common_tests)
        })
        
        aggregate_csv_path = os.path.join(aggregate_dir, f"aggregate_performance_ld{args.latent_dim}.csv")
        detailed_aggregate_df.to_csv(aggregate_csv_path, index=False)
        print(f"Saved aggregate performance CSV: {aggregate_csv_path}")
        
        return kld_aggregate_path, mle_aggregate_path, comprehensive_aggregate_path, aggregate_csv_path
        
    except Exception as e:
        print(f"Error creating aggregate plots: {e}")
        return None, None, None, None



def generate_test_data(test_to_run_next, best_mle_params_obs, args, remaining_data=None):
    """
    Generate or select data for the next test.
    
    Args:
        test_to_run_next: The selected test to run
        best_mle_params_obs: Ground truth MLE parameters
        args: Command line arguments
        remaining_data: Optional dictionary of remaining actual session data
        
    Returns:
        tuple: (new_data, updated_remaining_data)
    """
    if args.use_mle_generator_for_data:
        new_data = get_data_list_from_mle_data_generator(
            test_to_run_next, 
            best_mle_params_obs, 
            args.test_batch_size
        )
        return new_data, remaining_data
    elif args.use_actual_session_data and remaining_data is not None:
        if len(remaining_data[test_to_run_next]) == 0:
            return [], remaining_data
        new_data = [remaining_data[test_to_run_next][0]]
        remaining_data[test_to_run_next] = remaining_data[test_to_run_next][1:]
        return new_data, remaining_data
    elif args.use_synthetic_data and remaining_data is not None:  # MarkLu
        if test_to_run_next not in remaining_data or len(remaining_data[test_to_run_next]) == 0:
            return [], remaining_data
        new_data = [remaining_data[test_to_run_next][0]]
        remaining_data[test_to_run_next] = remaining_data[test_to_run_next][1:]
        return new_data, remaining_data
    else:
        raise RuntimeError("must either use mle generator or actual data...")


def update_performance_tracking(performance_tracking, performance, num_tests_run):
    """
    Update performance tracking dictionary with new performance metrics.
    
    Args:
        performance_tracking: Dictionary containing performance tracking data
        performance: Dictionary containing new performance metrics
        num_tests_run: Current number of tests run
    """
    for metric in performance:
        if metric == "meu_z" or metric == "sigma_z": # this is a list of lists, so we need to append the list
            performance_tracking[metric].append(performance[metric])
        else:
            performance_tracking[metric].append(safe_to_scalar(performance[metric]))
    performance_tracking["num_tests_run"].append(num_tests_run)


def log_performance(logger, performance, num_tests_run):
    """
    Log performance metrics to logger.
    
    Args:
        logger: Configured logger instance
        performance: Dictionary containing performance metrics
        num_tests_run: Current number of tests run
    """
    for metric in performance:
        if metric == "meu_z" or metric == "sigma_z":
            logger.info(f"{metric}: {performance[metric]}")
        else:
            logger.info(f"{metric}: {performance[metric]:.4f}")


def initialize_model_and_prior(args, logger):
    """
    Initialize the model and prior distribution.
    
    Args:
        args: Command line arguments
        logger: Configured logger instance
        
    Returns:
        tuple: (model, prior_x, meu_z, sigma_z)
    """
    # Load the pre-trained neural network model
    logger.info(f"Loading pre-trained model with latent dimension: {args.latent_dim}")
    if args.model_path is None:
            model = load_trained_model(latent_dim=args.latent_dim)
    else:
            model = load_trained_model(latent_dim=args.latent_dim, model_path=args.model_path)
    logger.info("Successfully loaded pre-trained model")
    
    # Setup prior distribution
    if args.use_population_prior:
        # Use empirically derived population prior from training data
        prior_file = f"../data/{DATASET}/prior_latent_pos_ld{args.latent_dim}.pt"
        logger.info(f"Loading population prior from: {prior_file}")
        popn_prior = torch.load(prior_file)
        prior_x = NormalPrior(popn_prior["meu_z"], popn_prior["sigma_z"])
        meu_z = torch.nn.Parameter(popn_prior["meu_z"].to(COMPUTE_DEVICE))
        sigma_z = torch.nn.Parameter(popn_prior["sigma_z"].to(COMPUTE_DEVICE))
        logger.info("Using population-based prior distribution")
    else:
        # Use standard normal prior (mean=0, std=1)
        prior_x = NormalPrior(torch.zeros(1, args.latent_dim), torch.ones(1, args.latent_dim))
        meu_z = torch.nn.Parameter(torch.zeros(1, args.latent_dim))
        sigma_z = torch.nn.Parameter(torch.ones(1, args.latent_dim))
        logger.info("Using standard normal prior distribution")
    
    return model, prior_x, meu_z, sigma_z


def initialize_performance_tracking():
    """
    Initialize the performance tracking dictionary to match the keys returned by evaluate_model_fit_performance.
    
    Returns:
        dict: Initialized performance tracking dictionary
    """
    return {
        "num_tests_run": [],
        "total_kld": [],
        "mle_ratio_naive": [],
        "mle_ratio_correct": [],
        "model_log_prob_data": [],
        "mle_log_prob_data": [],
        "meu_z": [],
        "sigma_z": []
    }


def initialize_data_storage(metrics):
    """
    Initialize data storage dictionary for all metrics.
    
    Args:
        metrics: List of metrics to track
        
    Returns:
        dict: Initialized data storage dictionary
    """
    return {metric: [] for metric in metrics}

def load_synthetic_data(args, logger): #MarkLu
    """
    Load synthetic data for the specified session.
    
    Args:
        args: Command line arguments
        logger: Logger instance
        
    Returns:
        tuple: (best_mle_params_obs, synthetic_data_dict)
    """
    synthetic_data_file = args.synthetic_data_file
    logger.info(f"Loading synthetic data from: {synthetic_data_file}")
    all_synthetic_data = torch.load(synthetic_data_file)

    key = args.test_session_id                 # e.g. "307run5"
    if key not in all_synthetic_data:
        key = f"{key}_sim{args.sim_index}"
        if key not in all_synthetic_data:
            raise KeyError(
                f"Neither '{args.test_session_id}' nor '{key}' in {all_synthetic_data}. "
                f"Available keys: {list(all_synthetic_data.keys())[:5]}…"
            )
    
    # get current session data
    session_data = all_synthetic_data[key] 
    
    # Convert the string to a dictionary (if necessary)
    if isinstance(session_data, str):
        import ast
        session_data = ast.literal_eval(session_data)
    
    # Use the synthetic data format directly, without converting to a 4-tuple format.
    # This format can be used directly by the sampling logic in generate_test_data()
    synthetic_data_dict = {}
    for metric in RELEVANT_METRICS:
        if metric in session_data:
            synthetic_data_dict[metric] = session_data[metric].copy()
        else:
            synthetic_data_dict[metric] = []
    
    # mle params
    mle_params_file = args.mle_params_file
    logger.info(f"Loading MLE parameters from: {mle_params_file}")
    all_best_mle_params = torch.load(mle_params_file)
    best_mle_params_obs = all_best_mle_params[args.test_session_id]
    logger.info(f"Loaded MLE parameters for session {args.test_session_id}")
    
    logger.info(f"Loaded synthetic data for session {args.test_session_id}")
    logger.info(f"Sample data for debugging: {list(synthetic_data_dict.keys())[:3]}")
    
    return best_mle_params_obs, synthetic_data_dict

def load_oracle_data(args, logger):
    """
    Load or generate oracle data for performance comparison.
    
    Args:
        args: Command line arguments
        logger: Configured logger instance
        
    Returns:
        tuple: (best_mle_params_obs, data_dict)
    """

    if args.use_synthetic_data:
        return load_synthetic_data(args, logger)

    # Load ground truth MLE parameters
    mle_params_file = args.mle_params_file

    logger.info(f"Loading MLE parameters from: {mle_params_file}")
    all_best_mle_params = torch.load(mle_params_file)
    best_mle_params_obs = all_best_mle_params[args.test_session_id]
    logger.info(f"Loaded MLE parameters for session {args.test_session_id}")
    
    # Prepare held-out test data
    logger.info("Preparing held-out test data...")
    data_dict, all_metrics, _ = prepare_data(heldout_obs_ids=[args.test_session_id], get_heldout_instead=True)
    logger.info(f"Prepared data for {len(all_metrics)} metrics")
    
    return best_mle_params_obs, data_dict


def save_results(run_dir, args, performance_tracking, update_w_data):
    """
    Save final results and data.
    
    Args:
        run_dir: Directory to save results
        args: Command line arguments
        performance_tracking: Dictionary containing performance metrics
        update_w_data: Dictionary containing collected data
    """
    # Save performance tracking data
    performance_df = pd.DataFrame(performance_tracking)
    performance_csv_path = os.path.join(run_dir, "analysis", f"performance_tracking_session_{args.test_session_id}.csv")
    performance_df.to_csv(performance_csv_path, index=False)
    
    # Save collected data
    data_dir = os.path.join(run_dir, "data")
    update_data_file = os.path.join(data_dir, f"final_update_w_data_session_{args.test_session_id}_ld{args.latent_dim}.pt")
    torch.save(update_w_data, update_data_file)

def save_intermediate_data(run_dir, args, num_tests_run, update_w_data):
    """
    Save final results and data.
    
    Args:
        run_dir: Directory to save results
        args: Command line arguments
        update_w_data: Dictionary containing collected data
    """
   
    
    # Save collected data
    data_dir = os.path.join(run_dir, "data")
    update_data_file = os.path.join(data_dir, f"num_tests_run_{num_tests_run}_update_w_data_session_{args.test_session_id}_ld{args.latent_dim}.pt")
    torch.save(update_w_data, update_data_file)

def create_final_visualizations(run_dir, args, performance_tracking):
    """
    Create final visualizations of the run.
    
    Args:
        run_dir: Directory to save visualizations
        args: Command line arguments
        performance_tracking: Dictionary containing performance metrics
    """
    # Get logger from the run directory
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("Creating individual session performance plots...")
    
    # Create performance plots
    kld_plot_path, mle_plot_path, combined_plot_path, comprehensive_plot_path = create_performance_plots(
        performance_tracking, run_dir, args
    )
    
    # Log the created plot paths
    if kld_plot_path:
        logger.info(f"Created KLD evolution plot: {kld_plot_path}")
    if mle_plot_path:
        logger.info(f"Created MLE ratio evolution plot: {mle_plot_path}")
    if combined_plot_path:
        logger.info(f"Created combined performance plot: {combined_plot_path}")
    if comprehensive_plot_path:
        logger.info(f"Created comprehensive analysis plot: {comprehensive_plot_path}")
    
    if not any([kld_plot_path, mle_plot_path, combined_plot_path, comprehensive_plot_path]):
        logger.warning("No individual session performance plots were created!")
    
    # Combine marginal fits PDFs if they were created
    if args.create_marginal_fits_visualizations:
        logger.info("Creating combined marginal fits PDF...")
        combined_marginal_fits_path = create_combined_marginal_fits_pdf(run_dir, args, logger)
        if combined_marginal_fits_path:
            logger.info(f"Created combined marginal fits PDF: {combined_marginal_fits_path}")
        else:
            logger.warning("Failed to create combined marginal fits PDF")

# Global variable to store primer sequence
PRIMER_SEQUENCE = []

def generate_primer_sequence(args):
    """
    Generate primer sequence for task selection.
    
    Args:
        args: Command line arguments containing primer_sequence_task_repetitions
        
    Returns:
        list: 4x8 list of tasks (4 repetitions × 8 tasks from SUMMARIZED_METRICS)
    """
    global PRIMER_SEQUENCE
    
    sequence = []
    repetitions = args.primer_sequence_task_repetitions
    
    # For each repetition
    for _ in range(repetitions):
        # Sample each task from SUMMARIZED_METRICS
        for task in SUMMARIZED_METRICS:
            if "CorsiComplex" in task:
                # Further sample from RELEVANT_METRICS_COMPLEX_SPAN
                selected_task = random.choice(RELEVANT_METRICS_COMPLEX_SPAN)
            elif "SimpleSpan" in task:
                # Further sample from RELEVANT_METRICS_SIMPLE_SPAN  
                selected_task = random.choice(RELEVANT_METRICS_SIMPLE_SPAN)
            else:
                # Use the task as is
                selected_task = task
            
            sequence.append(selected_task)
    
    PRIMER_SEQUENCE = sequence
    return sequence

def compute_probs_data_and_KLD_for_syn_data(
    curr_posterior_belief_latent_dist,
    model, 
    synthetic_data_dict, 
    all_metrics, 
    best_mle_params_obs, 
    n_samples=500, 
    metrics_dict=None,
):
    """
    Calculate probabilities and KLD for synthetic data format.
    
    Args:
        curr_posterior_belief_latent_dist: Current posterior belief distribution
        model: The DLVM model
        synthetic_data_dict: Dictionary with format {'metric': [data_list]}
        all_metrics: List of metrics to evaluate
        best_mle_params_obs: Ground truth MLE parameters
        n_samples: Number of samples for evaluation
        metrics_dict: Metric configuration dictionary
        
    Returns:
        tuple: (total_log_prob_data, mle_log_prob, total_kld)
    """
    # Sample from the latent distribution
    latent_points = curr_posterior_belief_latent_dist.rsample((n_samples,)) 
    # Pass the latent samples through the model to get predictions
    f = model(latent_points.to(COMPUTE_DEVICE)) 
    
    total_n_samples = n_samples 
    total_log_prob_data = 0 
    mle_log_prob = 0.0 
    total_kld = 0.0
    
    # Iterate over each metric
    for metric in all_metrics: 
        # Skip if no data for this metric
        if metric not in synthetic_data_dict or len(synthetic_data_dict[metric]) == 0:
            continue
            
        # Get the type of distribution to use for this metric
        metric_type = metrics_dict[metric]['type'] 
        
        # Get the indices of the features in the model's output that correspond to this metric
        fidxs = metrics_dict[metric]['f_idxs']
        
        # Get the synthetic data for this metric
        data_list = synthetic_data_dict[metric]
        
        # Convert data to tensor format
        if metric_type == "binary" or metric_type == "binarySpan":
            # For binary metrics, we need the sum and count
            data_sum = sum(data_list)
            counts = len(data_list)
            data = torch.tensor([data_sum], dtype=torch.float).to(COMPUTE_DEVICE)
            counts_tensor = torch.tensor([counts], dtype=torch.float).to(COMPUTE_DEVICE)
        else:
            # For timing/continuous metrics, use individual values
            data = torch.tensor(data_list, dtype=torch.float).to(COMPUTE_DEVICE)
            counts_tensor = torch.tensor([len(data_list)], dtype=torch.float).to(COMPUTE_DEVICE)
        
        # Calculate the parameters of the distribution for this metric using the model's predictions
        dist_params = activation_dict[metric_type](
            f[:, :, fidxs], 
            counts_tensor, 
            metrics_dict[metric]['length']
        )
        
        # Create a PyTorch distribution object for this metric
        dist = dist_dict[metric_type](*dist_params)  
        
        # Calculate the log probability of the data for this metric under the model's predictions
        if metric_type == "binary" or metric_type == "binarySpan":
            prob_data = dist.log_prob(data).sum() / total_n_samples
        else:
            prob_data = dist.log_prob(data).sum() / total_n_samples
        
        # Calculate the MLE of the data for this metric
        if "Complex" in metric: 
            mle_dist_params = best_mle_params_obs["CorsiComplex"] 
        elif "Simple" in metric:
            mle_dist_params = best_mle_params_obs["SimpleSpan"]
        else:
            mle_dist_params = best_mle_params_obs[metric] 
        
        # Create MLE distribution
        if metric_type == "binary" or metric_type == "binarySpan":
            mle_dist = mle_params_to_dist(
                metric, 
                mle_dist_params, 
                metric_type, 
                counts=counts_tensor.item()
            )
            mle_prob = mle_dist.log_prob(data).sum()
        else:
            mle_dist = mle_params_to_dist(
                metric, 
                mle_dist_params, 
                metric_type, 
                counts=1  # For continuous data
            )
            mle_prob = mle_dist.log_prob(data).sum()
        
        # Set the MLE of the data to the maximum of the MLE and the probability of the data under the model's predictions
        mle_prob = max(mle_prob, prob_data)
        
        # Accumulate the probability of the data under the model's predictions and the MLE of the data
        total_log_prob_data += prob_data.item() 
        mle_log_prob += mle_prob.item() 
        
        # Compute the KL Divergence for the metric
        mle_dist = move_distribution_to_cuda(mle_dist)
        total_kld += calculate_kld_given_metric(mle_dist, dist, metric) 
    
    return total_log_prob_data, mle_log_prob, total_kld

def evaluate_model_fit_performance_for_syn_data(
    curr_posterior_belief_latent_dist,
    model,
    synthetic_data_dict,
    all_metrics,
    best_mle_params_obs,
    n_samples=500,
    metrics_dict=None,
    mle_ratio_type="naive",
):
    """
    Evaluate model fit performance using synthetic data format.
    
    Args:
        curr_posterior_belief_latent_dist: Current posterior belief distribution
        model: The DLVM model
        synthetic_data_dict: Dictionary with format {'metric': [data_list]}
        all_metrics: List of metrics to evaluate
        best_mle_params_obs: Ground truth MLE parameters
        n_samples: Number of samples for evaluation
        metrics_dict: Metric configuration dictionary
        mle_ratio_type: Type of MLE ratio calculation
        
    Returns:
        dict: Performance metrics
    """
    total_log_prob_data, mle_log_prob_data, total_kld = compute_probs_data_and_KLD_for_syn_data(
        curr_posterior_belief_latent_dist,
        model,
        synthetic_data_dict,
        all_metrics,
        best_mle_params_obs,
        n_samples=n_samples,
        metrics_dict=metrics_dict,
    )
    
    performance = {}
    performance["mle_ratio_naive"] = compute_naive_mle_ratio(total_log_prob_data, mle_log_prob_data)
    performance["mle_ratio_correct"] = compute_correct_mle_ratio(total_log_prob_data, mle_log_prob_data)
    performance["model_log_prob_data"] = total_log_prob_data
    performance["mle_log_prob_data"] = mle_log_prob_data
    performance["total_kld"] = total_kld
    performance["meu_z"] = curr_posterior_belief_latent_dist.mean.tolist()
    performance["sigma_z"] = curr_posterior_belief_latent_dist.stddev.tolist()
    
    return performance


def run_active_learning(args, run_dir, logger):
    """
    Run the active learning process.
    
    Args:
        args: Command line arguments
        run_dir: Directory to save results
        logger: Logger instance
    """
    # Initialize model and prior
    model, prior_x, meu_z, sigma_z = initialize_model_and_prior(args, logger)
    
    # Initialize performance tracking
    performance_tracking = initialize_performance_tracking()
    
    # Initialize data storage
    update_w_data = initialize_data_storage(RELEVANT_METRICS)
    
    # Load or generate oracle data
    best_mle_params_obs, test_data = load_oracle_data(args, logger)
    
    # Initialize posterior belief distribution
    curr_posterior_beleif_latent_dist = prior_x

    test_to_run_next = None
    # Main active learning loop
    for iteration in range(args.test_budget):
        logger.info(f"\nStarting iteration {iteration}/{args.test_budget} with test {test_to_run_next}")
        
        # # Perform active learning update
        # curr_posterior_beleif_latent_dist, meu_z, sigma_z, lowest_loss, test_to_run_next = perform_active_learning_update(
        #     curr_posterior_beleif_latent_dist, model, update_w_data, args, CURR_METRICS_DICT
        # )
        curr_posterior_beleif_latent_dist, meu_z, sigma_z, lowest_loss, test_to_run_next = perform_active_learning_update(
                curr_posterior_beleif_latent_dist, model, update_w_data, args, CURR_METRICS_DICT, num_restarts=args.num_restarts)
  
        # Check if primer sequence is available and not exhausted
        global PRIMER_SEQUENCE
        if args.enable_primer_sequence and len(PRIMER_SEQUENCE) > 0:
            # Use primer sequence instead of active learning
            test_to_run_next = PRIMER_SEQUENCE.pop(0)
            logger.info(f"Using primer sequence task: {test_to_run_next}, {len(PRIMER_SEQUENCE)} tasks remaining")


        if args.random_baseline: # select random test from the span metrics
            test_to_run_next = random.choice(SUMMARIZED_METRICS)
            if "CorsiComplex" in test_to_run_next:
                test_to_run_next = random.choice(RELEVANT_METRICS_COMPLEX_SPAN)
            elif "SimpleSpan" in test_to_run_next:
                test_to_run_next = random.choice(RELEVANT_METRICS_SIMPLE_SPAN)
        
        # Generate new test data
        new_data, test_data = generate_test_data(
            test_to_run_next, best_mle_params_obs, args, test_data
        )
        
        # Update data storage with new data
        update_w_data[test_to_run_next].extend(new_data)
        
        # Evaluate performance
        if not args.use_synthetic_data:
            performance = evaluate_model_fit_performance(
                curr_posterior_beleif_latent_dist,
                model,
                test_data,
                RELEVANT_METRICS,
                best_mle_params_obs,
                metrics_dict=CURR_METRICS_DICT,
                mle_ratio_type=args.mle_ratio_type
            )
        else:
            performance = evaluate_model_fit_performance_for_syn_data(
                curr_posterior_beleif_latent_dist,
                model,
                test_data,
                RELEVANT_METRICS,
                best_mle_params_obs,
                metrics_dict=CURR_METRICS_DICT,
                mle_ratio_type=args.mle_ratio_type
            )
        # Update performance tracking
        update_performance_tracking(performance_tracking, performance, iteration + 1)
        
        # Log performance
        log_performance(logger, performance, iteration + 1)
        
        # Create visualizations if enabled
        if args.create_marginal_fits_visualizations and (iteration + 1) % args.visualization_frequency == 0:
            create_marginal_fits_visualization(
                curr_posterior_beleif_latent_dist, model, update_w_data,
                best_mle_params_obs, run_dir, args, iteration + 1, test_to_run_next
            )

        # save intermediate data
        save_intermediate_data(run_dir, args, iteration, update_w_data)
    
    # Save final results
    save_results(run_dir, args, performance_tracking, update_w_data)
    
    # Create final visualizations
    create_final_visualizations(run_dir, args, performance_tracking)
    
    logger.info("Active learning process completed successfully")
    return performance_tracking


def run(args):
    """
    Main execution function that handles multiple test sessions.
    Supports debug mode for testing and handles errors gracefully.
    
    Args:
        args: Parsed command-line arguments
    """
    
    # =============================================================================
    # DEBUG MODE ADJUSTMENTS: Only modify specific parameters for debug
    # =============================================================================
    
    if args.run_mode == "debug":
        # Set reduced parameters for quick testing
        args.test_budget = 10  # Only difference: limit to 10 tests
        args.eval_test_session_ids = args.eval_test_session_ids[:3]  # Only difference: limit to 3 sessions
        # Debug folder difference is handled in setup_run_directory
        print(f"Running in DEBUG MODE:")
        print(f"  - Test budget limited to: {args.test_budget}")
        print(f"  - Sessions limited to: {args.eval_test_session_ids}")
        print(f"  - Results will be stored in debug folders")

    # =============================================================================
    # MULTI-SESSION EXECUTION: Run DALE on multiple test sessions
    # =============================================================================

    # Store performance data from all sessions for aggregate analysis
    all_session_performance_data = {}
    successful_sessions = []

    # create run_name for the session (prefer provided name)
    run_name = args.run_name.strip() if getattr(args, 'run_name', None) else generate_model_name()
    # Only prefix debug_ when run_name not provided explicitly
    if args.run_mode == "debug" and not getattr(args, 'run_name', None):
        run_name = f"debug_{run_name}"

    # save the run wide config parameters
    base_results_dir = args.base_results_dir
    run_dir = os.path.join(base_results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(args.__dict__, f)
    print(f"Saved run config to: {os.path.join(run_dir, 'run_config.json')}")

    for session_id in args.eval_test_session_ids:
        # Create separate run directory for each session
        args.test_session_id = session_id
        run_dir, logger = setup_run_directory(args, run_name, session_id, base_results_dir)
        
        logger.info(f"Starting DALE with heldout session: {session_id}")
        
        # Generate primer sequence if enabled
        if args.enable_primer_sequence:
            logger.info(f"Generating primer sequence with {args.primer_sequence_task_repetitions} repetitions")
            sequence = generate_primer_sequence(args)
            logger.info(f"Generated primer sequence with {len(sequence)} tasks")
        
        try:
            performance_tracking = run_active_learning(args, run_dir, logger)
            logger.info(f"Successfully completed session: {session_id}")
            
            # Store performance data for aggregate analysis
            if performance_tracking is not None:
                all_session_performance_data[session_id] = performance_tracking
                successful_sessions.append(session_id)
                
                # =============================================================================
                # UPDATE AGGREGATE ANALYSIS: Create/update aggregate plots after each session
                # =============================================================================
                
                print(f"\n=== Updating Aggregate Analysis: {len(successful_sessions)} Session(s) ===")
                print(f"Sessions completed so far: {successful_sessions}")
                
                if len(successful_sessions) == 1:
                    # For single session, put aggregate in the existing run directory
                    aggregate_run_dir = run_dir
                    print(f"Creating analysis summary for session: {successful_sessions[0]}")
                else:
                    # For multiple sessions, create/update a dedicated aggregate run directory
                    original_test_session_id = args.test_session_id
                    original_run_mode = args.run_mode
                    args.run_mode = "run"  # Ensure we don't get debug_run name
                    args.test_session_id = f"aggregate_sessions"
                    aggregate_run_dir, aggregate_logger = setup_run_directory(args, run_name, args.test_session_id, base_results_dir)
                    args.run_mode = original_run_mode  # Restore original mode
                    args.test_session_id = original_test_session_id  # Restore original session ID
                    print(f"Updating aggregate analysis directory: {aggregate_run_dir}")
                
                # Create/update aggregate plots
                aggregate_results = create_aggregate_plots(all_session_performance_data, aggregate_run_dir, args)
                
                if aggregate_results and aggregate_results[0] is not None:
                    kld_agg_path, mle_agg_path, comp_agg_path, csv_agg_path = aggregate_results
                    print(f"Updated aggregate analysis:")
                    print(f"  - KLD aggregate: {kld_agg_path}")
                    print(f"  - MLE ratio aggregate: {mle_agg_path}")
                    print(f"  - Comprehensive aggregate: {comp_agg_path}")
                    print(f"  - Aggregate data CSV: {csv_agg_path}")
                else:
                    print("Failed to create/update aggregate plots")
            
        except Exception as e:
            logger.error(f"Failed to run session {session_id}: {str(e)}")
            logger.exception("Full traceback:")
            continue

    # =============================================================================
    # FINAL SUMMARY: Report on completed sessions
    # =============================================================================
    
    # Final summary
    mode_str = "DEBUG" if args.run_mode == "debug" else "FULL"
    print(f"\n=== {mode_str} RUN COMPLETED ===")
    print(f"Successfully processed {len(successful_sessions)} out of {len(args.eval_test_session_ids)} sessions")
    if args.run_mode == "debug":
        print("Debug run completed! Turn off debug mode for full runs with all sessions and full test budget.")


def main():
    """
    Main function that parses command-line arguments and initiates the DALE experiment.
    Configures all hyperparameters and experimental settings.
    """
    from pathlib import Path
    REPO_ROOT = Path(__file__).resolve().parents[2]

    # =============================================================================
    # ARGUMENT PARSING: Define all experimental configuration parameters
    # =============================================================================
    
    parser = argparse.ArgumentParser() 
    parser.add_argument('--model_path', type=str, default=None)

    # Optional explicit run name to use for results folder
    parser.add_argument('--run_name', type=str, default=None)

    parser.add_argument('--base_results_dir', type=str, default=f'./result')
    parser.add_argument('--use_synthetic_data', type=bool, default=True)
    parser.add_argument('--synthetic_data_file', type=str, default=str(REPO_ROOT / f'analysis/dlvm_imle_comparison/synthetic_data/{DATASET}/all_synthetic_data_N240.pt'))
    # parser.add_argument('--mle_params_file', type=str, default=f'../../data/{DATASET}/all_data-best_mle_params_mpf100.pt')
    parser.add_argument('--mle_params_file', type=str, default=str(REPO_ROOT / 'data/COLL10_SIM/synthetic_ground_truth_parameters.pt'))
    parser.add_argument("--sim_index", type=int, default=1,
                    help=("Which synthetic replicate to use, e.g. "
                          "2  →  key '<session>_sim2'. "
                          "If omitted, will try the raw session key first "
                          "then fallback '_sim1'"))

    # Basic experiment settings
    parser.add_argument('--verbose', type=bool, default=True)  
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--test_session_id", type=str, default="-1")
    
    # Data split configuration
    # parser.add_argument("--trained_model_held_out_ids", nargs='+', type=str, 
    #                    help='List IDs to be held out during training', 
    #                    default=['306run7', '307run0', '307run1', '307run2'])
    parser.add_argument("--eval_test_session_ids", nargs='+', type=str, 
                       help='List IDs for evaluation sessions', 
                       default=['406run9', '307run0', '307run5', '301run2', '405run1', '411run4', '404run5', '305run3', '411run2', '404run8'])
    
    # Session selection parameters
    # parser.add_argument('--min_heldout_session_num', type=int, default=6)  
    # parser.add_argument('--max_heldout_session_num', type=int, default=100)  
    
    # Model architecture
    parser.add_argument('--latent_dim', type=int, default=3)
    
    # Active learning parameters
    parser.add_argument('--test_budget', type=int, default=-1)  # -1 means use COLL10_SIM data count
    parser.add_argument('--test_batch_size', type=int, default=1)
    
    # Optimization parameters
    parser.add_argument('--relearn_latent_dim_from_scratch', type=bool, default=False)
    parser.add_argument('--grad_clip', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--max_epcohs', type=int, default=500)
    parser.add_argument('--max_n_progress_fails', type=int, default=200)
    parser.add_argument('--min_allowed_log_prob', type=int, default=-3000)
    
    # Information-theoretic parameters
    parser.add_argument('--T', type=int, default=100)  # Number of samples for KL estimation
    parser.add_argument('--M', type=int, default=100)  # Number of Monte Carlo samples
    
    # Logging and tracking
    
    # Experimental strategies
    parser.add_argument('--random_baseline', type=bool, default=False)
    parser.add_argument('--use_fixed_dist_of_tasks', type=bool, default=False)
    parser.add_argument('--use_mle_generator_for_data', type=bool, default=True)
    parser.add_argument('--use_actual_session_data', type=bool, default=False)
    parser.add_argument('--oracle_is_raw_data', type=bool, default=True)
    
    # Update strategies
    parser.add_argument('--run_once', type=bool, default=False)
    parser.add_argument('--run_batch_update', type=bool, default=False)
    
    # Prior configuration
    parser.add_argument('--use_population_prior', type=bool, default=False)
    
    # Test battery configuration
    parser.add_argument('--init_w_fixed_test_battery', type=bool, default=False)
    parser.add_argument('--init_w_fixed_battery_taskwise_update', type=bool, default=False)
    
    # Analysis and debugging
    parser.add_argument('--run_mode', type=str, default="run", choices=["debug", "run"])
    parser.add_argument('--get_df_with_all_info', type=bool, default=True)
    parser.add_argument('--extract_df_with_all_info', type=bool, default=False)
    
    # Model saving
    parser.add_argument('--save_model_at', type=int, default=-1)
    
    # Performance evaluation
    parser.add_argument('--mle_ratio_type', default="naive")
    
    # Visualization parameters
    parser.add_argument('--create_marginal_fits_visualizations', type=bool, default=True)
    parser.add_argument('--visualization_frequency', type=int, default=1)  # Create visualization every N tests

    # Active learning parameters
    parser.add_argument('--num_restarts', type=int, default=1)    
    parser.add_argument('--num_grid_points', type=int, default=500)
    parser.add_argument('--use_grid_search', type=bool, default=False, help="Whether to use grid search to select the most likely latent points given the observed data")

    # Primer sequence parameters
    parser.add_argument('--enable_primer_sequence', type=bool, default=False, help="Enable primer sequence algorithm for task selection")
    parser.add_argument('--primer_sequence_task_repetitions', type=int, default=4, help="Number of repetitions for each task in primer sequence")


    # =============================================================================
    # EXPERIMENT EXECUTION: Parse arguments and run experiment
    # =============================================================================
    
    args = parser.parse_args() 
    
    # Validate MLE ratio type
    assert args.mle_ratio_type in ["naive", "correct"]  # naive is the version used in paper
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Execute the experiment
    run(args)


if __name__ == "__main__":
    main()
