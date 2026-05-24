import argparse
import os
import sys
import torch
import matplotlib.pyplot as plt
import pandas as pd
import logging
import numpy as np
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from nmil_dlvm.paths import ANALYSIS_ARTIFACTS_ROOT, ensure_dir
from nmil_dlvm.analysis.dlvm_params_comparison.build_metric_cache import METRICS
from nmil_dlvm.utils.data_distribution_utils import CURR_METRICS_DICT, RELEVANT_METRICS, DATASET

from nmil_dlvm.plotting.create_marginal_fits import (
    visualize_marginal_fits_many_methods,
    combine_pdfs_in_folder
)

DLVM_PARAMS_COMPARISON_ROOT = ANALYSIS_ARTIFACTS_ROOT / "dlvm_params_comparison"
DLVM_IMLE_SYNTHETIC_DATA_ROOT = ANALYSIS_ARTIFACTS_ROOT / "dlvm_imle_comparison" / "synthetic_data" / DATASET
LOGS_DIR = ensure_dir(DLVM_PARAMS_COMPARISON_ROOT / "logs")

def setup_file_logger():
    """
    Sets up a logger that writes to a file named after the script.
    """
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    log_file = LOGS_DIR / f"{script_name}.log"

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Set the lowest level to capture all messages

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)  # Set the level for the file
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

logger = setup_file_logger()

def _get_all_N_values(metric_caches_dir):
    # Regex to find N value in filenames like 'all_synthetic_data_N5.pt'
    n_pattern = re.compile(r"all_synthetic_data_N(\d+)\.pt")
    found_n_values = set()
    for filename in os.listdir(metric_caches_dir):
        match = n_pattern.match(filename)
        if match:
            found_n_values.add(int(match.group(1)))
    
    if not found_n_values:
        sys.exit(1)
    
    n_values_to_process = sorted(list(found_n_values))
    return n_values_to_process

def _extract_n(filename: str) -> int:
    """
    Extract the N value from a filename like '406run9_N100.pdf'.
    Returns the integer N.
    """
    m = re.search(r'_N(\d+)\.pdf$', filename)
    return int(m.group(1)) if m else 999999

def create_combined_marginal_fits_pdf(run_dir, logger):
    """
    Combine all PDFs in <run_dir>/individual_runs/ into one file placed in <run_dir>/.
    """
    try:
        indiv_dir = os.path.join(run_dir, "individual_runs")
        if not os.path.exists(indiv_dir):
            logger.warning(f"{indiv_dir} not found – skipping PDF combination")
            return None

        pdf_files = [f for f in os.listdir(indiv_dir) if f.endswith('.pdf')]
        pdf_files.sort(key=_extract_n)

        if not pdf_files:
            logger.warning(f"No PDFs in {indiv_dir} – skipping PDF combination")
            return None

        combined_pdf_path = os.path.join(run_dir, "combined_marginal_fits_session.pdf")
        combine_pdfs_in_folder(indiv_dir, combined_pdf_path)
        return combined_pdf_path

    except Exception as e:
        logger.error(f"Error combining PDFs in {run_dir}: {e}")
        return None

def detect_N_values(caches_dir):
    """Return all N values that have a cache file."""
    pat = re.compile(r"holdout_metrics_N(\d+)\.pt$")
    Ns = [int(m.group(1)) for f in os.listdir(caches_dir) if (m := pat.match(f))]
    return sorted(set(Ns))

def directory_search(marginal_fits_root, logger):
    """
    For every immediate sub‑folder (a run_id) inside <marginal_fits_root>,
    create a combined PDF.
    """
    combined_paths = []
    if not os.path.exists(marginal_fits_root):
        return []

    for entry in os.scandir(marginal_fits_root):
        if not entry.is_dir():
            continue

        run_dir = entry.path
        out_path = create_combined_marginal_fits_pdf(run_dir, logger)
        if out_path:
            combined_paths.append(out_path)
            logger.info(f"Combined PDF written → {out_path}")

    return combined_paths

def to_list(x):
    """Ensure a Python list (strip np.ndarray → list, leave list/tuple alone)."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    return list(x) if not isinstance(x, list) else x

# -----------------------------------------------------------------------------


def gather_and_plot(model, N_list, output_dir, plot_gt, restarts_to_plot, 
                    grids_to_plot, add_grid_lines):
    """
    Gathers data from cache files and generates marginal fit plots based on user specifications.

    Args:
        model (str): The model name.
        N_list (list): A list of N values to process.
        output_dir (str): The directory to save plots.
        plot_gt (bool): Whether to plot the ground truth.
        restarts_to_plot (list or None): A list of specific restarts to plot, or None to plot all available.
        grids_to_plot (list or None): A list of specific grid searches to plot, or None to plot all available.
    """
    heldout_ids = [
        '406run9_sim1', '307run0_sim1', '307run5_sim1', '301run2_sim1',
        '405run1_sim1', '411run4_sim1', '404run5_sim1', '305run3_sim1',
        '411run2_sim1', '404run8_sim1'
    ]
    
    colors = plt.get_cmap('tab10').colors

    metric_caches_dir = DLVM_PARAMS_COMPARISON_ROOT / model / "metric_caches_only_holdouts"

    if N_list is None:
        N_list = _get_all_N_values(metric_caches_dir)

    for N in N_list:
        synthetic_path = DLVM_IMLE_SYNTHETIC_DATA_ROOT / f"all_synthetic_data_N{N}.pt"
        if not os.path.exists(synthetic_path):
            logger.error(f"Synthetic data file not found: {synthetic_path} – skipping N={N}")
            continue
        synthetic_data = torch.load(synthetic_path, weights_only=False, map_location='cpu')

        metric_cache_path = os.path.join(metric_caches_dir, f'holdout_metrics_N{N}.pt')
        if not os.path.exists(metric_cache_path):
            logger.error(f"Metric cache file not found: {metric_cache_path} – skipping N={N}")
            continue
        metric_cache = torch.load(metric_cache_path, weights_only=False, map_location='cpu')

        for heldout_id, synthetic_dict in synthetic_data.items():
            if heldout_id not in heldout_ids:
                continue
            
            if heldout_id not in metric_cache:
                logger.warning(f"Heldout ID {heldout_id} not found in metric cache for N={N} – skipping.")
                continue
            
            cache_entry = metric_cache[heldout_id]
            models_data = []
            title_parts = [f"ID: {heldout_id}", f"N={N}"]
            color_idx = 0

            # --- Determine which restarts and grids to plot for this specific run ---
            restarts_for_run = []
            if restarts_to_plot is None:  # None signals to find all available
                optim_keys = cache_entry.get('optim_meu_zs', {}).keys()
                if optim_keys:
                    # This correctly extracts and sorts the restart numbers
                    restarts_for_run = sorted([int(k.split('_')[-1]) for k in optim_keys])
                    logger.debug(f"N={N}, found all restarts: {restarts_for_run}")
                else:
                    logger.warning(f"Requested all restarts, but none found in cache for N={N}.")
            elif restarts_to_plot:
                 restarts_for_run = restarts_to_plot

            grids_for_run = []
            if grids_to_plot is None:  # None signals to find all available
                grid_keys = cache_entry.get('grid_meu_zs', {}).keys()
                if grid_keys:
                    # FIX: Sort grid keys numerically based on the number in the string
                    grids_for_run = sorted(list(grid_keys), key=lambda s: int(re.search(r'grid_(\d+)_points', s).group(1)))
                    logger.debug(f"N={N}, found all grids: {grids_for_run}")
                else:
                    logger.warning(f"Requested all grids, but none found in cache for N={N}.")
            elif grids_to_plot:
                 grids_for_run = grids_to_plot

            # --- Prepare raw data for plotting ---
            valid_rows = [{'metric': k, 'result': float(val)} for k, v in synthetic_dict.items() if v for val in v]
            raw_df = pd.DataFrame(valid_rows)

            # --- 1. Ground Truth ---
            if plot_gt:
                gt_p = {}
                gt_missing = False
                for metric_label in METRICS.keys():
                    try:
                        gt_p[metric_label] = to_list(cache_entry['metrics'][metric_label]['gt_params'])
                    except KeyError:
                        logger.warning(f"Ground Truth for metric '{metric_label}' missing in {heldout_id} – skipping GT plot.")
                        gt_missing = True
                        break
                if not gt_missing:
                    models_data.append({
                        'params': gt_p, 'raw_data': raw_df,
                        'label': 'Ground Truth', 'color': 'gray', 'alpha': 0.7
                    })

            # --- 2. Optimization Restarts ---
            if restarts_for_run:
                for r in restarts_for_run:
                    key = f'n_restart_{r}'
                    params, metric_losses = {}, {}
                    try:
                        total_loss = cache_entry['optim_losses'][key]
                        for metric_label in METRICS.keys():
                            params[metric_label] = to_list(cache_entry['metrics'][metric_label]['optim_dict'][key])
                            metric_losses[metric_label] = float(cache_entry['metrics'][metric_label]['optim_metric_losses'][key])
                        models_data.append({
                            'params': params, 'raw_data': raw_df, 'metric_losses': metric_losses,
                            'label': f'Restart={r}', 'color': colors[color_idx % len(colors)], 'alpha': 0.7
                        })
                        title_parts.append(f'R={r}, Loss={total_loss:.4f}')
                        color_idx += 1
                    except KeyError:
                        logger.warning(f"Restart data for '{key}' not found in {heldout_id} for N={N} – skipping")

            # --- 3. Grid Searches ---
            if grids_for_run:
                metrics_dict = cache_entry.get('metrics', {})
                # Check if any metric has grid search data
                has_grid_data = any('grid_params_dict' in v for v in metrics_dict.values())
                if not has_grid_data:
                    logger.warning(f"No 'grid_params_dict' found in metric cache for {heldout_id}, N={N}. Skipping all grid plots.")
                else:
                    for grid_key in grids_for_run:
                        params = {}
                        try:
                            total_loss = cache_entry['grid_losses'][grid_key]
                            for metric_label in METRICS.keys():
                                params[metric_label] = to_list(cache_entry['metrics'][metric_label]['grid_params_dict'][grid_key])
                            
                            grid_label = grid_key.replace('_points', '').replace('grid_', '')
                            models_data.append({
                                'params': params, 'raw_data': raw_df,
                                'label': f'Grid={grid_label}', 'color': colors[color_idx % len(colors)], 'alpha': 0.7
                            })
                            title_parts.append(f'Grid-{grid_label}, Loss={total_loss:.4f}')
                            color_idx += 1
                        except KeyError:
                            logger.warning(f"Grid data for '{grid_key}' not found in {heldout_id} for N={N} – skipping")

            # --- Plot & Save ---
            if not models_data:
                logger.info(f"Nothing to plot for {heldout_id}, N={N} with current settings. Skipping file generation.")
                continue

            title = "; ".join(title_parts)
            fig = visualize_marginal_fits_many_methods(
                models_data=models_data, show_raw_data=True, show_curves=True,
                line_thickness=2, show_grid=add_grid_lines, plot_title=title,
                title_fontsize=14, show_metric_loss=True
            )
            heldout_save_id = heldout_id.split('_')[0]
            run_dir = os.path.join(output_dir, heldout_save_id)
            single_run_dir = os.path.join(run_dir, "individual_runs")
            os.makedirs(single_run_dir, exist_ok=True)

            fname = os.path.join(single_run_dir, f"{heldout_save_id}_N{N:03d}.pdf")
            fig.savefig(fname)
            plt.close(fig)
            logger.info(f"Saved file at: {fname}")

    directory_search(output_dir, logger)

def parse_cli():
    p = argparse.ArgumentParser(
        description="Plot marginal fits for various N values and methods. \n"
                    "If no plot-specific flags are given, defaults to plotting Ground Truth only.\n\n"
                    "Examples:\n"
                    "1. Plot GT only (default): python plot_marginal_fits.py --Ns 10 50\n"
                    "2. Plot everything: python plot_marginal_fits.py --plot_all --Ns 10 50\n"
                    "3. Plot GT and specific restarts: python plot_marginal_fits.py --plot_restarts 1 5 --Ns 10\n"
                    "4. Plot all grids and GT: python plot_marginal_fits.py --plot_grids --Ns 10\n"
                    "5. Do not plot GT, but plot specific grids: python plot_marginal_fits.py --dont_plot_gt --plot_grids 100 300 --Ns 10",
        formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument('--model', type=str, default='honest-frost-2316',
                   help="Model name (e.g., 'honest-frost-2316')")
    p.add_argument('--Ns', type=int, nargs='+', default=None,
                   help="List of N values to process (e.g., --Ns 1 10 100); omit to auto-detect all available N values.")
    p.add_argument('--add_grid_lines', action='store_true',
                   help="Add grid lines to the plots for better readability.")
    
    p.add_argument('--dont_plot_gt', action='store_true',
                   help="Exclude Ground Truth from the plot.")
    p.add_argument('--plot_restarts', nargs='*', type=int,
                   help="Plot optimization restarts. Provide specific restart numbers (e.g., 1 10) or no arguments to plot all available restarts.")
    p.add_argument('--plot_grids', nargs='*', type=str,
                   help="Plot grid searches. Provide specific grid point numbers (e.g., 10 300) or no arguments to plot all available grids.")
    p.add_argument('--plot_all', action='store_true',
                   help="Convenience flag to plot Ground Truth and all available restarts and grids. Overrides other plot flags.")
    
    return p.parse_args()


# ------------- main driver ----------------------------------------------------
def main(args):
    logger.info("Starting marginal fits plotting script with args: %s", args)

    caches_dir = DLVM_PARAMS_COMPARISON_ROOT / args.model / "metric_caches_only_holdouts"

    if args.Ns is None:
        n_values_to_process = detect_N_values(caches_dir)
    else:
        n_values_to_process = args.Ns

    plot_gt_flag = True
    restarts_list = []  # Default to plotting no restarts
    grids_list = []     # Default to plotting no grids

    # Check if any specific plotting arguments were provided by the user
    is_any_plot_arg_given = (
        args.plot_all or
        args.dont_plot_gt or
        args.plot_restarts is not None or
        args.plot_grids is not None
    )

    if not is_any_plot_arg_given:
        # DEFAULT BEHAVIOR: If no flags are set, plot only the ground truth
        logger.info("No specific plotting flags provided. Defaulting to plot Ground Truth only.")
        plot_gt_flag = True
    elif args.plot_all:
        # PLOT ALL: User explicitly asked for everything
        logger.info("Plotting all available data due to --plot_all flag.")
        plot_gt_flag = True
        restarts_list = None  # None signals to find and plot all
        grids_list = None     # None signals to find and plot all
    else:
        # CUSTOM PLOTS: User has specified what they want
        if args.dont_plot_gt:
            plot_gt_flag = False
        
        if args.plot_restarts is not None:
            # If the list is empty (e.g., --plot_restarts), set to None to find all.
            # Otherwise, sort the provided list of numbers.
            restarts_list = sorted(args.plot_restarts) if args.plot_restarts else None
        
        if args.plot_grids is not None:
            # If the list is empty (e.g., --plot_grids), set to None to find all.
            # Otherwise, format the list of numbers into grid keys after sorting.
            if not args.plot_grids:
                grids_list = None # Find all
            else:
                # Convert grid point strings to integers for sorting, then format back to strings.
                sorted_grid_nums = sorted([int(n) for n in args.plot_grids])
                grids_list = [f"grid_{n}_points" for n in sorted_grid_nums]

    output_dir = DLVM_PARAMS_COMPARISON_ROOT / args.model / "plots" / "marginal_fits"
    os.makedirs(output_dir, exist_ok=True)

    gather_and_plot(args.model, n_values_to_process, output_dir, 
                    plot_gt_flag, restarts_list, grids_list,
                    args.add_grid_lines)
    logger.info("Script finished.")

if __name__ == "__main__":
    main(parse_cli())
