import os, sys, math, logging, re, argparse
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ====================================================================
#  Paths & logging
# ====================================================================
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from nmil_dlvm.paths import ANALYSIS_ARTIFACTS_ROOT, ensure_dir

DLVM_PARAMS_COMPARISON_ROOT = ANALYSIS_ARTIFACTS_ROOT / "dlvm_params_comparison"
LOGS_DIR = ensure_dir(DLVM_PARAMS_COMPARISON_ROOT / "logs")

LOG_FILE = LOGS_DIR / "plot_trends.log"

def _setup_logging(log_file: str) -> logging.Logger:
    """Configure logging to output to console **and** file."""
    logger = logging.getLogger(__name__)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = False
    return logger

logger = _setup_logging(LOG_FILE)

# ====================================================================
#  Plot Configurations
# ====================================================================
# Define which keys from the cache files should be collected. This makes it
# easy to add new metrics without changing the collection logic.
PER_METRIC_KEYS = {"optim_RMSEs", "grid_RMSEs", "grid_params_dict", "gt_params", "optim_dict"}

PLOT_CONFIGURATIONS = {
    "gt_rmse": {
        "is_per_metric": True,
        "title_template": "Predicted Parameter v. Ground Truth RMSE\nfor {metric_display_name}",
        "ylabel_template": "Mean RMSE{std_suffix}",
        "output_filename_template": "gt_rmse_by_metric/rmse_v_gt_{metric_name}{std_suffix_file}.png",
        "traces": [
            {
                "type": "restart",
                "label_template": "{restart_display_name} restarts v. GT",
                "data_path_template": ["by_metric", "{metric_name}", "optim_RMSEs", "{key}"],
            },
            {
                "type": "grid",
                "label_template": "{grid_display_name} v. GT",
                "data_path_template": ["by_metric", "{metric_name}", "grid_RMSEs", "{key}"],
            },
        ],
    },
    "aggregated_rmse": {
        "is_per_metric": False,
        "title_template": "Predicted Parameter v. Ground Truth RMSE\nfor All Metrics",
        "ylabel_template": "Mean RMSE{std_suffix}",
        "output_filename_template": "aggregated_rmse_v_gt{std_suffix_file}.png",
        "traces": [
            {
                "type": "restart",
                "label_template": "{restart_display_name} restarts v. GT",
                "data_path_template": ["overall", "all_metrics_optim_gt_rmse", "{key}"],
            },
            {
                "type": "grid",
                "label_template": "{grid_display_name} v. GT",
                "data_path_template": ["overall", "all_metrics_grid_gt_rmse", "{key}"],
            },
        ],
    },
    "loss": {
        "is_per_metric": False,
        "title_template": "Gradient Descent Loss vs. N",
        "ylabel_template": "Mean Loss{std_suffix}",
        "output_filename_template": "optim_loss_trends{std_suffix_file}.png",
        "traces": [
            {
                "type": "restart",
                "label_template": "{restart_display_name} restarts",
                "data_path_template": ["overall", "optim_losses", "{key}"],
            }
        ],
    },
}

# ====================================================================
#  Helper utilities
# ====================================================================

def save_plot(fig, out_png: str) -> None:
    """Save *fig* to *out_png* (creating dirs) and close it."""
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    logger.info("Saved plot → %s", out_png)

def _plot_line(ax, x: List[int], y: np.ndarray, label: str, color: str) -> None:
    """Consistent line style (thicker lines, larger markers)."""
    ax.plot(x, y, "-o", label=label, color=color, linewidth=2, markersize=5)

def _fill_between(ax, x: List[int], mean: np.ndarray, std: Optional[np.ndarray], color: str, show: bool) -> None:
    """Draw ±1 SD band if *show* and *std* finite."""
    if not show or std is None or np.all(np.isnan(std)):
        return
    lower = np.maximum(mean - std, 0)
    ax.fill_between(x, lower, mean + std, alpha=0.2, color=color)

def get_display_name(metric: str) -> str:
    """Generates a cleaner, more readable name for plots."""
    if "grid" in metric:
        points = metric.split("_")[1]
        return f"Grid Search ({points} pts)"
    mapping = {"Countermanding": "Countermanding", 
               "Pasat": "Pasat", 
               "Stroop": "Stroop", 
               "Corsi": "Corsi Complex", 
               "Simple": "Simple Span", 
               "D2": "Cancellation", 
               "len_3": "Running Span Length 3", 
               "len_2": "Running Span Length 2"}
    for key, name in mapping.items():
        if key in metric:
            return name
    return metric

def _get_nested(data: dict, path: List[str]) -> Any:
    """Safely retrieve a value from a nested dict using a list of keys."""
    for key in path:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data

STD_MODE_MAP = {0: ("without_std", False), 1: ("with_std", True), 2: ("both", None)}

def detect_N_values(caches_dir: str) -> List[int]:
    """Return all N values that have a cache file."""
    pat = re.compile(r"all_metrics_N(\d+)\.pt$")
    Ns = [int(m.group(1)) for f in os.listdir(caches_dir) if (m := pat.match(f))]
    return sorted(set(Ns))

# ====================================================================
#  Generalized Data Collection
# ====================================================================


def collect_all_trends(caches_dir: str, N_list: list[int]) -> Dict[str, Any]:
    """Load cached metric dictionaries, compute mean/std, and track missing data."""
    overall_raw = defaultdict(lambda: defaultdict(list))
    metric_raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    bad_counter = Counter()

    for N in N_list:
        cache_path = os.path.join(caches_dir, f"all_metrics_N{N}.pt")
        if not os.path.isfile(cache_path):
            logger.warning("[MISS] cache missing for N=%d", N)
            continue
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if not cache:
            logger.warning("[MISS] empty cache for N=%d", N)
            continue

        for run_id, run in cache.items():
            # --- Collect data that exists directly at the run level ---
            if "grid_losses" in run:
                overall_raw["grid_losses"][N].append(run["grid_losses"])
            if "optim_losses" in run:
                overall_raw["optim_losses"][N].append(run["optim_losses"])

            if "metrics" not in run:
                bad_counter["metrics"] += 1
                continue

            # --- Collect per-metric data AND aggregate where needed ---
            for metric, block in run["metrics"].items():
                # 1. Collect all specified keys for the individual metric
                for key in PER_METRIC_KEYS:
                    if key in block:
                        metric_raw[metric][key][N].append(block[key])
                    else:
                        bad_counter[key] += 1
                
                # 2. Aggregate per-metric RMSEs into the overall dataset
                if "grid_RMSEs" in block:
                    overall_raw["all_metrics_grid_gt_rmse"][N].append(block["grid_RMSEs"])
                if "optim_RMSEs" in block:
                    overall_raw["all_metrics_optim_gt_rmse"][N].append(block["optim_RMSEs"])

    def _reduce_nested(raw_data: defaultdict) -> Dict[str, Any]:
        """Helper to compute mean/std for nested restart/grid key structures."""
        processed_data = defaultdict(lambda: defaultdict(dict))
        for top_key, n_data in raw_data.items():
            for N, values_list in n_data.items():
                # This handles keys like optim_losses, grid_RMSEs, etc.
                if values_list and isinstance(values_list[0], dict):
                    all_sub_keys = {k for d in values_list for k in d}
                    for sub_key in all_sub_keys: # e.g., 'n_restart_1', 'grid_100_points'
                        vals = []
                        for d in values_list:
                            val = d.get(sub_key)
                            if isinstance(val, (int, float)) and math.isfinite(val):
                                vals.append(val)
                        if N not in processed_data[top_key][sub_key]:
                            processed_data[top_key][sub_key][N] = {}
                        processed_data[top_key][sub_key][N]['mean'] = np.mean(vals) if vals else np.nan
                        processed_data[top_key][sub_key][N]['std'] = np.std(vals) if len(vals) > 1 else np.nan
        
        # Restructure for plotting
        final_structure = defaultdict(dict)
        for top_key, sub_key_data in processed_data.items():
            for sub_key, n_data in sub_key_data.items():
                sorted_n = sorted(n_data.keys())
                final_structure[top_key][sub_key] = {
                    "mean": [n_data[n]['mean'] for n in sorted_n],
                    "std": [n_data[n]['std'] for n in sorted_n]
                }
        return final_structure

    # --- Reduce collected raw data to mean/std ---
    overall_processed = _reduce_nested(overall_raw)
    metric_processed = {m: _reduce_nested(d) for m, d in metric_raw.items()}

    logger.info("Data collection complete – BAD_COUNTER total = %d", sum(bad_counter.values()))

    return {"overall": overall_processed, "by_metric": metric_processed, "N": N_list}

# ====================================================================
#  Unified Plotting Function
# ====================================================================

def discover_keys(processed_data: dict) -> (List[str], List[str]):
    """
    Scans the processed data to find all available restart and grid keys.
    This is used when the user doesn't specify them via CLI args.
    """
    restart_keys = set()
    grid_keys = set()

    # Check both overall and per-metric data structures
    data_sources = [processed_data.get("overall", {})]
    data_sources.extend(processed_data.get("by_metric", {}).values())

    for source in data_sources:
        for top_key, data_dict in source.items():
            # Keys are like 'optim_losses', 'grid_RMSEs', etc.
            if "optim" in top_key:
                restart_keys.update(data_dict.keys())
            if "grid" in top_key:
                grid_keys.update(data_dict.keys())

    # --- Custom sort keys to handle numbers in strings ---
    def sort_key_restart(key: str) -> int:
        """Extracts the number from 'n_restart_X' for sorting."""
        match = re.search(r'n_restart_(\d+)', key)
        return int(match.group(1)) if match else -1

    def sort_key_grid(key: str) -> int:
        """Extracts the number from 'grid_X_points' for sorting."""
        match = re.search(r'grid_(\d+)_points', key)
        return int(match.group(1)) if match else -1

    # Filter for the correct pattern and sort numerically
    final_restarts = sorted([k for k in restart_keys if k.startswith('n_restart_')], key=sort_key_restart)
    final_grids = sorted([k for k in grid_keys if k.startswith('grid_')], key=sort_key_grid)
    
    return final_restarts, final_grids


def create_trend_plot(*, processed_data: dict, config: dict,
                      N_list: list, restart_keys: list, grid_keys: list,
                      show_std: bool, out_dir: str, context: Optional[dict] = None) -> None:
    """
    Creates a single trend plot based on a configuration object.
    This function replaces all previous individual plotting functions.
    """
    context = context or {}
    fig, ax = plt.subplots(figsize=(6, 4.5))
    drawn_lines = 0

    # Prepare colors
    colors_opt = cm.viridis(np.linspace(0, 0.85, len(restart_keys))) if restart_keys else []
    colors_grid = cm.autumn(np.linspace(0.2, 0.7, len(grid_keys))) if grid_keys else []


    # Determine if grid search should be plotted
    show_grid_search = context.get("show_grid_search", True)

    for trace_config in config.get("traces", []):
        keys_to_iter = []
        colors = None
        trace_type = trace_config["type"]

        if trace_type == "restart":
            keys_to_iter = restart_keys
            colors = colors_opt
        elif trace_type == "grid" and show_grid_search:
            keys_to_iter = grid_keys
            colors = colors_grid
        else:
            continue

        for i, key in enumerate(keys_to_iter):
            trace_context = context.copy()
            trace_context["key"] = key
            if trace_type == "restart":
                # Extract the number from 'n_restart_10' -> '10'
                trace_context["restart_display_name"] = key.split("_")[-1]
            elif trace_type == "grid":
                trace_context["grid_display_name"] = get_display_name(key)

            # Get data using templated path
            data_path = [p.format(**trace_context) for p in trace_config["data_path_template"]]
            data_series = _get_nested(processed_data, data_path)

            if not data_series or np.all(np.isnan(data_series.get("mean", []))):
                continue
            
            mean = np.asarray(data_series["mean"])
            std = np.asarray(data_series["std"]) if show_std else None
            label = trace_config["label_template"].format(**trace_context)
            
            _plot_line(ax, N_list, mean, label, colors[i])
            _fill_between(ax, N_list, mean, std, colors[i], show_std)
            drawn_lines += 1

    if drawn_lines == 0:
        plt.close(fig)
        title_str = config["title_template"].format(**context) if "metric_name" in context else config["title_template"]
        logger.warning("Skipped plot with no data for config with title: '%s'", title_str)
        return

    # --- Finalize and save plot ---
    std_context = {"std_suffix": " ± 1 SD" if show_std else "", "std_suffix_file": "_std" if show_std else ""}
    context.update(std_context)

    ax.set_xscale("log")
    ax.set_xticks(N_list); ax.set_xticklabels(N_list)
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_xlabel("N (synthetic data points per task)")
    ax.set_ylabel(config["ylabel_template"].format(**context))
    ax.set_title(config["title_template"].format(**context))
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_png = os.path.join(out_dir, config["output_filename_template"].format(**context))
    save_plot(fig, out_png)

# ====================================================================
#  Main Workflow
# ====================================================================

def parse_cli() -> argparse.Namespace:
    """
    Example:
        # Plot specific restarts and grid points
        python plot_trends.py \\
            --model_id honest-frost-2316 \\
            --plot-types gt_rmse aggregated_rmse loss \\
            --restarts 1 4 7 10 \\
            --grid_points 200 \\
            --std 2
            
        # Auto-detect and plot all available restarts and grids
        python plot_trends.py --model_id honest-frost-2316
    """
    p = argparse.ArgumentParser(description="analyze cached metric dictionaries and make trend plots")
    p.add_argument("--model_id", default="honest-frost-2316", help="Folder name that contains metric_caches/")
    p.add_argument("--Ns", type=int, nargs="+", help="Explicit N values; omit to auto-detect in metric_caches/")
    p.add_argument("--restarts", type=int, nargs="+", default=None, help="Optimization restart counts. Omit to auto-detect all.")
    p.add_argument("--grid_points", type=int, nargs="+", default=None, help="Grid search point counts. Omit to auto-detect all.")
    p.add_argument("--std", type=int, choices=STD_MODE_MAP, default=0, help="0: w/o-std | 1: w-std | 2: both")
    p.add_argument("--hide_grid", action="store_true", help="Omit Grid-Search from plots")
    p.add_argument("--plot_types", nargs="+", default=["gt_rmse", "aggregated_rmse", "loss"],
                   choices=PLOT_CONFIGURATIONS.keys(), help="Select which plots to generate.")
    return p.parse_args()

def main() -> None:
    args = parse_cli()
    logger.info("Selected plot types: %s", args.plot_types)
    logger.info("Selected STD mode = %d (%s)", args.std, STD_MODE_MAP[args.std][0])

    caches_dir = DLVM_PARAMS_COMPARISON_ROOT / args.model_id / "metric_caches"
    plots_dir = DLVM_PARAMS_COMPARISON_ROOT / args.model_id / "plots"

    N_list = args.Ns or detect_N_values(caches_dir)
    if not N_list:
        logger.error("No cache files found in %s", caches_dir)
        return
    logger.info("N list: %s", N_list)
    
    # 1. Collect all data once
    processed_data = collect_all_trends(caches_dir, N_list)
    if not processed_data.get("overall") and not processed_data.get("by_metric"):
        logger.error("No data was collected. Aborting.")
        return

    # 2. Determine which restart and grid keys to use
    if args.restarts:
        args.restarts.sort() # Sort user input numerically
        restart_keys = [f"n_restart_{n}" for n in args.restarts]
        logger.info("Using user-specified restart keys: %s", restart_keys)
    else:
        restart_keys, _ = discover_keys(processed_data)
        logger.info("Auto-detected restart keys: %s", restart_keys)

    if args.grid_points:
        args.grid_points.sort() # Sort user input numerically
        grid_keys = [f"grid_{n}_points" for n in args.grid_points]
        logger.info("Using user-specified grid keys: %s", grid_keys)
    else:
        _, grid_keys = discover_keys(processed_data)
        logger.info("Auto-detected grid keys: %s", grid_keys)
        
    # Determine which std variants to generate
    std_settings = [STD_MODE_MAP[args.std][1]] if STD_MODE_MAP[args.std][1] is not None else [False, True]

    # 3. Loop through std settings and plot types to generate plots
    for show_std_flag in std_settings:
        variant_name = "with_std" if show_std_flag else "without_std"
        out_base = os.path.join(plots_dir, variant_name)
        os.makedirs(out_base, exist_ok=True)
        
        logger.info("--- Generating plots for variant: %s ---", variant_name)

        for plot_key in args.plot_types:
            config = PLOT_CONFIGURATIONS[plot_key]
            
            base_context: Dict[str, Any] = {"show_grid_search": not args.hide_grid}

            if config.get("is_per_metric"):
                # Ensure there is data for this metric type before iterating
                if not processed_data.get("by_metric"):
                    continue
                for metric_name in processed_data["by_metric"].keys():
                    metric_context = base_context.copy()
                    metric_context.update({
                        "metric_name": metric_name,
                        "metric_display_name": get_display_name(metric_name),
                    })
                    create_trend_plot(
                        processed_data=processed_data, config=config,
                        N_list=N_list, restart_keys=restart_keys, grid_keys=grid_keys,
                        show_std=show_std_flag, out_dir=out_base, context=metric_context,
                    )
            else: # Aggregated plot
                create_trend_plot(
                    processed_data=processed_data, config=config,
                    N_list=N_list, restart_keys=restart_keys, grid_keys=grid_keys,
                    show_std=show_std_flag, out_dir=out_base, context=base_context,
                )

    logger.info("Plotting complete → %s\n", plots_dir)

if __name__ == "__main__":
    main()
