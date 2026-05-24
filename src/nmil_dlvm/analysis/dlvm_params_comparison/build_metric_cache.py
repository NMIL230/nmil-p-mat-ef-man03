import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import math
import time
import numbers
from typing import Dict, Any, Optional, List
import argparse
import contextlib
import copy
import glob, re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

SCRIPT_DIR = Path(__file__).resolve().parent

# ====================================================================
# Configuration & Setup
# ====================================================================

# --- Project-specific Imports & Constants ---
from nmil_dlvm.paths import ANALYSIS_ARTIFACTS_ROOT, SAVED_MODELS_ROOT, data_dir
from nmil_dlvm.utils.set_seed import set_seed
from nmil_dlvm.utils.data_distribution_utils import (
    DATASET, RANDOM_SEED, COMPUTE_DEVICE, load_trained_model
)
from nmil_dlvm.utils.grid_search_utils import (
    run_grid_search,
    compute_predictions_fom_latent_points,
    extract_model_params_from_predictions
)

from nmil_dlvm.analysis.dlvm_params_comparison.optimization_w_metric_loss import run_optimization_search

set_seed(RANDOM_SEED)

# --- Build paths to data files ---
param_fits_dir = ANALYSIS_ARTIFACTS_ROOT / "dlvm_imle_comparison" / "synthetic_data" / DATASET / "param_fits"
ground_truth_path = data_dir(DATASET) / "all_data-best_mle_params_mpf100.pt"

# --- Logging Setup ---
def setup_app_logger() -> logging.Logger:
    """Sets up a dedicated logger for this application."""
    logger = logging.getLogger('metric_builder_app')
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    # This is crucial: it prevents messages from being passed to the root logger's handlers,
    # isolating our application's logging from any logging configured by libraries.
    logger.propagate = False

    fmt  = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    sh   = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger

app_logger = setup_app_logger()

# All metrics we care about and how many parameters each has
METRICS = {
    "Countermanding_reaction_time": 2,
    "D2_hit_accuracy":              1,
    "PasatPlus_correctly_answered": 1,
    "RunningSpan_correct_w_len_2":  1,
    "RunningSpan_correct_w_len_3":  1,
    "Stroop_reaction_time":         2,
    "CorsiComplex":                 2,
    "SimpleSpan":                   2,
}

# Only these runs are evaluated and cached
HELDOUT_IDS = [
    "406run9_sim1", "307run0_sim1", "307run5_sim1", "301run2_sim1",
    "405run1_sim1", "411run4_sim1", "404run5_sim1", "305run3_sim1",
    "411run2_sim1", "404run8_sim1",
]


BAD_LIMIT = 10

@contextlib.contextmanager
def stopwatch():
    """
    Context manager to measure elapsed time.
    Usage:
        with stopwatch() as elapsed:
            # code block
        print(f"Elapsed time: {elapsed()} seconds")
    """
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start

def load_simulated_data(file_path: str) -> dict:
    """Load data from a .pt file."""
    logger_instance = logging.getLogger('metric_builder_app')
    if not os.path.exists(file_path):
        logger_instance.error(f"Data file not found at {file_path}")
        raise FileNotFoundError(f"Data file not found at {file_path}")
    try:
        data = torch.load(file_path, weights_only=False, map_location=COMPUTE_DEVICE)
        logger_instance.info(f"Successfully loaded data from '{file_path}' with {len(data)} runs.")
        return data
    except Exception as e:
        logger_instance.error(f"Failed to load data from '{file_path}': {e}")
        raise

def _get_all_N_values(synthetic_data_dir: str) -> List[int]:
    app_logger.info("No --N value provided. Searching for N values in: %s", synthetic_data_dir)
    # Regex to find N value in filenames like 'all_synthetic_data_N5.pt'
    n_pattern = re.compile(r"all_synthetic_data_N(\d+)\.pt")
    found_n_values = set()
    for filename in os.listdir(synthetic_data_dir):
        match = n_pattern.match(filename)
        if match:
            found_n_values.add(int(match.group(1)))
    
    if not found_n_values:
        app_logger.error("Could not find any synthetic data files with N values. Exiting.")
        sys.exit(1)
    
    n_values_to_process = sorted(list(found_n_values))
    app_logger.info("Found the following N values to process: %s", n_values_to_process)
    return n_values_to_process

def _to_np(x) -> np.ndarray:
    """
    Convert tensor / scalar / nested sequence to 1-D float64 np.ndarray.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64).ravel()

    if isinstance(x, numbers.Number):
        return np.asarray([x], dtype=np.float64)

    if isinstance(x, (list, tuple, np.ndarray)):
        # Recursively flatten
        flat = []
        for item in x:
            flat.extend(_to_np(item))
        return np.asarray(flat, dtype=np.float64)

    raise TypeError(f"Cannot convert {type(x)} to NumPy")

def _to_tensor(x, device: str = COMPUTE_DEVICE) -> torch.Tensor:
    """
    Accept torch.Tensor / list / numpy array and return a tensor on *device*.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.as_tensor(x, dtype=torch.float32, device=device)

def _rmse_tensor(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Compute RMSE between two tensors.
    """
    assert pred.shape == gt.shape, "Predictions and ground truth must have the same shape"
    return torch.sqrt(torch.mean((pred - gt) ** 2, dim=-1))

# --------------------------------------------------------------------
# single‑run worker – kept outside the Pool so it can be pickled easily
# --------------------------------------------------------------------
def _process_single_run(
    rid: str,
    synth_d: dict,
    gt_p: dict,
    model: torch.nn.Module,
    max_restarts: list[int],
    num_points_list: list[int]
) -> tuple[str, dict]:

    model_local = copy.deepcopy(model).to(COMPUTE_DEVICE)

    logger = logging.getLogger('metric_builder_app')
    error_cnt = 0
    truth_rid = rid.split("_")[0]

    # Sort lists to ensure predictable key order
    num_points_list.sort()
    max_restarts.sort()

    # ----------------------------------------------------------------
    # 1. grid searches
    # ----------------------------------------------------------------
    grid_results = {}
    for n_points in num_points_list:
        grid_key = f"grid_{n_points}_points"
        grid_loss, grid_best_z = run_grid_search(
            model_local, synth_d[rid], num_points=n_points
        )
        with torch.no_grad():
            grid_f = compute_predictions_fom_latent_points(
                grid_best_z, model, with_activation=True
            )
            grid_params = extract_model_params_from_predictions(
                grid_f, synth_d[rid].keys()
            )
        grid_results[grid_key] = {
            "loss": grid_loss,
            "meu_z": grid_best_z,
            "params": grid_params
        }

    # ----------------------------------------------------------------
    # 2. optimization search(es) – requires grads inside
    # ----------------------------------------------------------------
    optim_results = {}
    pending = set(max_restarts)
    attempt = 0

    while pending:
        attempt += 1
        for restarts in sorted(list(pending)): # iterate in sorted order
            loss, best_z, per_metric_losses = run_optimization_search(
                model_local, synth_d[rid], num_restarts=restarts
            )
            # 1. Create a set of the *aggregated* metric names we expect to see.
            expected_aggregated_metrics = set()
            for metric_name in synth_d[rid].keys():
                if "CorsiComplex" in metric_name:
                    expected_aggregated_metrics.add("CorsiComplex")
                elif "SimpleSpan" in metric_name:
                    expected_aggregated_metrics.add("SimpleSpan")
                else:
                    expected_aggregated_metrics.add(metric_name)

            # 2. Perform the stricter check using the aggregated names.
            is_successful = False
            missing_metrics = set()
            if best_z is not None and per_metric_losses:
                returned_metrics = set(per_metric_losses.keys())
                if expected_aggregated_metrics.issubset(returned_metrics):
                    is_successful = True
                else:
                    missing_metrics = expected_aggregated_metrics - returned_metrics
            if not is_successful:
                logger.warning(
                    "Run %s | optimization attempt %d failed for %d restarts. Missing metrics: %s",
                    rid, attempt, restarts, missing_metrics or "N/A"
                )
                continue
            # inference-only predictions for this successful restart
            with torch.no_grad():
                pf = compute_predictions_fom_latent_points(
                    best_z, model_local, with_activation=True
                )
                pp = extract_model_params_from_predictions(
                    pf, synth_d[rid].keys()
                )

            optim_results[f"n_restart_{restarts}"] = {
                "loss": float(loss),
                "meu_z": best_z.detach(),      # tensor for now
                "metric_losses": per_metric_losses,
                "per_metric": pp,              # tensors
            }
            pending.remove(restarts)           # mark this restart as done

    if not optim_results:
        logger.warning("Run %s had no successful optimization", rid)
        return (None, None)

    # Get sorted keys to ensure final dicts have predictable order
    sorted_optim_keys = sorted(
        optim_results.keys(),
        key=lambda k: int(k.split('_')[-1])
    )

    # ----------------------------------------------------------------
    # 3. metric‑level work (vectorised; tensors stay on device)
    # ----------------------------------------------------------------
    per_metric: dict[str, Any] = {}
    for metric, k in METRICS.items():
        # Check if the metric is present in the first grid search result as a proxy
        if metric not in list(grid_results.values())[0]['params']:
            continue
        try:
            gt_t = _to_tensor(gt_p[truth_rid][metric][:k])
        except (KeyError, IndexError):
            error_cnt += 1
            logger.warning(
                "Run %s | metric %s (%s) has no ground truth params (%d), skipping",
                rid, metric, "ground_truth", k
            )
            if error_cnt >= BAD_LIMIT:
                raise RuntimeError(
                    f"Aborting run {rid}: ≥{BAD_LIMIT} metric errors"
                )
            continue

        grid_params_dict = {}
        grid_RMSEs = {}
        # Sort grid keys numerically by point count for consistent order
        sorted_grid_keys = sorted(
            grid_results.keys(),
            key=lambda k: int(k.split('_')[1])
        )
        for grid_key in sorted_grid_keys:
            grid_data = grid_results[grid_key]
            try:
                grid_t = _to_tensor(grid_data["params"][metric][:k])
                grid_params_dict[grid_key] = grid_t.detach().cpu().tolist()
                grid_RMSEs[grid_key] = _rmse_tensor(grid_t, gt_t).item()
            except (KeyError, IndexError):
                logger.warning(
                    "Run %s | metric %s (%s) has no grid params (%d), skipping",
                    rid, metric, grid_key, k
                )
                error_cnt += 1
                if error_cnt >= BAD_LIMIT:
                    raise RuntimeError(f"Aborting run {rid}: ≥{BAD_LIMIT} metric errors")
                continue

        # build optimization tensors for this metric
        opt_param_t = {}
        mean_losses = {}
        for rk in sorted_optim_keys:
            if rk not in optim_results: continue
            rdat = optim_results[rk]

            if metric in rdat["per_metric"]:
                raw = rdat["per_metric"][metric][:k]      # ← list
                t = _to_tensor(raw)
                if t.numel() == gt_t.numel():
                    opt_param_t[rk] = t
                else:
                    logger.warning(
                        "Run %s | metric %s (%s) has %d params, but ground truth has %d",
                        rid, metric, rk, t.numel(), gt_t.numel()
                    )
                    error_cnt += 1
                    if error_cnt >= BAD_LIMIT:
                        raise RuntimeError(
                            f"Aborting run {rid}: ≥{BAD_LIMIT} metric errors"
                        )
                    continue

            ml = rdat["metric_losses"]
            if metric in ml:
                mean_losses[rk] = float(ml[metric])

        optim_rmses = {
            rk: _rmse_tensor(opt_param_t[rk], gt_t).item()
            for rk in sorted_optim_keys if rk in opt_param_t
        }

        per_metric[metric] = {
            "grid_params_dict": grid_params_dict,
            "grid_RMSEs": grid_RMSEs,
            "gt_params": gt_t.detach().cpu().tolist(),
            "optim_dict": {
                rk: opt_param_t[rk].detach().cpu().tolist()
                for rk in sorted_optim_keys if rk in opt_param_t
            },
            "optim_RMSEs": optim_rmses,
            "optim_metric_losses": {
                rk: mean_losses[rk]
                for rk in sorted_optim_keys if rk in mean_losses
            }
        }

    if not per_metric:
        logger.warning("Run %s has no valid metrics — skipped", rid)
        return (None, None)

    # ----------------------------------------------------------------
    # 4. package run‑level dict (convert tensors just once)
    # ----------------------------------------------------------------

    run_dict = {
        "grid_meu_zs": {
            k: grid_results[k]['meu_z'].detach().cpu().tolist()
            for k in sorted_grid_keys
        },
        "grid_losses": {k: float(grid_results[k]['loss']) for k in sorted_grid_keys},
        "metrics": per_metric,
        "optim_losses": {
            k: optim_results[k]["loss"] for k in sorted_optim_keys
        },
        "optim_meu_zs": {
            k: optim_results[k]["meu_z"].detach().cpu().tolist()
            for k in sorted_optim_keys
        },
    }
    return (rid, run_dict)

# ====================================================================
# Core generation function
# ====================================================================
def build_and_save_metric_dict(
    N: int,
    synthetic_data_dir: str,
    gt_p: dict,
    model: torch.nn.Module,
    out_dir: str,
    max_restarts: list[int],
    num_points_list: list[int],
    max_runs: Optional[int] = None,
    holdouts_only: bool = False
) -> None:
    """
    Process every synthetic run for a single participant-count N, writing a full
    cache to `{out_dir}/all_metrics_N{N}.pt`.

       A *temporary* snapshot is also written every 20 new runs to
           {out_dir}/temporary_saves/N{N}/all_metrics_N{N}_runs_<total>.pt
       The snapshot includes **all** completed runs to that point (both the
       ones loaded from disk *and* the new ones finished in this session).
       On restart we load the snapshot with the largest <total>, merge it with
       any final cache file, and continue with only the unfinished run-IDs.
    """

    logger = logging.getLogger('metric_builder_app')

    # -------------------------------------------------------------------- data
    synth_path = os.path.join(synthetic_data_dir,
                              f"all_synthetic_data_N{N}.pt")
    synth_d = load_simulated_data(synth_path)
    holdout_dir = os.path.join(out_dir, "..", "metric_caches_only_holdouts")
    os.makedirs(holdout_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    if holdouts_only:
        save_path = os.path.join(holdout_dir, f"holdout_metrics_N{N}.pt")

        all_run_ids = [rid for rid in synth_d if rid in HELDOUT_IDS]
        if max_runs is not None:
            all_run_ids = all_run_ids[:max_runs]
    else:
        save_path = os.path.join(out_dir, f"all_metrics_N{N}.pt")

        all_run_ids = (list(synth_d) if max_runs is None
                       else list(synth_d)[:max_runs])
    
    # -------------------------------------------------------------------- load

    # 1  Final cache (if it exists)
    prev_dict: dict[str, dict] = {}
    if os.path.isfile(save_path):
        prev_dict = torch.load(save_path, map_location="cpu")
        logger.info("N=%d | found full cache with %d runs.",
                    N, len(prev_dict))

    # 2  Latest temporary snapshot
    tmp_dir = os.path.join(out_dir, "temporary_saves", f"N{N}")
    tmp_pattern = os.path.join(tmp_dir, f"all_metrics_N{N}_runs_*.pt")
    latest_tmp, latest_cnt = None, -1
    rx = re.compile(rf"all_metrics_N{N}_runs_(\d+)\.pt$")

    for fp in glob.glob(tmp_pattern):
        m = rx.search(os.path.basename(fp))
        if m:
            cnt = int(m.group(1))
            if cnt > latest_cnt:
                latest_tmp, latest_cnt = fp, cnt

    if latest_tmp:
        tmp_dict = torch.load(latest_tmp, map_location="cpu")
        logger.info("N=%d | resuming from temp snapshot (%d runs): %s",
                    N, len(tmp_dict), latest_tmp)
        # newer snapshot values override overlapping keys from prev_dict
        prev_dict = {**prev_dict, **tmp_dict}

    # ------------------------------------------------------------------ todo-set
    done_ids = set(prev_dict.keys())
    todo_ids = [rid for rid in all_run_ids if rid not in done_ids]

    if not todo_ids:
        logger.info("N=%d | all %d runs already cached – nothing to do.",
                    N, len(done_ids))
        
        if not holdouts_only:
            os.makedirs(os.path.join(out_dir, "..", "metric_caches_only_holdouts"), exist_ok=True)
            holdout_dict = {
                rid: data for rid, data in prev_dict.items() if rid in HELDOUT_IDS
            }
            if holdout_dict:
                holdout_save_path = os.path.join(holdout_dir, f"holdout_metrics_N{N}.pt")
                if not os.path.exists(holdout_save_path):
                    try:
                        torch.save(holdout_dict, holdout_save_path)
                        logger.info(
                            "N=%d | Created holdout-only subset with %d runs from existing cache → %s",
                            N, len(holdout_dict), holdout_save_path
                        )
                    except Exception as e:
                        logger.error("Failed to save holdout-only subset from existing cache: %s", e)
        return

    logger.info("N=%d | %d already done, %d remaining.",
                N, len(done_ids), len(todo_ids))

    # ------------------------------------------------------------------ process
    big_dict_new: dict[str, dict] = {}
    new_run_counter = 0  # counts *this* session’s runs

    for idx, rid in enumerate(todo_ids, 1):
        with stopwatch() as elapsed:
            rid_key, rd = _process_single_run(
                rid, synth_d, gt_p, model, max_restarts, num_points_list
            )

        if rid_key is not None:
            big_dict_new[rid_key] = rd
            logger.info("N=%d | run %s (%d/%d) finished in %.2f s",
                        N, rid, idx, len(todo_ids), elapsed())

        new_run_counter += 1

        # periodic FULL snapshot every 20 new runs
        if new_run_counter > 0 and new_run_counter % 20 == 0:
            os.makedirs(tmp_dir, exist_ok=True)
            snapshot = {**prev_dict, **big_dict_new}   # ← COMPLETE dict
            run_total = len(snapshot)                  # how many runs stored
            temp_path = os.path.join(
                tmp_dir, f"all_metrics_N{N}_runs_{run_total}.pt"
            )
            logger.info("Saving temp snapshot (%d runs) → %s",
                        run_total, temp_path)
            torch.save(snapshot, temp_path)

    if not big_dict_new:
        logger.warning("N=%d | no new runs processed – cache unchanged.", N)
        return

    # ------------------------------------------------------------------ persist
    merged = {**prev_dict, **big_dict_new}      # newest results win duplicates
    try:
        tmp = save_path + ".tmp"
        torch.save(merged, tmp)
        os.replace(tmp, save_path)              # atomic replace
        logger.info("N=%d | wrote %d new runs (total %d) → %s",
                    N, len(big_dict_new), len(merged), save_path)
    except Exception as e:
        logger.error("Failed to save metric cache: %s", e)
        fallback = save_path + f".{int(time.time())}.pt"
        torch.save(merged, fallback)
        logger.info("Fallback cache written to %s", fallback)

    # If we processed all runs, also create and save a holdout-only subset
    if not holdouts_only:
        holdout_dict = {
            rid: data for rid, data in merged.items() if rid in HELDOUT_IDS
        }
        if holdout_dict:
            holdout_save_path = os.path.join(holdout_dir, f"holdout_metrics_N{N}.pt")
            try:
                # Use atomic save for the holdout dictionary as well
                tmp_holdout = holdout_save_path + ".tmp"
                torch.save(holdout_dict, tmp_holdout)
                os.replace(tmp_holdout, holdout_save_path)
                logger.info(
                    "N=%d | Saved holdout-only subset with %d runs → %s",
                    N, len(holdout_dict), holdout_save_path
                )
            except Exception as e:
                logger.error("Failed to save holdout-only subset: %s", e)
        else:
            logger.warning("N=%d | No holdout IDs found in the full results, subset not saved.", N)

# ====================================================================
# Main Execution
# ====================================================================
def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate per-run metric dictionaries and save them as .pt"
    )
    p.add_argument("--Ns", type=int, nargs="+", default=None,
                   help="One or more N values (e.g. --Ns 1 2 5) If omitted, "
                        "the script will auto-detect available N values.")
    p.add_argument("--latent_dim", type=int, default=3,
                   help="Latent dimensionality of the DLVM")
    p.add_argument("--model_id", type=str, default="honest-frost-2316",
                   help="Identifier used in the model filename (default: honest-frost-2316)")
    p.add_argument("--max_runs", type=int, default=None,
                   help="Limit runs per N (omit for all)")
    p.add_argument("--n_restarts", type=int, nargs="+", required=True,
                   help="One or more restart values for optimization (e.g. --n_restarts 1 10)")
    p.add_argument("--num_points_list", type=int, nargs="+", default=[200],
                   help="A list of point counts for grid search (default: 200)")
    p.add_argument("--holdouts_only", action="store_true",
                   help="Use heldout IDs for processing (default: False)")
    return p.parse_args()

def main(args: argparse.Namespace) -> None:
    logger = logging.getLogger('metric_builder_app')
    unwanted_handlers = []
    for handler in logger.handlers:
        # We keep the StreamHandler for console output, but remove any others.
        if not isinstance(handler, logging.StreamHandler):
            unwanted_handlers.append(handler)
            
    if unwanted_handlers:
        logger.info("Removing %d unexpected log handler(s) to prevent duplicate logs.", len(unwanted_handlers))
        for handler in unwanted_handlers:
            logger.removeHandler(handler)
            handler.close()

    synthetic_data_dir = os.fspath(
        ANALYSIS_ARTIFACTS_ROOT / "dlvm_imle_comparison" / "synthetic_data" / DATASET
    )

    n_values_to_process = []
    if args.Ns is None:
        n_values_to_process = _get_all_N_values(synthetic_data_dir)
    else:
        n_values_to_process = args.Ns
        app_logger.info("Using provided N values: %s", n_values_to_process)

    # Load model & ground-truth
    model_path = os.fspath(
        SAVED_MODELS_ROOT / DATASET / "heldout_obsmulti" /
        f"variationalNN_relevant_only_latentdim{args.latent_dim}_{args.model_id}.pt"
    )
    model = load_trained_model(args.latent_dim, model_path=model_path)

    gt_path = os.fspath(data_dir(DATASET) / "all_data-best_mle_params_mpf100.pt")
    gt_p = load_simulated_data(gt_path)

    # Output directory
    if args.holdouts_only:
        cache_parent = "metric_caches_only_holdouts"
    else:
        cache_parent = "metric_caches"
    out_root = os.fspath(
        ANALYSIS_ARTIFACTS_ROOT / "dlvm_params_comparison" / args.model_id / cache_parent
    )
    logs_dir = os.path.join(out_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    for N in n_values_to_process:
        app_logger.info("===== Processing N=%d =====", N)

        # ----- per-N log-file handler --------------------------------
        log_path   = os.path.join(logs_dir, f"N{N}.log")

        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"))
        app_logger.addHandler(fh)

        try:
            build_and_save_metric_dict(
            N=N,
            synthetic_data_dir=synthetic_data_dir,
            gt_p=gt_p,
            model=model,
            out_dir=out_root,
            max_restarts=args.n_restarts,
            num_points_list=args.num_points_list,
            max_runs=args.max_runs,
            holdouts_only=args.holdouts_only
            )
        finally:
            # always detach & close to avoid handler pile-up
            app_logger.removeHandler(fh)
            fh.close()

if __name__ == "__main__":
    main(parse_cli())
