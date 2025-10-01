#!/usr/bin/env python3  
"""  
Script to compute and save comparison data from DALE performance tracking CSV files.  
This script separates computation from plotting for efficiency.  
Outputs a single CSV containing all six methods' time series data.  
"""  
  
import os  
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import random  
import pandas as pd  
import numpy as np  
import torch  
import argparse  
import ast  
import logging  
import sys  
from pathlib import Path  
from typing import Dict, List  
import json  
import csv
from collections import defaultdict
  
logging.basicConfig(  
    level=logging.DEBUG,  
    format="%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(message)s",  
    stream=sys.stdout,  
    force=True,  
)  
logger = logging.getLogger(__name__)  
logger.setLevel(logging.DEBUG)  
  
# --- make sure the project root is on sys.path ------------------------------  
ROOT = Path(__file__).resolve().parents[2]  
sys.path.insert(0, str(ROOT))  
  
# Import from existing codebase  
from utils.data_distribution_utils import (  
    load_trained_model,  
    SUMMARIZED_METRICS,  
    SUMMARIZED_METRICS_METRIC_TYPES,  
    CURR_METRICS_DICT,  
    DATASET,  
    RELEVANT_METRICS,  
)  
from utils.mle_utils import get_mle_params_per_metric  
from analysis.dlvm_imle_comparison.fit_models_on_sim_data import load_simulated_data  
from utils.grid_search_utils import get_predictions_dicts_from_latent_points, predict_parameters_from_data  
from utils.active_learning_utils import calculate_kld_given_metric  
from analysis.dlvm_imle_comparison.plot_compare_search import compute_rmse, compute_kld  
  
# Global model cache for worker processes
MODEL_CACHE = {}

def get_cached_model(model_path: str, latent_dim: int):
    key = (str(model_path), int(latent_dim))
    model = MODEL_CACHE.get(key)
    if model is None:
        model = load_trained_model(latent_dim=latent_dim, model_path=model_path)
        MODEL_CACHE[key] = model
    return model

# Ensure safe start method for multiprocessing with PyTorch
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # Start method may already be set; ignore
    pass
  
# Standardized method names  
STANDARD_METHOD_NAMES = {  
    "DLVM_DALE": "DLVM_DALE",  
    "DLVM_RANDOM": "DLVM_RANDOM",  
    "IMLE_TB": "IMLE_TB",  
    "IMLE_Random": "IMLE_Random",  
    "DLVM_TB": "DLVM_TB",  
    "IMLE_DALE": "IMLE_DALE",  
}  
  
# Constants  
RELEVANT_METRICS_SPAN_CONDENSED = ["CorsiComplex_", "SimpleSpan_"]  
RELEVANT_METRICS_COMPLEX_SPAN = [metric for metric in RELEVANT_METRICS if "CorsiComplex" in metric]  
RELEVANT_METRICS_SIMPLE_SPAN = [metric for metric in RELEVANT_METRICS if "SimpleSpan" in metric]  
  
EXCLUDED_SESSIONS = set()  
  
KLD_normalizer_params = {  
    'CorsiComplex': 6.5,  
    'Stroop_reaction_time': 2.5,  
    'RunningSpan_correct_w_len_2': 0.5,  
    'Countermanding_reaction_time': 2.5,  
    'SimpleSpan': 6.5,  
    'RunningSpan_correct_w_len_3': 0.5,  
    'D2_hit_accuracy': 0.5,  
    'PasatPlus_correctly_answered': 0.5,  
}  
  
RMSE_normalizer_params = {  
    'CorsiComplex': [12.234659463167189, 1.9972472935914993],  
    'Stroop_reaction_time': [1.0243954658508292, 0.5401461943984033],  
    'RunningSpan_correct_w_len_2': [0.9800000097602607],  
    'Countermanding_reaction_time': [0.7413206100463867, 0.38473375886678696],  
    'SimpleSpan': [12.079994469881058, 1.9972472935914993],  
    'RunningSpan_correct_w_len_3': [0.9800000097602607],  
    'D2_hit_accuracy': [0.6566666662693024],  
    'PasatPlus_correctly_answered': [0.8899999763816596]  
}  
  
  
def is_excluded_session(session_id: str) -> bool:  
    return session_id in EXCLUDED_SESSIONS  
  
  
def compute_rmse_normalized(model_params, gt_params, metrics, parameters, logger,  
                            main_param_only=True, num_bootstrap=100):  
    return compute_rmse(  
        model_params, gt_params, metrics, parameters, logger,  
        main_param_only=main_param_only, num_bootstrap=num_bootstrap,  
        normalize_errors=True,  
        rmse_metric_normalizer=RMSE_normalizer_params,  
        use_std_dev=True  
    )  
  
  
def compute_kld_normalized(model_params, gt_params, metrics, logger, num_bootstrap=100):  
    return compute_kld(  
        model_params, gt_params, metrics,  
        logger, num_bootstrap=num_bootstrap,  
        normalize_errors=True,  
        rmse_metric_normalizer=KLD_normalizer_params,  
        use_std_dev=True  
    )  
  
  
def compute_metrics_for_iteration(model_params, gt_params, metrics, parameters, logger,  
                                  session_id, iteration, method_name):  
    rmse = None  
    kld = None  
    try:  
        rmse, _ = compute_rmse_normalized(model_params, gt_params, metrics, parameters, logger)  
        if np.isnan(rmse):  
            rmse = None  
        elif iteration is not None and iteration <= 5:  
            logger.debug(f"[{method_name}] {session_id} iteration {iteration}: RMSE = {rmse:.4f}")  
    except Exception as e:  
        logger.debug(f"Error computing RMSE for {method_name} {session_id} iteration {iteration}: {e}")  
    try:  
        kld, _ = compute_kld_normalized(model_params, gt_params, metrics, logger)  
        if np.isnan(kld):  
            kld = None  
        elif iteration is not None and iteration <= 5:  
            logger.debug(f"[{method_name}] {session_id} iteration {iteration}: KLD = {kld:.4f}")  
    except Exception as e:  
        logger.debug(f"Error computing KLD for {method_name} {session_id} iteration {iteration}: {e}")  
    return rmse, kld  
  
  
def parse_meu_z_sigma_z(value_str):  
    try:  
        parsed = ast.literal_eval(value_str)  
        if isinstance(parsed[0], list):  
            return np.array(parsed[0])  
        return np.array(parsed)  
    except Exception:  
        return None  
  
  
def fill_missing_mle_params(current_mle_params, summarized_metrics):  
    filled_params = current_mle_params.copy()  
    for metric in summarized_metrics:  
        if metric not in filled_params:  
            if metric in ['CorsiComplex', 'SimpleSpan']:  
                filled_params[metric] = [0.0, 0.00001]  
            elif 'reaction_time' in metric:  
                filled_params[metric] = [0.0, 0.0]  
            else:  
                filled_params[metric] = [0.0]  
    return filled_params  
  
  
def get_session_ids_from_results(result_dir: str) -> List[str]:
    """Return session IDs found under a result directory.

    None-safe and existence-safe: returns an empty list if the path is
    None or does not exist.
    """
    session_ids = []
    if not result_dir:
        return session_ids
    result_path = Path(result_dir)
    if not result_path.exists() or not result_path.is_dir():
        return session_ids
    session_dirs = [d for d in result_path.iterdir() if d.is_dir() and 'aggregate' not in d.name.lower()]
    for session_dir in session_dirs:
        sid = session_dir.name
        if is_excluded_session(sid):
            print(f"Skipping excluded session '{sid}'")
            continue
        session_ids.append(sid)
    return session_ids
  
  
def get_consistent_session_list(
    dale_ps0_dir: str,
    dale_ps1_dir: str,
    dale_ps2_dir: str,
    dale_ps4_dir: str,
    random_dir: str,
    session_number: int = None,
    logger=None,
) -> List[str]:
    """Get session IDs present across all available result directories.

    - None-safe for any of PS1/PS2/PS4; only intersects across directories
      that are provided (non-None) and exist.
    - Always requires PS0 and RANDOM to be present for meaningful comparison.
    """
    if logger:
        logger.info(f"=== Getting Consistent Session List ===")
        logger.info(f"DLVM_DALE PS0 directory: {dale_ps0_dir}")
        logger.info(f"DLVM_DALE PS1 directory: {dale_ps1_dir}")
        logger.info(f"DLVM_DALE PS2 directory: {dale_ps2_dir}")
        logger.info(f"DLVM_DALE PS4 directory: {dale_ps4_dir}")
        logger.info(f"DLVM_RANDOM directory: {random_dir}")

    # Build list of present directories (label, path)
    dirs = [
        ("PS0", dale_ps0_dir),
        ("PS1", dale_ps1_dir),
        ("PS2", dale_ps2_dir),
        ("PS4", dale_ps4_dir),
        ("RANDOM", random_dir),
    ]
    present = []
    for label, path in dirs:
        if path is None:
            if logger:
                logger.info(f"{label} directory: None (skipping)")
            continue
        p = Path(path)
        if not p.exists() or not p.is_dir():
            if logger:
                logger.info(f"{label} directory: {p} not found or not a directory (skipping)")
            continue
        present.append((label, p))

    # Ensure we at least have PS0 and RANDOM
    present_labels = {label for label, _ in present}
    if "PS0" not in present_labels or "RANDOM" not in present_labels:
        print("Error: PS0 and RANDOM directories are required to compute consistent sessions.")
        return []

    # Collect sessions per present directory
    sessions_per_dir = {}
    for label, p in present:
        sessions_per_dir[label] = set(get_session_ids_from_results(p))

    # Intersect across all present directories
    iter_sets = iter(sessions_per_dir.values())
    try:
        common_sessions = next(iter_sets)
    except StopIteration:
        common_sessions = set()
    for s in iter_sets:
        common_sessions = common_sessions.intersection(s)

    sorted_sessions = sorted(list(common_sessions))
    if session_number is not None:
        sorted_sessions = sorted_sessions[:session_number]

    if logger:
        for label in ["PS0", "PS1", "PS2", "PS4", "RANDOM"]:
            if label in sessions_per_dir:
                logger.info(f"DALE {label} sessions: {len(sessions_per_dir[label])}")
            else:
                logger.info(f"DALE {label} sessions: skipped")
        logger.info(f"Common sessions: {len(common_sessions)}")
        logger.info(f"Total sessions to process: {len(sorted_sessions)}")

    print(f"Using consistent session list across {len(present)} directories: {len(sorted_sessions)} sessions")
    return sorted_sessions
  
  
def load_ground_truth_with_key_matching(mle_params_file: str, session_keys: List[str]) -> Dict:  
    print(f"\n=== Loading Ground Truth MLE Parameters ===")  
    all_mle_params = torch.load(mle_params_file)  
  
    excluded_in_keys = [sid for sid in session_keys if is_excluded_session(sid)]  
    if excluded_in_keys:  
        print(f"Excluding sessions (global blocklist): {excluded_in_keys}")  
    session_keys = [sid for sid in session_keys if not is_excluded_session(sid)]  
  
    ground_truth = {}  
    matched_count = 0  
  
    for session_id in session_keys:  
        if session_id in all_mle_params:  
            ground_truth[session_id] = all_mle_params[session_id]  
            matched_count += 1  
            print(f"✓ Matched session '{session_id}'")  
        else:  
            print(f"✗ Session '{session_id}' not found in ground truth")  
    print(f"\nSuccessfully matched {matched_count}/{len(session_keys)} sessions")  
    return ground_truth  
  
def _process_session_load_and_extract(session_dir: Path, ground_truth: Dict, metrics, parameters,
                                      max_iterations: int, method_label: str,
                                      model_path: str, latent_dim: int):
    """Worker: process a single session directory for DLVM_DALE or DLVM_RANDOM.
    Loads model inside the worker to avoid cross-process pickling issues."""
    session_id = session_dir.name
    iteration_data = {
        'num_tests': [],
        'kld_values': [],
        'rmse_values': []
    }
    # create a local logger in the worker
    local_logger = logging.getLogger(f"{__name__}.{method_label}.worker")
    try:
        model = get_cached_model(model_path, latent_dim)
    except Exception:
        model = None

    csv_paths = [
        session_dir / "analysis" / f"performance_tracking_session_{session_id}.csv",
        session_dir / f"performance_tracking_session_{session_id}.csv"
    ]
    for csv_path in csv_paths:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if max_iterations is not None:
                df = df.head(max_iterations)
            for _, row in df.iterrows():
                meu_z = parse_meu_z_sigma_z(row['meu_z'])
                if meu_z is None or model is None:
                    continue
                meu_z_tensor = torch.tensor(meu_z, dtype=torch.float32).unsqueeze(0)
                try:
                    predicted_params = get_predictions_dicts_from_latent_points(
                        meu_z_tensor, model, model_type="NN", with_activation=True
                    )[0]
                except Exception:
                    continue
                run_id = f"{session_id}_sim1"
                model_params = {run_id: predicted_params}
                gt_params = {session_id: ground_truth[session_id]}
                try:
                    iter_for_log = int(row.get('num_tests_run', None))
                except Exception:
                    iter_for_log = None
                rmse, kld = compute_metrics_for_iteration(
                    model_params, gt_params, metrics, parameters, local_logger,
                    session_id, iter_for_log, method_label
                )
                if rmse is not None or kld is not None:
                    if iter_for_log is None:
                        try:
                            iter_for_log = int(row.get('iteration', None))
                        except Exception:
                            iter_for_log = None
                    iteration_data['num_tests'].append(iter_for_log if iter_for_log is not None else np.nan)
                    iteration_data['kld_values'].append(kld if kld is not None else np.nan)
                    iteration_data['rmse_values'].append(rmse if rmse is not None else np.nan)
            break  # prefer the first existing CSV path
    return session_id, iteration_data

  
def load_and_extract_predicted_params(  
    result_dir: str,  
    model,  
    ground_truth: Dict,  
    metrics,  
    parameters,  
    logger,  
    session_list: List[str] = None,  
    max_iterations: int = None,  
    method_label: str = "DLVM_DALE",  
    workers: int = 1,
    model_path: str = None,
    latent_dim: int = None,
    append_rows_fn=None,
) -> Dict:  
    results = {}  
    result_path = Path(result_dir)  
  
    if session_list is not None:  
        session_dirs = []  
        for session_id in session_list:  
            session_dir = result_path / session_id  
            if session_dir.exists() and session_dir.is_dir():  
                session_dirs.append(session_dir)  
        logger.info(f"[{method_label}] Processing {len(session_dirs)} sessions from provided session list")
        print(f"  → [{method_label}] Found {len(session_dirs)} valid sessions to process")
    else:  
        session_dirs = [d for d in result_path.iterdir() if d.is_dir() and 'aggregate' not in d.name.lower()]  
        logger.info(f"[{method_label}] Processing {len(session_dirs)} sessions from directory scan")
        print(f"  → [{method_label}] Found {len(session_dirs)} sessions to process")

    total_sessions = len(session_dirs)
    processed_count = 0
    futures = []
    with ProcessPoolExecutor(max_workers=max(1, int(workers)), mp_context=mp.get_context('spawn')) as ex:
        for session_dir in session_dirs:
            session_id = session_dir.name
            print(f"  → [{method_label}] Processing session: {session_id}")
            if is_excluded_session(session_id):
                print(f"    ✗ Skipping excluded session '{session_id}'")
                continue
            if session_id not in ground_truth:
                print(f"    ✗ Skipping session {session_id} - no ground truth available")
                continue
            futures.append(ex.submit(
                _process_session_load_and_extract,
                session_dir, ground_truth, metrics, parameters, max_iterations, method_label, model_path, latent_dim
            ))
        for fut in as_completed(futures):
            try:
                session_id, iteration_data = fut.result()
                if iteration_data['num_tests']:
                    results[session_id] = iteration_data
                    processed_count += 1
                    print(f"    ✓ Loaded session {session_id}: {len(iteration_data['num_tests'])} iterations")
                    if append_rows_fn is not None:
                        try:
                            append_rows_fn(method_label, session_id, iteration_data)
                        except Exception as e:
                            logger.error(f"Failed streaming CSV rows for {method_label}:{session_id}: {e}")
                else:
                    print(f"    ✗ No valid data found for session {session_id}")
            except Exception as e:
                logger.error(f"{method_label} worker error: {e}")

    print(f"  → [{method_label}] Completed: {processed_count}/{total_sessions} sessions processed successfully")
    return results  
  
  
def _process_session_imle_tb(session_item, synthetic_data: Dict, ground_truth: Dict, metrics, parameters,
                             imle_task_switch_interval: int):
    session_id, dale_data = session_item
    synthetic_key = f"{session_id}_sim1"
    if synthetic_key not in synthetic_data:
        return session_id, None
    if session_id not in ground_truth:
        return session_id, None
    task_sequence = SUMMARIZED_METRICS.copy()
    np.random.shuffle(task_sequence)
    imle_metrics_data = {metric: [] for metric in RELEVANT_METRICS}
    data_indices = {metric: 0 for metric in RELEVANT_METRICS}
    current_task_index = 0
    task_iteration_count = 0
    current_selected_metric = None
    iteration_data = {'num_tests': [], 'kld_values': [], 'rmse_values': []}
    num_iterations = len(dale_data['num_tests'])
    for iteration in range(num_iterations):
        if task_iteration_count >= imle_task_switch_interval:
            current_task_index = (current_task_index + 1) % len(task_sequence)
            task_iteration_count = 0
        test_to_run_next = task_sequence[current_task_index]
        if "CorsiComplex" == test_to_run_next:
            current_selected_metric = random.choice(RELEVANT_METRICS_COMPLEX_SPAN)
        elif "SimpleSpan" == test_to_run_next:
            current_selected_metric = random.choice(RELEVANT_METRICS_SIMPLE_SPAN)
        else:
            current_selected_metric = test_to_run_next
        if current_selected_metric in synthetic_data[synthetic_key]:
            idx = data_indices[current_selected_metric]
            if idx < len(synthetic_data[synthetic_key][current_selected_metric]):
                value = synthetic_data[synthetic_key][current_selected_metric][idx]
                imle_metrics_data[current_selected_metric].append(value)
                data_indices[current_selected_metric] += 1
        task_iteration_count += 1
        has_data = any(len(imle_metrics_data[m]) > 0 for m in RELEVANT_METRICS)
        if not has_data:
            continue
        try:
            current_mle_params = get_mle_params_per_metric(
                imle_metrics_data,
                mpf=100,
                metrics=RELEVANT_METRICS
            )
            current_mle_params = fill_missing_mle_params(current_mle_params, SUMMARIZED_METRICS)
            run_id = f"{session_id}_sim1"
            model_params = {run_id: current_mle_params}
            gt_params = {session_id: ground_truth[session_id]}
            num_tests_run = dale_data['num_tests'][iteration]
            rmse, kld = compute_metrics_for_iteration(
                model_params, gt_params, metrics, parameters, logger,
                session_id, num_tests_run, "IMLE_TB"
            )
            if rmse is not None or kld is not None:
                iteration_data['num_tests'].append(num_tests_run)
                iteration_data['kld_values'].append(kld if kld is not None else np.nan)
                iteration_data['rmse_values'].append(rmse if rmse is not None else np.nan)
        except Exception:
            continue
    return session_id, iteration_data if iteration_data['num_tests'] else None

def compute_imle_tb_data(dale_iteration_data: Dict, synthetic_data: Dict, ground_truth: Dict, model, metrics, parameters, logger, workers: int = 1, imle_task_switch_interval: int = 25, append_rows_fn=None, method_label: str = "IMLE_TB") -> Dict:  
    imle_results = {}  
    logger.info(f"Starting IMLE_TB computation for {len(dale_iteration_data)} sessions")
    total_sessions = len(dale_iteration_data)
    processed_count = 0
    print(f"  → [IMLE_TB] Processing {total_sessions} sessions")
    futures = []
    with ProcessPoolExecutor(max_workers=max(1, int(workers)), mp_context=mp.get_context('spawn')) as ex:
        for item in dale_iteration_data.items():
            futures.append(ex.submit(_process_session_imle_tb, item, synthetic_data, ground_truth, metrics, parameters, imle_task_switch_interval))
        for fut in as_completed(futures):
            try:
                session_id, data = fut.result()
                if data:
                    imle_results[session_id] = data
                    processed_count += 1
                    print(f"    ✓ IMLE_TB session {session_id}: {len(data['num_tests'])} iterations stored")
                    if append_rows_fn is not None:
                        try:
                            append_rows_fn(method_label, session_id, data)
                        except Exception as e:
                            logger.error(f"Failed streaming CSV rows for {method_label}:{session_id}: {e}")
                else:
                    print(f"    ✗ No valid data generated for IMLE_TB session {session_id}")
            except Exception as e:
                logger.error(f"IMLE_TB worker error: {e}")
    print(f"  → [IMLE_TB] Completed: {processed_count}/{total_sessions} sessions processed successfully")
    logger.info(f"IMLE_TB computation complete: {len(imle_results)} sessions with results")  
    return imle_results  
  
  
def _process_session_imle_random(session_item, synthetic_data: Dict, ground_truth: Dict, metrics, parameters):
    session_id, dale_data = session_item
    synthetic_key = f"{session_id}_sim1"
    if synthetic_key not in synthetic_data:
        return session_id, None
    if session_id not in ground_truth:
        return session_id, None
    imle_metrics_data = {metric: [] for metric in RELEVANT_METRICS}
    data_indices = {metric: 0 for metric in RELEVANT_METRICS}
    iteration_data = {'num_tests': [], 'kld_values': [], 'rmse_values': []}
    num_iterations = len(dale_data['num_tests'])
    for iteration in range(num_iterations):
        test_to_run_next = random.choice(SUMMARIZED_METRICS)
        if "CorsiComplex" in test_to_run_next:
            current_selected_metric = random.choice(RELEVANT_METRICS_COMPLEX_SPAN)
        elif "SimpleSpan" in test_to_run_next:
            current_selected_metric = random.choice(RELEVANT_METRICS_SIMPLE_SPAN)
        else:
            current_selected_metric = test_to_run_next
        if current_selected_metric in synthetic_data[synthetic_key]:
            idx = data_indices[current_selected_metric]
            if idx < len(synthetic_data[synthetic_key][current_selected_metric]):
                value = synthetic_data[synthetic_key][current_selected_metric][idx]
                imle_metrics_data[current_selected_metric].append(value)
                data_indices[current_selected_metric] += 1
        has_data = any(len(imle_metrics_data[m]) > 0 for m in RELEVANT_METRICS)
        if not has_data:
            continue
        try:
            current_mle_params = get_mle_params_per_metric(
                imle_metrics_data,
                mpf=100,
                metrics=RELEVANT_METRICS
            )
            current_mle_params = fill_missing_mle_params(current_mle_params, SUMMARIZED_METRICS)
            run_id = f"{session_id}_sim1"
            model_params = {run_id: current_mle_params}
            gt_params = {session_id: ground_truth[session_id]}
            num_tests_run = dale_data['num_tests'][iteration]
            rmse, kld = compute_metrics_for_iteration(
                model_params, gt_params, metrics, parameters, logger,
                session_id, num_tests_run, "IMLE_Random"
            )
            if rmse is not None or kld is not None:
                iteration_data['num_tests'].append(num_tests_run)
                iteration_data['kld_values'].append(kld if kld is not None else np.nan)
                iteration_data['rmse_values'].append(rmse if rmse is not None else np.nan)
        except Exception:
            continue
    return session_id, iteration_data if iteration_data['num_tests'] else None

def compute_imle_random_data(dale_iteration_data: Dict, synthetic_data: Dict, ground_truth: Dict, model, metrics, parameters, logger, workers: int = 1, append_rows_fn=None, method_label: str = "IMLE_Random") -> Dict:  
    imle_random_results = {}  
    logger.info(f"Starting IMLE_Random computation for {len(dale_iteration_data)} sessions")
    total_sessions = len(dale_iteration_data)
    processed_count = 0
    print(f"  → [IMLE_Random] Processing {total_sessions} sessions")
    futures = []
    with ProcessPoolExecutor(max_workers=max(1, int(workers)), mp_context=mp.get_context('spawn')) as ex:
        for item in dale_iteration_data.items():
            futures.append(ex.submit(_process_session_imle_random, item, synthetic_data, ground_truth, metrics, parameters))
        for fut in as_completed(futures):
            try:
                session_id, data = fut.result()
                if data:
                    imle_random_results[session_id] = data
                    processed_count += 1
                    print(f"    ✓ IMLE_Random session {session_id}: {len(data['num_tests'])} iterations stored")
                    if append_rows_fn is not None:
                        try:
                            append_rows_fn(method_label, session_id, data)
                        except Exception as e:
                            logger.error(f"Failed streaming CSV rows for {method_label}:{session_id}: {e}")
                else:
                    print(f"    ✗ No valid data generated for IMLE_Random session {session_id}")
            except Exception as e:
                logger.error(f"IMLE_Random worker error: {e}")
    print(f"  → [IMLE_Random] Completed: {processed_count}/{total_sessions} sessions processed successfully")
    logger.info(f"IMLE_Random computation complete: {len(imle_random_results)} sessions with results")  
    return imle_random_results  
  
  
def _process_session_dlvm_tb(session_item, synthetic_data: Dict, ground_truth: Dict, metrics, parameters,
                             imle_task_switch_interval: int, model_path: str, latent_dim: int):
    session_id, dale_data = session_item
    synthetic_key = f"{session_id}_sim1"
    if synthetic_key not in synthetic_data:
        return session_id, None
    if session_id not in ground_truth:
        return session_id, None
    try:
        model = get_cached_model(model_path, latent_dim)
    except Exception:
        return session_id, None
    task_sequence = SUMMARIZED_METRICS.copy()
    np.random.shuffle(task_sequence)
    accumulated_data = {metric: [] for metric in RELEVANT_METRICS}
    data_indices = {metric: 0 for metric in RELEVANT_METRICS}
    current_task_index = 0
    task_iteration_count = 0
    current_selected_metric = None
    iteration_data = {'num_tests': [], 'kld_values': [], 'rmse_values': []}
    num_iterations = len(dale_data['num_tests'])
    for iteration in range(num_iterations):
        if task_iteration_count >= imle_task_switch_interval:
            current_task_index = (current_task_index + 1) % len(task_sequence)
            task_iteration_count = 0
        test_to_run_next = task_sequence[current_task_index]
        if "CorsiComplex" == test_to_run_next:
            current_selected_metric = random.choice(RELEVANT_METRICS_COMPLEX_SPAN)
        elif "SimpleSpan" == test_to_run_next:
            current_selected_metric = random.choice(RELEVANT_METRICS_SIMPLE_SPAN)
        else:
            current_selected_metric = test_to_run_next
        if current_selected_metric in synthetic_data[synthetic_key]:
            idx = data_indices[current_selected_metric]
            if idx < len(synthetic_data[synthetic_key][current_selected_metric]):
                value = synthetic_data[synthetic_key][current_selected_metric][idx]
                accumulated_data[current_selected_metric].append(value)
                data_indices[current_selected_metric] += 1
        task_iteration_count += 1
        has_data = any(len(accumulated_data[m]) > 0 for m in RELEVANT_METRICS)
        if not has_data:
            continue
        try:
            current_dlvm_params = predict_parameters_from_data(
                accumulated_data, model, num_points=200, use_grid_search=True
            )
            run_id = f"{session_id}_sim1"
            model_params = {run_id: current_dlvm_params}
            gt_params = {session_id: ground_truth[session_id]}
            num_tests_run = dale_data['num_tests'][iteration]
            rmse, kld = compute_metrics_for_iteration(
                model_params, gt_params, metrics, parameters, logger,
                session_id, num_tests_run, "DLVM_TB"
            )
            if rmse is not None or kld is not None:
                iteration_data['num_tests'].append(num_tests_run)
                iteration_data['kld_values'].append(kld if kld is not None else np.nan)
                iteration_data['rmse_values'].append(rmse if rmse is not None else np.nan)
        except Exception:
            continue
    return session_id, iteration_data if iteration_data['num_tests'] else None

def compute_dlvm_tb_data(dale_iteration_data: Dict, synthetic_data: Dict, ground_truth: Dict, model, metrics, parameters, logger, workers: int = 1, imle_task_switch_interval: int = 25, model_path: str = None, latent_dim: int = None, append_rows_fn=None, method_label: str = "DLVM_TB") -> Dict:  
    dlvm_tb_results = {}  
    logger.info(f"Starting DLVM_TB computation for {len(dale_iteration_data)} sessions")
    total_sessions = len(dale_iteration_data)
    processed_count = 0
    print(f"  → [DLVM_TB] Processing {total_sessions} sessions")  
    futures = []
    with ProcessPoolExecutor(max_workers=max(1, int(workers)), mp_context=mp.get_context('spawn')) as ex:
        for item in dale_iteration_data.items():
            futures.append(ex.submit(_process_session_dlvm_tb, item, synthetic_data, ground_truth, metrics, parameters, imle_task_switch_interval, model_path, latent_dim))
        for fut in as_completed(futures):
            try:
                session_id, data = fut.result()
                if data:
                    dlvm_tb_results[session_id] = data
                    processed_count += 1
                    print(f"    ✓ DLVM_TB session {session_id}: {len(data['num_tests'])} iterations stored")
                    if append_rows_fn is not None:
                        try:
                            append_rows_fn(method_label, session_id, data)
                        except Exception as e:
                            logger.error(f"Failed streaming CSV rows for {method_label}:{session_id}: {e}")
                else:
                    print(f"    ✗ No valid data generated for DLVM_TB session {session_id}")
            except Exception as e:
                logger.error(f"DLVM_TB worker error: {e}")
    print(f"  → [DLVM_TB] Completed: {processed_count}/{total_sessions} sessions processed successfully")
    logger.info(f"DLVM_TB computation complete: {len(dlvm_tb_results)} sessions with results")  
    return dlvm_tb_results  
  
def _process_session_imle_dale(session_dir: Path, ground_truth: Dict, metrics, parameters, max_iterations: int):
    session_id = session_dir.name
    if is_excluded_session(session_id):
        return session_id, None
    if session_id not in ground_truth:
        return session_id, None
    data_dir = session_dir / "data"
    if not data_dir.exists():
        return session_id, None
    data_files = []
    for file_path in data_dir.glob("num_tests_run_*_update_w_data_session_*.pt"):
        try:
            parts = file_path.stem.split('_')
            iteration_idx = None
            for i, part in enumerate(parts):
                if part == "run" and i + 1 < len(parts):
                    iteration_num = int(parts[i + 1])
                    iteration_idx = iteration_num
                    break
            if iteration_idx is not None:
                data_files.append((iteration_idx, file_path))
        except (ValueError, IndexError):
            continue
    data_files.sort(key=lambda x: x[0])
    if max_iterations is not None:
        data_files = data_files[:max_iterations]
    if not data_files:
        return session_id, None
    iteration_data = {'num_tests': [], 'kld_values': [], 'rmse_values': []}
    for iteration_num, file_path in data_files:
        try:
            imle_metrics_data = torch.load(file_path)
            current_mle_params = get_mle_params_per_metric(
                imle_metrics_data,
                mpf=100,
                metrics=RELEVANT_METRICS
            )
            current_mle_params = fill_missing_mle_params(current_mle_params, SUMMARIZED_METRICS)
            run_id = f"{session_id}_sim1"
            model_params = {run_id: current_mle_params}
            gt_params = {session_id: ground_truth[session_id]}
            num_tests_run = iteration_num + 1
            rmse, kld = compute_metrics_for_iteration(
                model_params, gt_params, metrics, parameters, logger,
                session_id, num_tests_run, "IMLE_DALE"
            )
            if rmse is not None or kld is not None:
                iteration_data['num_tests'].append(num_tests_run)
                iteration_data['kld_values'].append(kld if kld is not None else np.nan)
                iteration_data['rmse_values'].append(rmse if rmse is not None else np.nan)
        except Exception:
            continue
    return session_id, iteration_data if iteration_data['num_tests'] else None

def compute_imle_dale_data(dale_dir: str, ground_truth: Dict, model, metrics, parameters, logger, session_list: List[str] = None, max_iterations: int = None, workers: int = 1, append_rows_fn=None, method_label: str = "IMLE_DALE") -> Dict:  
    results = {}  
    result_path = Path(dale_dir)  
  
    if session_list is not None:  
        session_dirs = []  
        for session_id in session_list:  
            session_dir = result_path / session_id  
            if session_dir.exists() and session_dir.is_dir():  
                session_dirs.append(session_dir)  
        logger.info(f"IMLE_DALE processing {len(session_dirs)} sessions from provided session list")
        print(f"  → [IMLE_DALE] Found {len(session_dirs)} valid sessions to process")  
    else:  
        session_dirs = [d for d in result_path.iterdir() if d.is_dir() and 'aggregate' not in d.name.lower()]
        print(f"  → [IMLE_DALE] Found {len(session_dirs)} sessions to process")  
  
    total_sessions = len(session_dirs)
    processed_count = 0
    futures = []
    with ProcessPoolExecutor(max_workers=max(1, int(workers)), mp_context=mp.get_context('spawn')) as ex:
        for session_dir in session_dirs:
            futures.append(ex.submit(_process_session_imle_dale, session_dir, ground_truth, metrics, parameters, max_iterations))
        for fut in as_completed(futures):
            try:
                session_id, data = fut.result()
                if data:
                    results[session_id] = data
                    processed_count += 1
                    print(f"    ✓ Loaded IMLE_DALE session {session_id}: {len(data['num_tests'])} iterations")
                    if append_rows_fn is not None:
                        try:
                            append_rows_fn(method_label, session_id, data)
                        except Exception as e:
                            logger.error(f"Failed streaming CSV rows for {method_label}:{session_id}: {e}")
                else:
                    print(f"    ✗ No valid data generated for IMLE_DALE session {session_id}")
            except Exception as e:
                logger.error(f"IMLE_DALE worker error: {e}")
    print(f"  → [IMLE_DALE] Completed: {processed_count}/{total_sessions} sessions processed successfully")  
    return results  
  
  
  
def save_all_data_to_csv(all_method_data: Dict, output_path: Path, metadata: Dict = None):  
    """  
    Save all method data to a single CSV file with proper structure.  
  
    Args:  
        all_method_data: Dictionary with method names as keys and session data as values  
        output_path: Path to save the CSV file  
        metadata: Optional metadata to save alongside  
    """  
    # Prepare rows for CSV  
    csv_rows = []  
  
    for method_name, method_data in all_method_data.items():  
        for session_id, session_data in method_data.items():  
            for i in range(len(session_data['num_tests'])):  
                row = {  
                    'method': method_name,  
                    'session_id': session_id,  
                    'num_tests': int(session_data['num_tests'][i]),  
                    'kld_value': session_data['kld_values'][i],  
                    'rmse_value': session_data['rmse_values'][i]  
                }  
                csv_rows.append(row)  
  
    # Create DataFrame and save  
    df = pd.DataFrame(csv_rows)  
  
    # Sort for consistency  
    df = df.sort_values(['method', 'session_id', 'num_tests'])  
  
    # Save to CSV  
    csv_file = output_path / 'dale_comparison_data.csv'  
    df.to_csv(csv_file, index=False)  
    print(f"\nSaved {len(df)} rows to {csv_file}")  
  
    # Save metadata if provided  
    if metadata:  
        metadata_file = output_path / 'dale_comparison_metadata.json'  
        with open(metadata_file, 'w') as f:  
            json.dump(metadata, f, indent=2, default=str)  
        print(f"Saved metadata to {metadata_file}")  
  
    # Print summary statistics  
    print("\n=== Data Summary ===")  
    for method in df['method'].unique():  
        method_df = df[df['method'] == method]  
        n_sessions = method_df['session_id'].nunique()  
        n_rows = len(method_df)  
        print(f"{method}: {n_sessions} sessions, {n_rows} data points")  
  
    return df  
   

def main():  
    # Path setup  
    SCRIPT_ROOT = Path(__file__).resolve().parent  
    REPO_ROOT = Path(__file__).resolve().parents[2]  
    DEFAULT_RES = SCRIPT_ROOT / "result" / "re-2d-3d-full-final"  
  
    # Define preset configurations directly in main
    PLOT_DATA_DIR = REPO_ROOT / "analysis" / "dale_syn_data" / "plot_data"
    SYN_DATA_2D_PATH = REPO_ROOT / "analysis/dlvm_imle_comparison/synthetic_data/COLL10_SIM/D2_all_synthetic_data_N240.pt"  
    SYN_DATA_3D_PATH = REPO_ROOT/ "analysis/dlvm_imle_comparison/synthetic_data/COLL10_SIM/D3_all_synthetic_data_N240.pt"
    GROUND_TRUTH_MLE_2D_PATH = REPO_ROOT / "data/COLL10_SIM/D2_synthetic_ground_truth_parameters_mongoose-dive-7464.pt"
    GROUND_TRUTH_MLE_3D_PATH = REPO_ROOT / "data/COLL10_SIM/D3_synthetic_ground_truth_parameters_beaver-slide-5310.pt"  
  
    MODEL_3D_BEAVER = str(REPO_ROOT / 'saved_models/COLL10_SIM/heldout_obsmulti/variationalNN_relevant_only_latentdim3_beaver-slide-5310.pt')
    MODEL_2D_MONGOOSE = str(REPO_ROOT / 'saved_models/COLL10_SIM/heldout_obsmulti/variationalNN_relevant_only_latentdim2_mongoose-dive-7464.pt')

    configs = {
        "2d": {
            "description": "Current 2D experiment configuration",
            "dale_ps0_dir": DEFAULT_RES / "kraken-lunge-6972",
            "dale_ps1_dir": DEFAULT_RES / "panther-lunge-6972",
            "dale_ps2_dir": DEFAULT_RES / "wolverine-bound-6972",
            "dale_ps4_dir": DEFAULT_RES / "golem-zoom-7099",
            "random_dir": DEFAULT_RES / "raven-climb-6972", 
            "output_dir": PLOT_DATA_DIR / "2d_240_full",
            "model_path": MODEL_2D_MONGOOSE,
            "latent_dim": 2,
            "mle_params_file": str(GROUND_TRUTH_MLE_2D_PATH),
            "synthetic_data_path": str(SYN_DATA_2D_PATH),
            "compute_imle_tb": True,
            "compute_imle_random": True,
            "compute_dlvm_tb": True,
            "compute_imle_dale": True,
            "session_number": 88,
            "max_iterations": 240,
            "imle_task_switch_interval": 30
        },
        "3d": {
            "description": "3D experiment configuration",
            "dale_ps0_dir": DEFAULT_RES / "saber-skim-6935",
            "dale_ps1_dir": DEFAULT_RES / "kraken-shoot-6972",
            "dale_ps2_dir": DEFAULT_RES / "hound-hurdle-6972",
            "dale_ps4_dir": DEFAULT_RES / "otter-bound-6972",
            "random_dir": DEFAULT_RES / "tempest-hurdle-6972", 
            "output_dir": PLOT_DATA_DIR / "3d_240_full",
            "model_path": MODEL_3D_BEAVER,
            "latent_dim": 3,
            "mle_params_file": str(GROUND_TRUTH_MLE_3D_PATH),
            "synthetic_data_path": str(SYN_DATA_3D_PATH),
            "compute_imle_tb": True,
            "compute_imle_random": True,
            "compute_dlvm_tb": True,
            "compute_imle_dale": True,
            "session_number": 88,
            "max_iterations": 240,
            "imle_task_switch_interval": 30
        },

        # golem-zoom-7099	2d_dale_ps=4_golem-zoom-7099
        # hound-hurdle-6972	3d_dale_ps=2_hound-hurdle-6972
        # kraken-lunge-6972	2d_dale_ps=0_kraken-lunge-6972
        # kraken-shoot-6972	3d_dale_ps=1_kraken-shoot-6972
        # otter-bound-6972	3d_dale_ps=4_otter-bound-6972
        # panther-lunge-6972	2d_dale_ps=1_panther-lunge-6972
        # raven-climb-6972	2d_random_ps=0_raven-climb-6972
        # saber-skim-6935	3d_dale_ps=0_saber-skim-6935
        # tempest-hurdle-6972	3d_random_ps=0_tempest-hurdle-6972
        # wolverine-bound-6972	2d_dale_ps=2_wolverine-bound-6972


        # basilisk-skim-4097    2d_random_ps=0_basilisk-skim-4097
        # hyena-run-4097        2d_dale_ps=0_hyena-run-4097
        # bear-thrust-4097      2d_dale_ps=4_bear-thrust-4097

        # civet-lunge-4126      3d_dale_ps=0_civet-lunge-4126
        # cougar-climb-4126     3d_dale_ps=4_cougar-climb-4126
        # yeti-swoop-4126       3d_random_ps=0_yeti-swoop-4126
        "re100_3d": {
            "description": "3D experiment configuration",
            "dale_ps0_dir": DEFAULT_RES / "exp_c1_3d_dale_ps0",
            "dale_ps1_dir": None,
            "dale_ps2_dir": DEFAULT_RES / "exp_c2_3d_dale_ps2",
            "dale_ps4_dir": DEFAULT_RES / "exp_c3_3d_dale_ps4",
            "random_dir": DEFAULT_RES / "exp_c4_3d_random", 
            "output_dir": PLOT_DATA_DIR / "re100_3d_240_concurrent_2",
            "model_path": MODEL_3D_BEAVER,
            "latent_dim": 3,
            "mle_params_file": str(GROUND_TRUTH_MLE_3D_PATH),
            "synthetic_data_path": str(SYN_DATA_3D_PATH),
            "compute_imle_tb": True,
            "compute_imle_random": True,
            "compute_dlvm_tb": True,
            "compute_imle_dale": True,
            "session_number": 88,
            "max_iterations": 240,
            "imle_task_switch_interval": 30
        },
        "re10_2d": {
            "description": "2D experiment configuration",
            "dale_ps0_dir": DEFAULT_RES / "exp_c5_2d_dale_ps0",
            "dale_ps1_dir": None,
            "dale_ps2_dir": DEFAULT_RES / "exp_c6_2d_dale_ps2",
            "dale_ps4_dir": DEFAULT_RES / "exp_c7_2d_dale_ps4",
            "random_dir": DEFAULT_RES / "exp_c8_2d_random", 
            "output_dir": PLOT_DATA_DIR / "re10_2d_240_concurrent_2",
            "model_path": MODEL_2D_MONGOOSE,
            "latent_dim": 2,
            "mle_params_file": str(GROUND_TRUTH_MLE_2D_PATH),
            "synthetic_data_path": str(SYN_DATA_2D_PATH),
            "compute_imle_tb": True,
            "compute_imle_random": True,
            "compute_dlvm_tb": True,
            "compute_imle_dale": True,
            "session_number": 88,
            "max_iterations": 240,
            "imle_task_switch_interval": 30
        },
        

    }


    # First, create parser to check for config and list_configs arguments
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="re10_2d", help="Use preset configuration")
    pre_parser.add_argument("--list-configs", action="store_true", help="List available preset configurations and exit")
    pre_args, _ = pre_parser.parse_known_args()

    # Handle list configs request
    if pre_args.list_configs:
        print("\nAvailable preset configurations:")
        print("=" * 50)
        for name, config in configs.items():
            print(f"  {name}: {config['description']}")
            def _nm(p):
                try:
                    return p.name
                except Exception:
                    return str(p)
            print(f"    - Dale PS0 dir: {_nm(config['dale_ps0_dir'])}")
            print(f"    - Dale PS1 dir: {_nm(config['dale_ps1_dir'])}")
            print(f"    - Dale PS2 dir: {_nm(config['dale_ps2_dir'])}")
            print(f"    - Dale PS4 dir: {_nm(config['dale_ps4_dir'])}")
            print(f"    - Random dir: {_nm(config['random_dir'])}")
            print(f"    - Model: {'3D' if config['latent_dim'] == 3 else '2D'} (latent_dim={config['latent_dim']})")
            print(f"    - Sessions: {config['session_number'] or None}")
            print(f"    - Max iterations: {config['max_iterations']}")
            print()
        return 0

    # Load preset config if specified
    config_defaults = {}
    if pre_args.config:
        if pre_args.config not in configs:
            available = list(configs.keys())
            print(f"Error: Configuration '{pre_args.config}' not found. Available: {available}")
            return 1
        config_defaults = configs[pre_args.config]
        print(f"Using preset configuration: {pre_args.config}")
        print(f"Description: {config_defaults['description']}")

    # Argument parsing with config defaults  
    parser = argparse.ArgumentParser(description="Compute and save DALE comparison data for plotting.")  
    parser.add_argument("--config", type=str,help="Use preset configuration (2d_current, 3d_experiment, 2d_quick)")
    parser.add_argument("--list-configs", action="store_true", help="List available preset configurations and exit")
    parser.add_argument("--dale_ps0_dir", type=Path, default=config_defaults.get("dale_ps0_dir", DEFAULT_RES / "hound-charge-1105"))
    parser.add_argument("--dale_ps1_dir", type=Path, default=config_defaults.get("dale_ps1_dir", DEFAULT_RES / "sparrow-hunt-3210"))
    parser.add_argument("--dale_ps2_dir", type=Path, default=config_defaults.get("dale_ps2_dir", DEFAULT_RES / "falcon-dive-8765"))
    parser.add_argument("--dale_ps4_dir", type=Path, default=config_defaults.get("dale_ps4_dir", DEFAULT_RES / "eagle-soar-4321"))  
    parser.add_argument("--random_dir", type=Path, default=config_defaults.get("random_dir", DEFAULT_RES / "cyclone-whirl-0252"))  
    parser.add_argument("--output_dir", type=Path, default=config_defaults.get("output_dir", PLOT_DATA_DIR))  
    parser.add_argument("--model_path", type=str, default=config_defaults.get("model_path", str(MODEL_2D_MONGOOSE)))  
    parser.add_argument("--latent_dim", type=int, default=config_defaults.get("latent_dim", 2))  
    parser.add_argument("--mle_params_file", type=str, default=config_defaults.get("mle_params_file", str(GROUND_TRUTH_MLE_2D_PATH)))  
    parser.add_argument("--synthetic_data_path", type=str, default=config_defaults.get("synthetic_data_path", str(SYN_DATA_2D_PATH)))  
    parser.add_argument("--compute_imle_tb", action="store_true", default=config_defaults.get("compute_imle_tb", True))  
    parser.add_argument("--compute_imle_random", action="store_true", default=config_defaults.get("compute_imle_random", True))  
    parser.add_argument("--compute_dlvm_tb", action="store_true", default=config_defaults.get("compute_dlvm_tb", True))  
    parser.add_argument("--compute_imle_dale", action="store_true", default=config_defaults.get("compute_imle_dale", True))  
    parser.add_argument("--session_number", type=int, default=config_defaults.get("session_number", None))  
    parser.add_argument("--max_iterations", type=int, default=config_defaults.get("max_iterations", 1))  
    parser.add_argument("--imle_task_switch_interval", type=int, default=config_defaults.get("imle_task_switch_interval", 12))  
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of worker processes for session-level parallelism")
    parser.add_argument("--stream_csv", action="store_true", help="Write CSV incrementally as sessions complete")
  
    args = parser.parse_args()  
  
    # Setup  
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup file logging
    log_file = args.output_dir / "computation_progress.log"
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Add file handler to both the main logger and module logger
    main_logger = logging.getLogger(__name__)
    main_logger.addHandler(file_handler)
    logger.addHandler(file_handler)
    
    def log_and_print(message, level='INFO'):
        """Log to both console and file"""
        print(message)
        if level == 'INFO':
            main_logger.info(message)
        elif level == 'WARNING':
            main_logger.warning(message)
        elif level == 'ERROR':
            main_logger.error(message)
    
    log_and_print(f"=== Starting DALE Computation ===")
    log_and_print(f"Log file: {log_file}")  
    
    # Streaming CSV setup
    csv_stream_path = None
    method_sessions_count = defaultdict(int)
    method_total_points_count = defaultdict(int)

    def _append_rows(method_name: str, session_id: str, iteration_data: Dict):
        if csv_stream_path is None:
            return
        rows = []
        num_tests = iteration_data.get('num_tests', [])
        klds = iteration_data.get('kld_values', [])
        rmses = iteration_data.get('rmse_values', [])
        for i in range(len(num_tests)):
            rows.append({
                'method': method_name,
                'session_id': session_id,
                'num_tests': int(num_tests[i]) if not (isinstance(num_tests[i], float) and np.isnan(num_tests[i])) else np.nan,
                'kld_value': klds[i] if i < len(klds) else np.nan,
                'rmse_value': rmses[i] if i < len(rmses) else np.nan,
            })
        rows.sort(key=lambda r: (r['method'], r['session_id'], r['num_tests']))
        with open(csv_stream_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['method', 'session_id', 'num_tests', 'kld_value', 'rmse_value'])
            writer.writerows(rows)
        if rows:
            method_sessions_count[method_name] += 1
            method_total_points_count[method_name] += len(rows)

    if args.stream_csv:
        csv_stream_path = args.output_dir / 'dale_comparison_data.csv'
        with open(csv_stream_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['method', 'session_id', 'num_tests', 'kld_value', 'rmse_value'])
            writer.writeheader()
        print(f"Initialized streaming CSV at {csv_stream_path}")
  
    # Print configuration summary
    log_and_print(f"\n=== Configuration Summary ===")
    if pre_args.config:
        log_and_print(f"Using preset: {pre_args.config}")
    def _arg_name(p):
        try:
            return p.name
        except Exception:
            return str(p)
    log_and_print(f"DALE PS0 directory: {_arg_name(args.dale_ps0_dir)}")
    log_and_print(f"DALE PS1 directory: {_arg_name(args.dale_ps1_dir)}")  
    log_and_print(f"DALE PS2 directory: {_arg_name(args.dale_ps2_dir)}")
    log_and_print(f"DALE PS4 directory: {_arg_name(args.dale_ps4_dir)}")
    log_and_print(f"Random directory: {_arg_name(args.random_dir)}")  
    log_and_print(f"Model: {'3D' if args.latent_dim == 3 else '2D'} (latent_dim={args.latent_dim})")
    log_and_print(f"Sessions: {args.session_number or None}")
    log_and_print(f"Max iterations: {args.max_iterations}")
    log_and_print(f"Output directory: {args.output_dir}")
    log_and_print(f"Task switch interval: {args.imle_task_switch_interval}")  
  
    # Load model  
    log_and_print(f"\nLoading model with latent dimension {args.latent_dim}...")  
    model = load_trained_model(latent_dim=args.latent_dim, model_path=args.model_path)  
  
    # Get consistent session list  
    log_and_print(f"\n=== Getting Consistent Session List ===")  
    consistent_session_list = get_consistent_session_list(  
        args.dale_ps0_dir, args.dale_ps1_dir, args.dale_ps2_dir, args.dale_ps4_dir, args.random_dir, session_number=args.session_number, logger=logger  
    )  
  
    if not consistent_session_list:  
        log_and_print("Error: No common sessions found across available directories (need PS0 and RANDOM at minimum). Exiting.", level='ERROR')  
        return   
  
    # Load ground truth  
    ground_truth = load_ground_truth_with_key_matching(args.mle_params_file, consistent_session_list)  
    if not ground_truth:  
        log_and_print("Error: No ground truth data matched. Exiting.", level='ERROR')  
        return  
  
    # Define metrics and parameters  
    metrics = SUMMARIZED_METRICS  
    parameters = {metric: [0] for metric in metrics}  
  
    # Initialize storage for all methods  
    all_method_data: Dict[str, Dict] = {}  
  
    # Load DLVM_DALE PS0 data
    log_and_print(f"\n=== Loading DLVM_DALE_PS0 Time Series Data ===")  
    dale_ps0_iteration_data = load_and_extract_predicted_params(  
        args.dale_ps0_dir, model, ground_truth, metrics, parameters, logger,  
        session_list=consistent_session_list, max_iterations=args.max_iterations,  
        method_label="DLVM_DALE_PS0", workers=args.workers, model_path=args.model_path, latent_dim=args.latent_dim,
        append_rows_fn=_append_rows if args.stream_csv else None  
    )  
    if dale_ps0_iteration_data:  
        all_method_data['DLVM_DALE_PS0'] = dale_ps0_iteration_data

    # Load DLVM_DALE PS1 data (check if directory exists)
    if args.dale_ps1_dir and args.dale_ps1_dir.exists():
        log_and_print(f"\n=== Loading DLVM_DALE_PS1 Time Series Data ===")  
        dale_ps1_iteration_data = load_and_extract_predicted_params(  
            args.dale_ps1_dir, model, ground_truth, metrics, parameters, logger,  
            session_list=consistent_session_list, max_iterations=args.max_iterations,  
            method_label="DLVM_DALE_PS1", workers=args.workers, model_path=args.model_path, latent_dim=args.latent_dim,
            append_rows_fn=_append_rows if args.stream_csv else None  
        )  
        if dale_ps1_iteration_data:  
            all_method_data['DLVM_DALE_PS1'] = dale_ps1_iteration_data
    else:
        log_and_print(f"\n=== Skipping DLVM_DALE_PS1: Directory not found or not specified ===")  

    # Load DLVM_DALE PS2 data (check if directory exists)
    if args.dale_ps2_dir and args.dale_ps2_dir.exists():
        log_and_print(f"\n=== Loading DLVM_DALE_PS2 Time Series Data ===")  
        dale_ps2_iteration_data = load_and_extract_predicted_params(  
            args.dale_ps2_dir, model, ground_truth, metrics, parameters, logger,  
            session_list=consistent_session_list, max_iterations=args.max_iterations,  
            method_label="DLVM_DALE_PS2", workers=args.workers, model_path=args.model_path, latent_dim=args.latent_dim,
            append_rows_fn=_append_rows if args.stream_csv else None  
        )  
        if dale_ps2_iteration_data:  
            all_method_data['DLVM_DALE_PS2'] = dale_ps2_iteration_data
    else:
        log_and_print(f"\n=== Skipping DLVM_DALE_PS2: Directory not found or not specified ===")  

    # Load DLVM_DALE PS4 data (check if directory exists)
    if args.dale_ps4_dir and args.dale_ps4_dir.exists():
        log_and_print(f"\n=== Loading DLVM_DALE_PS4 Time Series Data ===")  
        dale_ps4_iteration_data = load_and_extract_predicted_params(  
            args.dale_ps4_dir, model, ground_truth, metrics, parameters, logger,  
            session_list=consistent_session_list, max_iterations=args.max_iterations,  
            method_label="DLVM_DALE_PS4", workers=args.workers, model_path=args.model_path, latent_dim=args.latent_dim,
            append_rows_fn=_append_rows if args.stream_csv else None  
        )  
        if dale_ps4_iteration_data:  
            all_method_data['DLVM_DALE_PS4'] = dale_ps4_iteration_data
    else:
        log_and_print(f"\n=== Skipping DLVM_DALE_PS4: Directory not found or not specified ===")    
  
    # Load DLVM_RANDOM data (was 'Random')  
    log_and_print(f"\n=== Loading DLVM_RANDOM Time Series Data ===")  
    random_iteration_data = load_and_extract_predicted_params(  
        args.random_dir, model, ground_truth, metrics, parameters, logger,  
        session_list=consistent_session_list, max_iterations=args.max_iterations,  
        method_label="DLVM_RANDOM", workers=args.workers, model_path=args.model_path, latent_dim=args.latent_dim,
        append_rows_fn=_append_rows if args.stream_csv else None  
    )  
    if random_iteration_data:  
        all_method_data['DLVM_RANDOM'] = random_iteration_data  
  
    # Load synthetic data once for all methods that need it
    synthetic_data = None
    methods_requiring_synthetic = [args.compute_imle_tb, args.compute_imle_random, args.compute_dlvm_tb]
    if any(methods_requiring_synthetic):
        log_and_print(f"\n=== Loading Synthetic Data ===")  
        try:  
            synthetic_data = load_simulated_data(args.synthetic_data_path, logger)
            log_and_print(f"✓ Synthetic data loaded successfully")
        except Exception as e:  
            log_and_print(f"✗ Error loading synthetic data: {e}", level='ERROR')  
            logger.error(f"Failed to load synthetic data: {e}", exc_info=True)
            synthetic_data = None

    # IMLE_TB computation
    if args.compute_imle_tb and synthetic_data is not None:  
        log_and_print(f"\n=== Computing IMLE_TB Time Series Data ===")  
        try:
            imle_iteration_data = compute_imle_tb_data(  
                dale_ps0_iteration_data, synthetic_data, ground_truth, model,  
                metrics, parameters, logger, workers=args.workers,  
                imle_task_switch_interval=args.imle_task_switch_interval,
                append_rows_fn=_append_rows if args.stream_csv else None, method_label="IMLE_TB"
            )  
            if imle_iteration_data:  
                all_method_data['IMLE_TB'] = imle_iteration_data
                log_and_print(f"✓ IMLE_TB computation completed: {len(imle_iteration_data)} sessions")
            else:
                log_and_print(f"✗ IMLE_TB computation failed: no data generated", level='WARNING')
        except Exception as e:  
            log_and_print(f"✗ Error computing IMLE_TB: {e}", level='ERROR')  
            logger.error(f"Failed IMLE_TB computation: {e}", exc_info=True)
    elif args.compute_imle_tb and synthetic_data is None:
        log_and_print(f"\n=== Skipping IMLE_TB: Synthetic data not available ===", level='WARNING')

    # IMLE_Random computation  
    if args.compute_imle_random and synthetic_data is not None:  
        print(f"\n=== Computing IMLE_Random Time Series Data ===")  
        try:
            imle_random_iteration_data = compute_imle_random_data(  
                dale_ps0_iteration_data, synthetic_data, ground_truth, model,  
                metrics, parameters, logger, workers=args.workers,
                append_rows_fn=_append_rows if args.stream_csv else None, method_label="IMLE_Random"
            )  
            if imle_random_iteration_data:  
                all_method_data['IMLE_Random'] = imle_random_iteration_data
                log_and_print(f"✓ IMLE_Random computation completed: {len(imle_random_iteration_data)} sessions")
            else:
                log_and_print(f"✗ IMLE_Random computation failed: no data generated", level='WARNING')
        except Exception as e:  
            log_and_print(f"✗ Error computing IMLE_Random: {e}", level='ERROR')  
            logger.error(f"Failed IMLE_Random computation: {e}", exc_info=True)
    elif args.compute_imle_random and synthetic_data is None:
        log_and_print(f"\n=== Skipping IMLE_Random: Synthetic data not available ===", level='WARNING')

    # DLVM_TB computation  
    if args.compute_dlvm_tb and synthetic_data is not None:  
        print(f"\n=== Computing DLVM_TB Time Series Data ===")  
        try:
            dlvm_tb_iteration_data = compute_dlvm_tb_data(  
                dale_ps0_iteration_data, synthetic_data, ground_truth, model,  
                metrics, parameters, logger, workers=args.workers,  
                imle_task_switch_interval=args.imle_task_switch_interval, model_path=args.model_path, latent_dim=args.latent_dim,
                append_rows_fn=_append_rows if args.stream_csv else None, method_label="DLVM_TB"
            )  
            if dlvm_tb_iteration_data:  
                all_method_data['DLVM_TB'] = dlvm_tb_iteration_data
                log_and_print(f"✓ DLVM_TB computation completed: {len(dlvm_tb_iteration_data)} sessions")
            else:
                log_and_print(f"✗ DLVM_TB computation failed: no data generated", level='WARNING')
        except Exception as e:  
            log_and_print(f"✗ Error computing DLVM_TB: {e}", level='ERROR')  
            logger.error(f"Failed DLVM_TB computation: {e}", exc_info=True)
    elif args.compute_dlvm_tb and synthetic_data is None:
        log_and_print(f"\n=== Skipping DLVM_TB: Synthetic data not available ===", level='WARNING')  
  
    # IMLE_DALE for multiple PS variants
    if args.compute_imle_dale:
        log_and_print(f"\n=== Computing IMLE_DALE Time Series Data (PS0/PS1/PS2/PS4) ===")

        # Helper to compute and store by PS label
        def compute_and_store_imle_dale(ps_label: str, dale_dir: Path):
            try:
                log_and_print(f"\n--- IMLE_DALE_{ps_label}: using directory {_arg_name(dale_dir)} ---")
                if not dale_dir or not dale_dir.exists():
                    log_and_print(f"✗ IMLE_DALE_{ps_label}: directory not found — skipping", level='WARNING')
                    return
                data = compute_imle_dale_data(
                    dale_dir, ground_truth, model, metrics, parameters, logger,
                    session_list=consistent_session_list, max_iterations=args.max_iterations, workers=args.workers,
                    append_rows_fn=_append_rows if args.stream_csv else None, method_label=f"IMLE_DALE_{ps_label}"
                )
                if data:
                    all_method_data[f'IMLE_DALE_{ps_label}'] = data
                    log_and_print(f"✓ IMLE_DALE_{ps_label} completed: {len(data)} sessions")
                else:
                    log_and_print(f"✗ IMLE_DALE_{ps_label}: no data generated", level='WARNING')
            except Exception as e:
                log_and_print(f"✗ Error computing IMLE_DALE_{ps_label}: {e}", level='ERROR')
                logger.error(f"Failed to compute IMLE_DALE_{ps_label}: {e}", exc_info=True)

        # Compute for each available PS directory (matching DLVM_DALE PS keys above)
        compute_and_store_imle_dale('PS0', args.dale_ps0_dir)
        compute_and_store_imle_dale('PS1', args.dale_ps1_dir)
        compute_and_store_imle_dale('PS2', args.dale_ps2_dir)
        # Note: repo uses PS4 instead of PS3
        compute_and_store_imle_dale('PS4', args.dale_ps4_dir)
  
    # Prepare metadata  
    metadata = {  
        'dale_ps0_dir': str(args.dale_ps0_dir),  
        'dale_ps1_dir': str(args.dale_ps1_dir),  
        'dale_ps2_dir': str(args.dale_ps2_dir),  
        'dale_ps4_dir': str(args.dale_ps4_dir),  
        'random_dir': str(args.random_dir),  
        'dale_ps0_dir_name': _arg_name(args.dale_ps0_dir),  
        'dale_ps1_dir_name': _arg_name(args.dale_ps1_dir),  
        'dale_ps2_dir_name': _arg_name(args.dale_ps2_dir),  
        'dale_ps4_dir_name': _arg_name(args.dale_ps4_dir),  
        'random_dir_name': _arg_name(args.random_dir),  
        'session_number': args.session_number,  
        'max_iterations': args.max_iterations,  
        'imle_task_switch_interval': args.imle_task_switch_interval,  
        'consistent_sessions': consistent_session_list,  
        'num_sessions': len(consistent_session_list),  
        'methods_computed': list(all_method_data.keys()),  
        'computation_date': pd.Timestamp.now().isoformat(),  
        'dataset': DATASET,  
        'model_path': str(args.model_path),  
        'latent_dim': args.latent_dim  
    }  
  
    # Add per-method statistics  
    if args.stream_csv:
        for method_name, sessions_count in method_sessions_count.items():
            metadata[f'{method_name}_sessions'] = sessions_count
            metadata[f'{method_name}_total_points'] = method_total_points_count.get(method_name, 0)
    else:
        for method_name, method_data in all_method_data.items():  
            metadata[f'{method_name}_sessions'] = len(method_data)  
            total_points = sum(len(session_data['num_tests']) for session_data in method_data.values())  
            metadata[f'{method_name}_total_points'] = total_points  
  
    # Save all data to CSV  
    if args.stream_csv:
        # Streaming write already performed; save metadata only
        metadata_file = args.output_dir / 'dale_comparison_metadata.json'  
        with open(metadata_file, 'w') as f:  
            json.dump(metadata, f, indent=2, default=str)  
        log_and_print("\n=== Computation Complete (streaming) ===")  
        log_and_print(f"Data saved to: {args.output_dir}/dale_comparison_data.csv")  
        log_and_print(f"Metadata saved to: {args.output_dir}/dale_comparison_metadata.json")  
        log_and_print(f"Log saved to: {log_file}")  
        if method_sessions_count:
            log_and_print(f"Total methods computed: {len(method_sessions_count)}")
            log_and_print(f"Methods: {', '.join(sorted(method_sessions_count.keys()))}")
    elif all_method_data:  
        log_and_print(f"\n=== Saving Computed Data ===")  
        df = save_all_data_to_csv(all_method_data, args.output_dir, metadata)  
  
        log_and_print("\n=== Computation Complete ===")  
        log_and_print(f"Data saved to: {args.output_dir}/dale_comparison_data.csv")  
        log_and_print(f"Metadata saved to: {args.output_dir}/dale_comparison_metadata.json")  
        log_and_print(f"Log saved to: {log_file}")  
        log_and_print(f"Total methods computed: {len(all_method_data)}")  
        log_and_print(f"Methods: {', '.join(all_method_data.keys())}")  
    else:  
        log_and_print("\nError: No data computed for any method!", level='ERROR')  
        # Close log file handler
        file_handler.close()
        main_logger.removeHandler(file_handler)
        return 1  

    # Close log file handler
    file_handler.close()
    main_logger.removeHandler(file_handler)
    return 0  
  
  
if __name__ == "__main__":  
    sys.exit(main())  
