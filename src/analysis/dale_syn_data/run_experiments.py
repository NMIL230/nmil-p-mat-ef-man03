#!/usr/bin/env python3
"""
Smoke‑test pipeline: synthetic data → DALE active learning
==============================================================

Generates a tiny synthetic dataset with **generate_simulation_data.py** and
immediately feeds it into **ml_run_DALE.py**.  Designed to verify that both
scripts can interoperate end‑to‑end.


"""

from __future__ import annotations
from pathlib import Path

import argparse
import importlib.util
import os
import pathlib
import random
import subprocess
import sys
from typing import List, Dict, Optional

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))            # …/scripts/
repo_root  = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))  # project root

os.chdir(script_dir)

sys.path.append(repo_root)

analysis_dir = os.path.join(repo_root, "analysis", "dlvm_imle_comparison")
scripts_dir  = script_dir 

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(
    cmd: List[str],
    *,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None
) -> None:
    """Echo **cmd**, run it, and abort on non‑zero exit, with optional cwd."""
    print(f"\n>>> cwd={cwd or os.getcwd()}  {' '.join(map(str, cmd))}")
    subprocess.run(cmd, check=True, env=env, cwd=cwd)

def require_pkgs(pkgs: List[str]) -> None:
    missing = [p for p in pkgs if importlib.util.find_spec(p) is None]
    if missing:
        print("[ERROR] Missing packages:", ", ".join(missing))
        sys.exit(1)

def select_random_sessions(n, all_sessions, seed=None):
    """Select n random sessions from the provided session list."""
    if seed is not None:
        random.seed(seed)
    
    # If n is "all" or greater than available sessions, return all
    if n == "all" or n >= len(all_sessions):
        return all_sessions
    
    if n <= 0:
        raise ValueError("n <= 0")
    
    return random.sample(all_sessions, n)

# ---------------------------------------------------------------------------
# Session ID Utilities
# ---------------------------------------------------------------------------

def generate_session_ids(latent_dim: int) -> List[str]:
    """Generate session IDs based on latent dimension.

    For latent_dim == 2 → LD2-001..LD2-088
    For latent_dim == 3 → LD3-001..LD3-088
    """
    if latent_dim not in (1, 2, 3):
        raise ValueError(f"Unsupported latent_dim: {latent_dim}")
    prefix = f"LD{latent_dim}-"
    return [f"{prefix}{i:03d}" for i in range(1, 89)]


def select_eval_sessions(num_eval_sessions, all_sessions: List[str]) -> List[str]:
    """Select evaluation sessions using flexible spec.

    Accepts:
      - "all" → all sessions
      - integer N (or numeric string) → first N sessions (deterministic)
      - list[str] or space-separated string → explicit IDs
    """
    # Case: explicit list already
    if isinstance(num_eval_sessions, list):
        return num_eval_sessions

    # Case: string directive
    if isinstance(num_eval_sessions, str):
        spec = num_eval_sessions.strip()
        if spec.lower() == "all":
            return all_sessions
        if spec.isdigit():
            n = int(spec)
            return all_sessions[: n] if n > 0 else []
        # treat as explicit space-separated list
        return spec.split()

    # Case: integer
    if isinstance(num_eval_sessions, int):
        return all_sessions[: num_eval_sessions] if num_eval_sessions > 0 else []

    raise ValueError(f"Unsupported num_eval_sessions spec: {num_eval_sessions}")

# ---------------------------------------------------------------------------
# Experiment Configurations
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

DATASET_NAME = "COLL10_SIM"

# Directories
SAVED_MODELS_DIR   = REPO_ROOT / "saved_models"
DATA_DIR           = REPO_ROOT / "data"
SYNTHETIC_DATA_DIR = REPO_ROOT / "analysis" / "dlvm_imle_comparison" / "synthetic_data"

# Model paths
MODEL_3D_PATH = SAVED_MODELS_DIR / DATASET_NAME / "heldout_obsmulti" / \
    "variationalNN_relevant_only_latentdim3_beaver-slide-5310.pt"
MODEL_2D_PATH = SAVED_MODELS_DIR / DATASET_NAME / "heldout_obsmulti" / \
    "variationalNN_relevant_only_latentdim2_mongoose-dive-7464.pt"

# MLE params file paths
MLE_PARAMS_3D_PATH = DATA_DIR / DATASET_NAME / "D3_synthetic_ground_truth_parameters_beaver-slide-5310.pt"
MLE_PARAMS_2D_PATH = DATA_DIR / DATASET_NAME / "D2_synthetic_ground_truth_parameters_mongoose-dive-7464.pt"

# Synthetic data file paths
SYNTHETIC_DATA_2D_PATH = SYNTHETIC_DATA_DIR / DATASET_NAME / "D2_all_synthetic_data_N240.pt"
SYNTHETIC_DATA_3D_PATH = SYNTHETIC_DATA_DIR / DATASET_NAME / "D3_all_synthetic_data_N240.pt"
# Model paths - adjust these to your actual model files
MODEL_3D_BEAVER = str(REPO_ROOT / 'saved_models/COLL10_SIM/heldout_obsmulti/variationalNN_relevant_only_latentdim3_beaver-slide-5310.pt')
MODEL_2D_MONGOOSE = './saved_models/COLL10_SIM/heldout_obsmulti/variationalNN_relevant_only_latentdim2_mongoose-dive-7464.pt'

EXPERIMENTS = [
    #  {
    #     "name": "exp1_dale_3d_coll10_ps",
    #     "dataset": "COLL10",
    #     "N": 100,
    #     "run_mode": "run",  # "run" or "debug"
    #     "model_path": MODEL_3D_BEAVER,
    #     "test_budget": 100,
    #     "num_restarts": 1,
    #     "latent_dim": 3,
    #     "held_out_sessions": [],  # Empty list means no held-out sessions
    #     "num_eval_sessions": 5,  # Use "all" to use all available sessions
    #     "session_seed": 42,
    #     "use_synthetic_data": True,
    #     "random_baseline": False,
    #     "enable_primer_sequence": True,  # Enable primer sequence generation
    #     "primer_sequence_task_repetitions": 4,  # Number of repetitions for primer
    # },
     {
        "name": "exp1",
        "dataset": "COLL10_SIM",
        "N": 240,
        "run_mode": "debug",  # "run" or "debug"
        "model_path": MODEL_3D_PATH,
        "mle_params_file": MLE_PARAMS_3D_PATH,
        "synthetic_data_file": SYNTHETIC_DATA_3D_PATH,
        "test_budget": 5,
        "num_restarts": 1,
        "latent_dim": 3,
        "held_out_sessions": [],  # Empty list means no held-out sessions
        "num_eval_sessions": "all",  # Use "all" to use all available sessions
        "session_seed": 42,
        "use_synthetic_data": True,
        "random_baseline": False,
        "enable_primer_sequence": True,  # Enable primer sequence generation
        "primer_sequence_task_repetitions": 4,  # Number of repetitions for primer
    },
   
    
   
  
    # Add more experiments as needed...
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(config: Dict, dale_script: str, analysis_dir: str) -> None:
    """Run a single experiment with the given configuration."""
    
    print("\n" + "="*80)
    print(f"Running Experiment: {config['name']}")
    print("="*80)
    
    # Require explicit file paths in config; do not synthesize
    required_paths = {
        "synthetic_data_file": "Synthetic data",
        "mle_params_file": "MLE params",
        "model_path": "Model",
    }
    for key, label in required_paths.items():
        path_val = config.get(key)
        if not path_val:
            print(f"[WARNING] Missing required config key '{key}' for: {label}")
            print(f"Skipping experiment: {config['name']}")
            return
        if not os.path.exists(os.fspath(path_val)):
            print(f"[WARNING] {label} file missing: {path_val}")
            print(f"Skipping experiment: {config['name']}")
            return
    
    # Build session list based on latent dimension and filter out held-out
    all_sessions = generate_session_ids(config["latent_dim"])
    available_sessions = [sid for sid in all_sessions if sid not in config["held_out_sessions"]]

    # Select evaluation sessions per flexible spec
    eval_sessions = select_eval_sessions(config["num_eval_sessions"], available_sessions)
    
    print(f"\nExperiment Configuration:")
    print(f"  Dataset: {config['dataset']}")
    print(f"  N: {config['N']}")
    print(f"  Run Mode: {config['run_mode']}")
    print(f"  Test Budget: {config['test_budget']}")
    print(f"  Num Restarts: {config['num_restarts']}")
    print(f"  Latent Dim: {config['latent_dim']}")
    print(f"  Held Out Sessions: {config['held_out_sessions']}")
    print(f"  Num Eval Sessions: {config['num_eval_sessions']}")
    print(f"  Random Baseline: {config['random_baseline']}")
    print(f"  Use Grid Search: {config.get('use_grid_search', False)}")
    print(f"  Selected Eval Sessions: {eval_sessions[:3]}... (showing first 3)")
    

    # Build DALE command
    dale_cmd = [
        sys.executable, dale_script,
        "--run_mode", config["run_mode"],
        "--test_budget", str(config["test_budget"]),
        "--use_synthetic_data", str(config["use_synthetic_data"]),
        "--num_restarts", str(config["num_restarts"]),
        "--latent_dim", str(config["latent_dim"]),
        "--enable_primer_sequence", str(config['enable_primer_sequence']),
        "--primer_sequence_task_repetitions", str(config['primer_sequence_task_repetitions']),
        "--synthetic_data_file", str(config['synthetic_data_file']),
        "--mle_params_file", str(config['mle_params_file']),
    ]
    
    # Add model path if specified
    if "model_path" in config:
        dale_cmd.extend(["--model_path", str(config["model_path"])])
    
    # Add optional parameters
    if config.get("random_baseline", True):
        dale_cmd.extend(["--random_baseline", "True"])
    
    # if config.get("use_grid_search", False):
    #     dale_cmd.extend(["--use_grid_search", "True"])
    
    # Add held-out sessions (only if there are any)
    if config.get("held_out_sessions", []):
        dale_cmd.append("--trained_model_held_out_ids")
        dale_cmd.extend(config["held_out_sessions"])
    
    # Add evaluation sessions (only if there are any)
    if eval_sessions:
        dale_cmd.append("--eval_test_session_ids")
        dale_cmd.extend(eval_sessions)
    
    # Set up environment
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([repo_root, env.get("PYTHONPATH", "")])
    
    # Run the experiment
    try:
        run(
            dale_cmd,
            env=env,
            cwd=os.path.dirname(dale_script)
        )
        print(f"\n✓ Experiment '{config['name']}' completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Experiment '{config['name']}' failed with error: {e}")
        # Continue with next experiment instead of stopping
        return

def main() -> None:
    ap = argparse.ArgumentParser(description="IMLE → DALE smoke test with multiple experiments")
    ap.add_argument("--experiments", nargs="+", type=int, 
                   help="Indices of experiments to run (0-based). If not specified, runs all.")
    ap.add_argument("--list", action="store_true", 
                   help="List all available experiments without running them")
    args = ap.parse_args()
    
    # List experiments if requested
    if args.list:
        print("\nAvailable Experiments:")
        for i, exp in enumerate(EXPERIMENTS):
            print(f"  [{i}] {exp['name']}")
            print(f"      Dataset: {exp['dataset']}, N: {exp['N']}, Mode: {exp['run_mode']}")
        return
    
    # Check DALE script exists
    dale_script = os.path.join(scripts_dir, "run_DALE_syn_data.py")
    if not os.path.exists(dale_script):
        sys.exit(f"DALE script not found: {dale_script}")
    
    # Check required packages
    require_pkgs(["gpytorch", "pyro"])
    
    # Determine which experiments to run
    if args.experiments:
        experiments_to_run = []
        for idx in args.experiments:
            if 0 <= idx < len(EXPERIMENTS):
                experiments_to_run.append(EXPERIMENTS[idx])
            else:
                print(f"[WARNING] Invalid experiment index: {idx}")
    else:
        experiments_to_run = EXPERIMENTS
    
    print(f"\nWill run {len(experiments_to_run)} experiment(s)")
    
    # Run selected experiments
    for config in experiments_to_run:
        run_experiment(config, dale_script, analysis_dir)
    
    print("\n" + "="*80)
    print("All experiments completed!")
    print("="*80)

if __name__ == "__main__":
    main()
