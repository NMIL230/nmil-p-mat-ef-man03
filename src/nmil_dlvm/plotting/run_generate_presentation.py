import os
import csv

import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
parent_dir = Path(__file__).resolve().parents[3]
if str(parent_dir / "src") not in sys.path:
    sys.path.insert(0, str(parent_dir / "src"))

from nmil_dlvm.plotting.create_plots import parse_csv_and_generate_analysis
from nmil_dlvm.plotting.create_slides import create_presentation
from nmil_dlvm.paths import model_training_analysis_dir

import numpy as np
import pandas as pd

import argparse

from nmil_dlvm.utils.data_distribution_utils import (
    DATASET,
    DEFAULT_HELDOUT_SET,
    SUMMARIZED_METRICS,
    SUMMARIZED_METRICS_MAIN_MOMENTS_LABEL_DICT
)
import matplotlib.pyplot as plt
import subprocess
os.environ["MAX_IMAGE_PIXELS"] = "500000000"  # Set to a suitable value based on your image size

def run_script_with_argument(latent_dim):
    script_path = os.path.join(os.fspath(script_dir), 'create_marginal_fits.py')  # Path to your main Python file
    command = ['python3', script_path, '--latent_dim', str(latent_dim)]
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    print("Script output:")
    print(result.stdout)
    if result.stderr:
        print("Script errors:")
        print(result.stderr)

def identify_csv_file_with_model_id(model_id, csv_files_dir):
    """
    Search through CSV files in a directory to find one containing a specific model ID.

    Args:
        model_id (str): The model ID to search for in the CSV files
        csv_files_dir (str): Directory path containing CSV files to search

    Returns:
        tuple or None: If found, returns (csv_file_path, latent_dim) where csv_file_path is the full path 
                      to the CSV file containing the model_id and latent_dim is the latent dimension value
                      for that model. Returns None if no matching model_id is found.
    """
    for file in os.listdir(csv_files_dir):
        if file.endswith(".csv"):
            with open(os.path.join(csv_files_dir, file), "r") as f:
                reader = csv.reader(f)
                # Get header row
                headers = next(reader)
                
                # Convert headers to indices
                model_id_idx = -1
                file_name_idx = -1
                latent_dim_idx = -1
                for i, header in enumerate(headers):
                    if header == "model_id":
                        model_id_idx = i
                    elif header == "file_name":
                        file_name_idx = i
                    elif header == "latent_dim":
                        latent_dim_idx = i
                
                # Debug: print found indices
                print(f"File: {file}, Headers: {headers}")
                print(f"Indices - model_id: {model_id_idx}, file_name: {file_name_idx}, latent_dim: {latent_dim_idx}")
                
                # Check each data row
                for row in reader:
                    # Ensure we have enough columns in the row
                    if len(row) <= max(model_id_idx, file_name_idx, latent_dim_idx):
                        continue
                    
                    if model_id_idx >= 0 and row[model_id_idx] == model_id:
                        return os.path.join(csv_files_dir, file), row[latent_dim_idx]
                    elif file_name_idx >= 0 and model_id in row[file_name_idx]:
                        return os.path.join(csv_files_dir, file), row[latent_dim_idx]
    return None, None
def main():
    parser = argparse.ArgumentParser(description="Generate PowerPoint slides")
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--model_id", type=str, default="")
    parser.add_argument("--mode", type=str, default="run_from_scratch", choices=["run_from_scratch", "presentation_only", "marginal_only"], help="Mode of operation")
    parser.add_argument("--use_full_primer_range", type=bool, default=False, help="Use full primer range") 
    parser.add_argument("--plot_scatters", type=bool, default=False, help="Plot scatters")
    parser.add_argument("--csv_files_dir", type=str, default=os.fspath(model_training_analysis_dir(DATASET)), help="Path to the CSV file directory")
    args = parser.parse_args()

    if args.model_id:
        runs_date_csv_path, latent_dim = identify_csv_file_with_model_id(args.model_id, args.csv_files_dir)
        if runs_date_csv_path is None:
            raise ValueError(f"Model ID {args.model_id} not found in any of the CSV files in {args.csv_files_dir}")
        else:
            print(f"Model ID {args.model_id} found in {runs_date_csv_path} with latent dimension {latent_dim}")
        args.latent_dim = int(latent_dim)
    else:
        runs_date_csv_path = os.path.join(args.csv_files_dir, "runs_data.csv")

    if args.mode == "run_from_scratch":
        # Run the full pipeline from scratch
        parse_csv_and_generate_analysis(
            csv_path=runs_date_csv_path,
            latent_dim=args.latent_dim,
            base_path=os.fspath(parent_dir),
            model_id=args.model_id if args.model_id else None,
            use_full_primer_range=args.use_full_primer_range,
            plot_scatters=args.plot_scatters
        )

        # Call the function
        create_presentation(
            csv_path=runs_date_csv_path,
            model_notes="",
            latent_dim=args.latent_dim,
        )

        # Generate marginals
        run_script_with_argument(args.latent_dim)

    elif args.mode == "presentation_only":
        # Only generate the presentation
        create_presentation(
            csv_path=runs_date_csv_path,
            model_notes="",
            latent_dim=args.latent_dim,
        )
        # create marginals as well
        run_script_with_argument(args.latent_dim)

    elif args.mode == "marginal_only":
        # Only generate marginals
        run_script_with_argument(args.latent_dim)

if __name__ == "__main__":
    main()
