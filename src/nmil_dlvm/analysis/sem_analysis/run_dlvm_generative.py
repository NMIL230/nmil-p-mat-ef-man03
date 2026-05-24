
import os
import sys
from pathlib import Path
# add absolute path to utils folder
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
# from nmil_dlvm.utils.data_distribution_utils import load_data
from nmil_dlvm.paths import ANALYSIS_ARTIFACTS_ROOT, SAVED_MODELS_ROOT, ensure_dir
from nmil_dlvm.utils.active_learning_utils import load_trained_model
from nmil_dlvm.utils.data_distribution_utils import CURR_METRICS_DICT
from nmil_dlvm.utils.grid_search_utils import compute_predictions_fom_latent_points, generate_grid

import numpy as np
import torch
import pandas as pd

import argparse
import pandas as pd

GENERATED_DATA_DIR = ensure_dir(ANALYSIS_ARTIFACTS_ROOT / "sem_analysis" / "generated_data")

def main(args):
    latent_dim = args.latent_dim
    model_id = args.model_id
    num_samples = args.num_samples

    model_path = SAVED_MODELS_ROOT / "COLL10_SIM" / "heldout_obsmulti" / f"variationalNN_relevant_only_latentdim{latent_dim}_{model_id}.pt"
    model = load_trained_model(latent_dim=latent_dim)

    # generate grid of latent points
    grid_points = generate_grid(model, num_points=num_samples)

    # predict the distributional parameters for each latent point
    predicted_parameters = compute_predictions_fom_latent_points(grid_points, model).squeeze(0).detach().cpu().numpy()
    
    idx_to_parameter = {i: None for i in range(predicted_parameters.shape[1])}
    
    # create headers for the results
    for metric in CURR_METRICS_DICT.keys():
        fidxs = CURR_METRICS_DICT[metric]["f_idxs"]  # indices of the model output relevant for the metric
        summary_metric_label = CURR_METRICS_DICT[metric]["summary_metric_label"]

        for i, fidx in enumerate(fidxs):
            idx_to_parameter[fidx] = f"{summary_metric_label}_param{i+1}"

    # extract the parameter headers from the dictionary ensuring they are in the correct order
    parameter_headers = [idx_to_parameter[i] for i in range(predicted_parameters.shape[1])]
    
    # create the results data frame using the headers and predicted parameters
    results = pd.DataFrame(predicted_parameters, columns=parameter_headers)

    # save the results to a CSV file in the generated_data folder
    results.to_csv(GENERATED_DATA_DIR / f"ld{latent_dim}_{model_id}_latent_space_preds.csv", index=False)
    print(parameter_headers)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate latent space results.")
    parser.add_argument("--latent_dim", type=int, default=3, help="Dimension of the latent space")
    parser.add_argument("--model_id", type=str, default="fragrant-lion-356", help="Model ID")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to generate in each dimension")

    # honest-frost-2316
    # hopeful-deluge-359
    # silvery-snowball-283
    # Get arguments
    args = parser.parse_args()

    main(args)
