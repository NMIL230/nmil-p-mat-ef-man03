import os, sys
from pathlib import Path

# from configs.datasets.generate_configs_file import SUMMARIZED_METRICS_METRIC_TYPES
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd
import matplotlib.pyplot as plt
import math
import torch

from nmil_dlvm.paths import ANALYSIS_ARTIFACTS_ROOT, SAVED_MODELS_ROOT, data_dir, ensure_dir
from nmil_dlvm.utils.grid_search_utils import predict_parameters_from_data

from nmil_dlvm.utils.data_distribution_utils import ALL_METRICS_MOMENTS_LABEL_DICT, SUMMARIZED_METRICS_METRIC_TYPES, VIS_ORDER_PREFERENCE_METRICS_ALL, load_trained_model

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ensure_dir(ANALYSIS_ARTIFACTS_ROOT / "dlvm_adhoc_analysis")
PLOTS_DIR = ensure_dir(OUTPUT_DIR / "plots")

data = torch.load(SCRIPT_DIR / "synthetic_dlvm_params_D3_N100.pt")
coll10_data = torch.load(data_dir("COLL10_SIM") / "all_data-best_mle_params_mpf100.pt")
# Ensure all items in the value lists are not tensors
def convert_tensors(data):
    for key in data:
        for sub_key in data[key]:
            data[key][sub_key] = [item.item() if isinstance(item, torch.Tensor) else item for item in data[key][sub_key]]
    return data

data = convert_tensors(data)
coll10_data = convert_tensors(coll10_data)

def process_data(data, output_csv):
    # Convert the nested dictionary into a DataFrame
    df = pd.DataFrame.from_dict(data, orient='index')

    # Flatten the lists in the dictionary to create columns for key_param1 and key_param2
    expanded_df = pd.DataFrame(
        {f"{col}_{param}": [val[i] if i < len(val) else None for val in df[col]]
         for col in df.columns for i, param in enumerate(["param1", "param2"])}
    )

    # Combine the expanded data with the original index
    expanded_df.index = df.index
    # Remove columns that are completely empty (all NaN values)
    expanded_df = expanded_df.dropna(axis=1, how='all')

    # Save the result to a CSV file
    expanded_df.to_csv(output_csv, index=True)
    
    return expanded_df

# Process data and coll10_data
expanded_df = process_data(data, OUTPUT_DIR / "dlvm_D3_params_N100.csv")
expanded_coll10_df = process_data(coll10_data, OUTPUT_DIR / "coll10_params.csv")

def plot_histograms(expanded_df, expanded_coll10_df, output_pdf, title = "DLVM Predicted vs. Ground truth (COLL10) parameter distributions"):
    # Determine the grid size (rows x cols)
    n_columns = len(VIS_ORDER_PREFERENCE_METRICS_ALL)
    n_columns = math.ceil(n_columns / 3)

    # Create subplots
    fig, axes = plt.subplots(3, n_columns, figsize=(5 * n_columns,5*3))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    # Plot each column's histogram in a separate subplot with two datasets
    # import pdb; pdb.set_trace()
    for idx, column in enumerate(VIS_ORDER_PREFERENCE_METRICS_ALL):
        
        ax = axes[idx]
        dlvm_data = expanded_df[column].dropna()
        coll10_data = expanded_coll10_df[column].dropna() if column in expanded_coll10_df.columns else pd.Series()

        ax.hist(dlvm_data, bins=15, color='blue', edgecolor='black', alpha=0.5, label=f'DLVM predictions (mean={dlvm_data.mean():.2f})')
        if not coll10_data.empty:
            ax.hist(coll10_data, bins=15, color='red', edgecolor='black', alpha=0.5, label=f'Ground truth (mean={coll10_data.mean():.2f})')
        
        # import pdb; pdb.set_trace()
        metric = "_".join(column.split("_")[:-1])

        metric_type = SUMMARIZED_METRICS_METRIC_TYPES.get(metric, column)
        if metric_type == "binary":
            axis_label = "Probability"
        elif metric_type == "timing":
            axis_label = "Log Time (ms)"
        elif metric_type == "binarySpan":
            axis_label = "# of items"
        else:
            axis_label = "Value"
        # import pdb; pdb.set_trace()
        ax.set_title(f"{ALL_METRICS_MOMENTS_LABEL_DICT.get(column, column)}")
        ax.set_xlabel(f"{axis_label}")
        ax.set_ylabel("Frequency")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend()

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_pdf)

# Call the function to plot histograms
plot_histograms(expanded_df, expanded_coll10_df, PLOTS_DIR / "before_fix_dlvm_vs_coll10_params.pdf", title="DLVM (honest-frost-2316) Predicted vs. Ground truth (COLL10) parameter distributions - Before bug fix")

# Load csv file custom_df_N100.csv
test_data = pd.read_csv(SCRIPT_DIR / "custom_df_N100.csv")[["user_session", "metric", "result"]]
#id = revived-bird-381
# honest-frost-2316
dlvm_model = load_trained_model(
    latent_dim=3,
    model_path=SAVED_MODELS_ROOT / "COLL10_SIM" / "heldout_obsmulti" / "variationalNN_relevant_only_latentdim3_revived-bird-381.pt",
)
# Print the length of each list in the dictionary for each user_session
# import pdb; pdb.set_trace()

# Create a dictionary of dictionaries for each user_session
user_session_dict = {}
parameter_dict = {}
for user_session, group in test_data.groupby('user_session'):
    user_session_dict[user_session] = group.groupby('metric')['result'].apply(list).to_dict()
    # Predict the parameters for each user_session
    predicted_params = predict_parameters_from_data(user_session_dict[user_session], dlvm_model)
    parameter_dict[user_session.replace("_sim1","")] = predicted_params

import pdb; pdb.set_trace()
parameter_df = process_data(parameter_dict, OUTPUT_DIR / "new_honest_frost_pred_params_N100.csv")

plot_histograms(parameter_df, expanded_coll10_df, PLOTS_DIR / "after_fix_dlvm_vs_coll10_params.pdf", title="DLVM (honest-frost-2316) Predicted vs. Ground truth parameter distributions - After bug fix")


