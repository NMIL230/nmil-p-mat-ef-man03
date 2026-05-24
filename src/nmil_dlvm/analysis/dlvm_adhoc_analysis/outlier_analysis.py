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

from nmil_dlvm.paths import ANALYSIS_ARTIFACTS_ROOT, data_dir, ensure_dir
from nmil_dlvm.utils.grid_search_utils import predict_parameters_from_data

from nmil_dlvm.utils.data_distribution_utils import ALL_METRICS_MOMENTS_LABEL_DICT, SUMMARIZED_METRICS_METRIC_TYPES, VIS_ORDER_PREFERENCE_METRICS_ALL, DATASET

OUTPUT_DIR = ensure_dir(ANALYSIS_ARTIFACTS_ROOT / "dlvm_adhoc_analysis")
PLOTS_DIR = ensure_dir(OUTPUT_DIR / "plots")

coll10_data = torch.load(data_dir(DATASET) / "all_data-best_mle_params_mpf100.pt")
# Ensure all items in the value lists are not tensors
def convert_tensors(data):
    for key in data:
        for sub_key in data[key]:
            data[key][sub_key] = [item.item() if isinstance(item, torch.Tensor) else item for item in data[key][sub_key]]
    return data

coll10_data = convert_tensors(coll10_data)

def process_data(data, output_csv):
    # Convert the nested dictionary into a DataFrame
    df = pd.DataFrame.from_dict(data, orient='index')

    # Flatten the lists in the dictionary to create columns for key_param1 and key_param2
    expanded_df = pd.DataFrame()
    for col in df.columns:
        print(col)
        # import pdb; pdb.set_trace()
        for i, param in enumerate(["param1", "param2"]):
            expanded_df[f"{col}_{param}"] = [val[i] if isinstance(val, list) and i < len(val) else float('nan') for val in df[col]]

    # Combine the expanded data with the original index
    expanded_df.index = df.index
    # Remove columns that are completely empty (all NaN values)
    expanded_df = expanded_df.dropna(axis=1, how='all')

    # Save the result to a CSV file
    expanded_df.to_csv(output_csv, index=True)
    
    return expanded_df

# import pdb; pdb.set_trace()
expanded_coll10_df = process_data(coll10_data, OUTPUT_DIR / f"{DATASET}_params.csv")

def plot_histograms(dfs, output_pdf, title="Parameter Distributions"):
    # Determine the grid size (rows x cols)
    n_columns = len(VIS_ORDER_PREFERENCE_METRICS_ALL)
    n_columns = math.ceil(n_columns / 3)

    # Create subplots
    fig, axes = plt.subplots(3, n_columns, figsize=(5 * n_columns, 5 * 3))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    # Plot each column's histogram in a separate subplot with multiple datasets
    for idx, column in enumerate(VIS_ORDER_PREFERENCE_METRICS_ALL):
        ax = axes[idx]

        for df in dfs:
            if column in df.columns:
                data = df[column].dropna()
                ax.hist(data, bins=15, alpha=0.5, edgecolor='black', label=f'{df.name} (mean={data.mean():.2f})')

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

        ax.set_title(f"{ALL_METRICS_MOMENTS_LABEL_DICT.get(column, column)}")
        ax.set_xlabel(f"{axis_label}")
        ax.set_ylabel("Frequency")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend()

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_pdf)

# Example usage:
expanded_coll10_df.name = f"Original {DATASET} (n={len(expanded_coll10_df)})"
dfs = [expanded_coll10_df]
for i, df in enumerate(dfs):
    df.name = f"Dataset  {i+1} (n={len(df)})"
plot_histograms(dfs, PLOTS_DIR / f"original_{DATASET}.pdf")

df = pd.read_csv(OUTPUT_DIR / f"{DATASET}_params.csv", index_col=0)
# compute the mean absolute deviation mean
def remove_mad_outliers(df):
    numeric_cols = df.select_dtypes(include=["number"])  # Select only numeric columns
    outlier_indices = set()
    
    for col in numeric_cols.columns:
        if col.endswith("_param1"):
            mean_col = numeric_cols[col].mean()
            mad_col = (numeric_cols[col] - mean_col).abs().mean()
            threshold = 4 * mad_col
            
            outliers = df[abs(df[col] - mean_col) > threshold].index
            outlier_indices.update(outliers)
            print(f"Outliers in {col}: {len(outliers)}, mean={mean_col}, mad={mad_col}, threshold={threshold}, indices={outliers}")
    
    cleaned_df = df.drop(index=outlier_indices).reset_index(drop=True)
    return list(outlier_indices), cleaned_df
outlier_indices, cleaned_df = remove_mad_outliers(df)

cleaned_df.name = f"Cleaned {DATASET} (n={len(cleaned_df)})"
dfs = [cleaned_df]
plot_histograms(dfs, PLOTS_DIR / f"cleaned_{DATASET}.pdf")
#write to txt file the outlier indices
with open(OUTPUT_DIR / "outliers.txt", "w") as f:
    for index in outlier_indices:
        f.write(f"{index} ")
#identfiy sessions
import pdb; pdb.set_trace()

