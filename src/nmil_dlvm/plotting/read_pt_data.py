import torch
import os
import pandas as pd
import numpy as np
from nmil_dlvm.utils.data_distribution_utils import (
    DATASET,
    DEFAULT_HELDOUT_SET
)

def export_data_details(file_path, default_holdout=None):
    data = torch.load(file_path)

    # Create an empty list to hold the flattened data
    flat_data = []

    # Iterate through the dictionary and flatten the inner dictionaries
    for run, values in data.items():
        if default_holdout and run in default_holdout:
            continue  # Skip the runs in the default_holdout list

        row = {'Run': run}
        for task, metrics in values.items():
            for i, metric in enumerate(metrics):
                row[f'{task}_{i+1}'] = metric.item() if hasattr(metric, 'item') else metric
        flat_data.append(row)

    # Convert the flattened data to a pandas DataFrame
    df = pd.DataFrame(flat_data)

    # Define the folder path
    folder_path = f"./temp/{DATASET}/"

    # Check if the folder exists, if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save the DataFrame to a CSV file within the "temp" folder
    csv_file_path = os.path.join(folder_path, 'refer_mle_params.csv')
    df.to_csv(csv_file_path, index=False)

    return csv_file_path  # Return the saved CSV file path

def modify_csv_first_row(csv_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Modify column names in the first row based on the last character
    new_column_names = []
    for col in df.columns[1:]:
        last_char = col[-1]  # Get the last character of the column name
        if last_char == "1":
            new_col = col[:-1] + "mean"  # Replace "1" with "mean"
        elif last_char == "2":
            new_col = col[:-1] + "std"  # Replace "2" with "std"
        else:
            new_col = col  # Keep the column name unchanged if last character isn't 1 or 2
        new_column_names.append(new_col)

    # Update the DataFrame with modified column names
    df.columns = list(df.columns[:1]) + new_column_names

    # Save the modified DataFrame back to the CSV file
    df.to_csv(csv_path, index=False)


