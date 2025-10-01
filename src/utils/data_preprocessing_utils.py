import sys
# Add parent directories to sys.path to allow importing modules from those directories
sys.path.append("../")
sys.path.append("../../")
import pandas as pd
import numpy as np
import json
import glob
import warnings
import pdb
import random
import os
import argparse
from utils.set_seed import set_seed # Custom function to set the random seed for reproducibility

warnings.filterwarnings("ignore")
import torch

def generate_full_matrix(DATASET, RELEVANT_METRICS, normalize_times=False):
    """
    Generates a full data matrix for the specified dataset and relevant metrics.

    Args:
        DATASET (str): The name of the dataset being used.
        RELEVANT_METRICS (list): List of relevant metrics to include in the data matrix.
        normalize_times (bool): Whether to normalize the times or not. Defaults to False.

    Returns:
        list: A list of all participant IDs included in the generated data matrix.
    """
    try:
        # Determine the home path of the project by getting the directory of the current script and moving up one level
        home_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
        
        # Load the dataset into a pandas DataFrame
        data = pd.read_csv(os.path.join(home_path, f"data/{DATASET}/all_data_time_not_normed.csv"))

        # Get a list of unique user sessions (participant IDs) from the dataset
        sessions = data["user_session"].unique()

        # Initialize a list to store all participant IDs
        ALL_PARTICIPANT_IDS = []

        # Initialize a list to store the data matrices for all sessions
        full_set = []

        # Set the maximum number of observations per metric to ensure consistent tensor shape
        max_obs_per_metric = 60  # Ensures that the tensor will have a consistent shape for all sessions
        count = 0 #Initializes a counter for the number of complete sessions processed
                
        # Define a list of required metrics (specific to the 'COLL10_SIM' dataset)
        REUIRED_METRICS = ["Stroop_reaction_time","Countermanding_reaction_time","PasatPlus_correctly_answered",    
                       "RunningSpan_correct_w_len_2","RunningSpan_correct_w_len_3",    "D2_hit_accuracy",
                       "CorsiComplex_correct_w_len_2","SimpleSpan_correct_w_len_2",] # TODO REMOVE THIS CONSTRAINT
        

        # Loop over each user session (participant)
        for session in sessions:
            # Get all data for the current session
            session_data = data[data["user_session"] == session]
            # Initialize a list to store observations for each metric in the current session
            session_obs = []
            incomplete = False # Flag to indicate if the session is incomplete
            

            # Loop over each relevant metric
            for metric in RELEVANT_METRICS:
                # Get the observations (results) for the current metric in the current session
                metric_obs = list(session_data[session_data["metric"] == metric]["result"])
                # For the 'COLL10_SIM' dataset, check if required metrics have data
                if DATASET =="COLL10_SIM" and (len(metric_obs) ==0) and (metric in REUIRED_METRICS): # TODO remove must have data for 
                    # If a required metric has no data, mark the session as incomplete and break the loop
                    incomplete = True 
                    break;
                    
                # Ensure that the number of observations does not exceed the maximum allowed
                if len(metric_obs) > max_obs_per_metric:  # Drop extra observations
                    # If there are more observations than the maximum, truncate the list
                    metric_obs = metric_obs[:max_obs_per_metric]
                elif len(metric_obs) < max_obs_per_metric:  # Fill up with NaNs
                    # If there are fewer observations, pad the list with NaNs to reach the maximum length
                    metric_obs += [float("nan")] * (max_obs_per_metric - len(metric_obs))
                # Append the observations for the current metric to the session's observations
                session_obs.append(metric_obs)
            
            if not incomplete:  # Ignore incomplete sessions
                # Convert the session observations to a NumPy array and transpose it
                # The shape becomes (max_obs_per_metric, number of metrics)
                session_obs = np.array(session_obs).T # Transpose to align observations correctly
                 # Append the session's data to the full dataset
                full_set.append(session_obs)
                # Append the participant ID to the list of all participant IDs
                ALL_PARTICIPANT_IDS.append(session)  # Extract the participant ID
                # Increment the counter of complete sessions
                count += 1
        
        # Convert the full dataset to a NumPy array and then to a PyTorch tensor
        full_data_matrix = torch.tensor(np.array(full_set))
        filename = home_path
        # Determine the label to use in the filename based on whether times are normalized
        normalize_label = "" if normalize_times else "_not_normed"
        
        # Print the shape of the full data matrix for confirmation
        print(f"All {DATASET} data", full_data_matrix.shape)
        
        # Save the full data matrix tensor to a file in the specified directory
        torch.save(full_data_matrix, os.path.join(filename, f"data/{DATASET}/full_data_matrix{normalize_label}.pt"))
        print(f'The generated file "full_data_matrix{normalize_label}.pt" is stored at {filename}')
        
        # Save the participant IDs to a CSV file for future reference
        pd.DataFrame({'ids': ALL_PARTICIPANT_IDS}).to_csv(os.path.join(filename, f"data/{DATASET}/participant_ids{normalize_label}.csv"))
        
        # Randomly sample 10% of the participant IDs
        sample_size = int(np.floor(0.1 * len(ALL_PARTICIPANT_IDS)))
        random_sample = random.sample(ALL_PARTICIPANT_IDS, sample_size)
        print("Random test set IDs", random_sample)
        
        # Return the list of all participant IDs processed
        return ALL_PARTICIPANT_IDS

    except Exception as e:
        # If an error occurs during processing, print an error message with details
        print("Error occurred: Could not generate data matrix needed for further processing -- see details below")
        print(f"{e}")
        sys.exit(1)
