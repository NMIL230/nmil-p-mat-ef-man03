import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
import pandas as pd
import torch
from nmil_dlvm.utils.variational_NN import variationalNN
from nmil_dlvm.utils.data_distribution_utils import (
    metrics_dict,
    activation_dict,
    dist_dict,
    prepare_data,
    RELEVANT_METRICS,
    DATASET,
)
from nmil_dlvm.utils.active_learning_utils import (
    get_data_list_from_mle_data_generator
)
from nmil_dlvm.utils.grid_search_utils import generate_grid
from nmil_dlvm.plotting.create_plots import (
    create_histograms,
    create_logprob_barplot,
    load_2d_models_and_plot_slices
)
import random
import numpy as np
import os
import re
import pdb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
import ipywidgets as widgets
from IPython.display import display
import pickle

from datetime import datetime
from PIL import Image

from numpy.linalg import norm
import ast
from matplotlib.lines import Line2D
from datetime import date

mean_log_probs = torch.load("./data/mean_log_probabilities.pt")
std_log_probs = torch.load("./data/std_log_probabilities.pt")

def parse_csv_and_generate_analysis(csv_path, time_stamp=None, latent_dim=None, held_out_session_ids=None, lr=None, kld_factor=None, model_path=None):
    file_names = []

    if model_path is None:
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_path)

        # Filter the DataFrame based on the provided parameters
        if time_stamp is not None:
            df = df[df['time_stamp'] == time_stamp]
        if latent_dim is not None:
            df = df[df['latent_dim'] == latent_dim]
        if held_out_session_ids is not None:
            df = df[df['held_out_session_ids'].isin(held_out_session_ids)]
        if lr is not None:
            df = df[df['lr'] == lr]
        if kld_factor is not None:
            df = df[df['kld_factor'] == kld_factor]

        # Get the file names
        file_names = df['file_name'].tolist()
    else:
        # If a specific model_path is provided, extract the file name and add it to the list
        file_names.append(os.path.basename(model_path))
    
    output_path = "./models"
    specific_holdout = ['406run9', '307run0', '307run5', '301run2', '405run1', '411run4', '404run5', '305run3', '411run2', '404run8']
    N = 90

    for file_name in file_names:
        model_path = os.fspath(
            REPO_ROOT / "artifacts" / "models" / DATASET / "heldout_obsmulti" / file_name.lstrip("/")
        )
        
        analysis, min_value, max_value = generate_analysis_csv(
            model_path,
            latent_dim = latent_dim,
            N = N,
            output_path = output_path,
            specific_holdout = specific_holdout)

        figure_path = create_histograms(
            model_path = model_path,
            output_path = output_path,
            min_value = min_value, 
            max_value = max_value,
            plot_meu_z = True,
            plot_session_id = True,
            sort_by_diff = True
        )

        for i in range(0,3):
            sorts = ['best_logprob', 'meu_z_logprob','meu_z_mag']
            create_logprob_barplot(
                model_path = model_path,
                output_path = output_path,
                sort_by = sorts[i], 
                show_session_ids = True
            )

        if latent_dim == 2:
            load_2d_models_and_plot_slices(model_path=model_path,
                                        output_path = output_path,
                                        latent_dim = latent_dim,
                                        N = N)


def generate_analysis_csv(model_path, latent_dim, N, output_path, specific_holdout=None):
    if specific_holdout is None:
        specific_holdout = ['']

    # Extract the model ID
    filename = os.path.basename(model_path)  # Get the filename from the path
    model_id = filename.rsplit("_", 1)[-1].split(".")[0]  # Extract the model ID

    # Load dale model using the extracted model ID
    dale_model = load_trained_model(model_path=model_path, latent_dim=latent_dim, N=N)

    # Define output directory
    output_dir = f'{output_path}/{model_id}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the CSV file
    participant_ids = pd.read_csv(f'./data/participant_ids_not_normed.csv')

    # Create a copy and add new columns
    analysis = participant_ids.copy()
    analysis.columns = ['index', 'ids']  # Assigning column names
    analysis['session_type'] = 'train'  # Initializing to 'train'
    analysis['best_logprob'] = np.nan  # Initializing to NaN
    analysis['meu_z'] = np.nan  # Initializing to NaN
    analysis['best_latent_pos'] = np.nan  # Initializing to NaN
    analysis['meu_z_logprob'] = np.nan  # Initializing to NaN
    analysis['logprob_mean'] = np.nan  # Initializing to NaN
    analysis['logprob_variance'] = np.nan  # Initializing to NaN

    # Define holdout sessions
    default_holdout = ["304run0", "304run4", "306run1"]

    # Update session_type column
    analysis.loc[analysis['ids'].isin(specific_holdout), 'session_type'] = 'test'
    analysis.loc[analysis['ids'].isin(default_holdout), 'session_type'] = 'outlier'

    # Initialize the train index
    train_index = 0

    # Initialize maximum value tracker
    max_value = float('-inf')
    min_value = float('inf')
    # Initialize a dictionary to hold filtered_values for each session
    filtered_values_dict = {}

    # Loop through the rows of the analysis DataFrame
    for index, row in analysis.iterrows():
        session_id = row['ids']
        data_dict, all_metrics, _ = prepare_data(heldout_obs_ids =[session_id], get_heldout_instead = True)

        # Place these lines within the loop
        update_w_data = {}
        ORACLE_update_w_data = {}
        all_metrics = RELEVANT_METRICS

        for metric in all_metrics:
            update_w_data[metric] = []
            ORACLE_update_w_data[metric] = []

        for metric in all_metrics:
            data, _, counts, _ = data_dict[metric]
            data = data[torch.logical_not(torch.isnan(data))].tolist()

            if ("binary" in metrics_dict[metric]["type"]) and len(data) > 0:
                data_list = []
                for _ in range(int(data[0])):
                    data_list.append(1.0)
                for _ in range(int(counts[0] - data[0])):
                    data_list.append(0.0)
                data = data_list
            ORACLE_update_w_data[metric] += data

        remaining_data = {}

        for metric in RELEVANT_METRICS:
            remaining_data[metric] = ORACLE_update_w_data[metric]

        lpl_result, meu_z = run_grid_search(model=dale_model, update_w_data=ORACLE_update_w_data, num_points=150)

        meu_z_cpu = meu_z.cpu()
        lpl_result_cpu = lpl_result.cpu()

        # Define the percentage to keep
        percentile_to_keep = 95  # keep smallest n% of lpl_result values

        # Calculate the cutoff for the smallest n% of lpl_result values
        cutoff = np.percentile(lpl_result_cpu.numpy(), percentile_to_keep)

        # Create masks for the values and coordinates that meet the cutoff
        values_mask = lpl_result_cpu < cutoff
        meu_z_cpu_flattened = meu_z_cpu.view(meu_z_cpu.shape[0], -1)
        coords_mask = values_mask.unsqueeze(-1).expand_as(meu_z_cpu_flattened)

        # Apply the masks
        filtered_values = lpl_result_cpu[values_mask]
        filtered_coords = meu_z_cpu_flattened[coords_mask].reshape(-1, meu_z_cpu.shape[-1])

        # Find the minimum value in the scattered set of dots and corresponding coordinates
        min_value_index = np.argmin(filtered_values)
        min_value_coord = filtered_coords[min_value_index]

        # Check and update maximum value if needed
        max_value_session = torch.max(filtered_values).item()
        if max_value_session > max_value:
            max_value = max_value_session

        # Check and update minimum value if needed
        min_value_session = torch.min(filtered_values).item()
        if min_value_session < min_value:
            min_value = min_value_session

        # Store filtered_values to the dictionary
        filtered_values_dict[session_id] = filtered_values

        # Populate 'best_log_prob' and 'best_latent_pos'
        analysis.at[index, 'best_logprob'] = filtered_values[min_value_index].item()
        analysis.at[index, 'best_latent_pos'] = str(min_value_coord.numpy().tolist())
        
        # Add mean and variance of log probabilities
        analysis.at[index, 'logprob_mean'] = torch.mean(filtered_values).item()
        analysis.at[index, 'logprob_variance'] = torch.var(filtered_values).item()
        
        # If the row is 'train', then populate 'meu_z' and 'meu_z_logprob' using the train_index
        if row['session_type'] == 'train':
            meu_z_value = dale_model.meu_z.cpu()[train_index].unsqueeze(0).unsqueeze(0)
            analysis.at[index, 'meu_z'] = str(meu_z_value.squeeze(0).squeeze(0).numpy().tolist())
            analysis.at[index, 'meu_z_logprob'] = log_prob_loss(meu_z_value, model=dale_model, update_w_data=ORACLE_update_w_data).item()  # Call the function log_prob_loss
            train_index += 1  # Increment train_index only if it's a 'train' row

    # Save the filtered_values_dict dictionary to disk
    pickle.dump(filtered_values_dict, open(f'{output_dir}/{model_id}-logprob_dict.pkl', 'wb'))

    # Later, when you want to plot, you can load the dictionary from disk
    filtered_values_dict = pickle.load(open(f'{output_dir}/{model_id}-logprob_dict.pkl', 'rb'))
        
    # Reorder the columns
    new_column_order = ['index', 'ids', 'session_type', 'meu_z', 'best_latent_pos', 'meu_z_logprob', 'best_logprob', 'logprob_mean', 'logprob_variance']
    analysis = analysis[new_column_order]

    # Save the new DataFrame as CSV
    analysis.to_csv(f'{output_dir}/{model_id}-performance.csv', index=False)

    return analysis, min_value, max_value

def load_trained_model(model_path,model_output_dim=12, latent_dim=2, N = 99):

    model = variationalNN(N, latent_dim, model_output_dim)

    model.load_state_dict(torch.load(model_path))
    model = model.eval()
    # make sure model params remain frozen
    for param in model.parameters():
        param.requires_grad = False
    model.cuda()
    return model

def log_prob_loss(meu_z, model = None, update_w_data = None, min_allowed_log_prob=-1000):
   
    # draw samples from the latent distribution
    latent_points = meu_z
    # pass samples through the model
    f = model(latent_points.cuda())  # Nsamps x 1 x 23  
    
    # initialize log probability of data under the latent distribution to 0
    total_log_prob_data = 0

    norm_const = 0

    # for each metric in the update data
    for metric in update_w_data.keys():
        # get data for the metric
        data = update_w_data[metric]

        # if there is data for the metric
        if len(data) > 0:
            
            # convert data to a tensor
            data = torch.tensor(data).float().unsqueeze(-1).unsqueeze(-1)

            # get the type of distribution to use for the metric
            metric_type = metrics_dict[metric]["type"]

            # get the output indices of the model that are relevant for the metric
            fidxs = metrics_dict[metric]["f_idxs"]

            # update the normalization constant
            norm_const += len(data)

            # get the parameters for the distribution for the metric
            counts = torch.tensor(data.shape[0]).reshape(1)
            
            dist_params = activation_dict[metric_type](
                f[:, :, fidxs], counts, metrics_dict[metric]["length"]
            )

            # create the distribution for the metric using the parameters
            dist = dist_dict[metric_type](*dist_params)

            # if the metric is binary, sum the data
            if metric_type.startswith("binary"):
                data = data.sum()

            # compute the log probability of the data under the distribution
            probs = dist.log_prob(data.cuda())

            # clamp the log probabilities to the minimum allowed value
            probs = torch.clamp(probs, min=min_allowed_log_prob)

            if not metric_type.startswith("binary"):
                probs = probs.sum(axis=0) # get total log prob of all data points as single value

            # pdb.set_trace()
        
            # compute the log probability of the data for this metric
            if probs.dim()==2:
                log_prob_task = (probs.sum(axis=1) - mean_log_probs[metric])/std_log_probs[metric]
            else:
                log_prob_task = (probs - mean_log_probs[metric])/std_log_probs[metric]
            # add the log probability of the data for this metric to the total log probability
            total_log_prob_data += log_prob_task

    # ensure the normalization constant is at least 1
    norm_const = max(1, norm_const)

    return -(total_log_prob_data/norm_const)

def run_grid_search(model, update_w_data, num_points =100):
    meu_z = generate_grid(model, num_points = num_points).cuda().unsqueeze(1)
    loss = log_prob_loss(meu_z, model =model, update_w_data = update_w_data, min_allowed_log_prob=-1000)
    return loss,meu_z
