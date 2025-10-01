import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))

sys.path.append(parent_dir)

import pandas as pd
import pingouin as pg

from utils.grid_search_utils import compute_loss_landscape, compute_predictions_fom_latent_points, predict_parameters_from_data, run_grid_search, generate_grid, log_prob_loss
from utils.mle_utils import extract_mle_parameters

import torch
from utils.variational_NN import variationalNN as variationalNN
from utils.data_distribution_utils import (
    VIS_ORDER_PREFERENCE_METRICS_MAIN_ONLY,
    get_summarized_metric_details,
    CURR_METRICS_DICT,
    dist_dict,
    prepare_data,
    model_output_dim,
    load_trained_model,
    RELEVANT_METRICS,
    DATASET,
    DEFAULT_HELDOUT_SET,
    SUMMARIZED_METRICS,
    SUMMARIZED_METRICS_MAIN_MOMENTS_LABEL_DICT,
    ALL_METRICS_MOMENTS_LABEL_DICT,
    OUTLIER_HELDOUT_SESSIONS,
    COMPUTE_DEVICE
)

import random
import math
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from IPython.display import display
import pickle
from numpy.linalg import norm
import ast
from matplotlib.lines import Line2D
import pdb

from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt, Emu
from pptx import Presentation
from datetime import date
import pandas as pd
import os
from PIL import Image
from io import BytesIO
from torch.nn.functional import softplus
import hashlib


def plot_single_histogram(data_df1, data_df2,filename="_mle_params.pdf", title = "MLE parameters", df1_label = "MLE", df2_label="Sampled"):
    
    labels = SUMMARIZED_METRICS_MAIN_MOMENTS_LABEL_DICT
        
    # Extract the names for histograms from the first row
    histogram_names = VIS_ORDER_PREFERENCE_METRICS_MAIN_ONLY if VIS_ORDER_PREFERENCE_METRICS_MAIN_ONLY else data_df2.columns.tolist()  # Exclude the first column

    # Create subplots for histograms of each column in a 3x4 grid
    if len(labels.keys()) == 8:
        print("Plotting for 8 metrics")
        fig, axs = plt.subplots(2, 4, figsize=(20, 12))  # 2 rows, 4 columns
    else:
        print("Plotting for 12 metrics")
        fig, axs = plt.subplots(3, 4, figsize=(20, 18))  # 3 rows, 4 columns
        
    axs = axs.flatten()

    for i, col in enumerate(histogram_names):

        data_class1 = data_df1[histogram_names[i]]  # Extract data for the current class
        
        # Filter out NaN and non-numeric values
        data_class_numeric1 = data_class1[pd.to_numeric(data_class1, errors='coerce').notnull()]
        

        axs[i].hist(data_class_numeric1, bins=20, alpha=1 if not ("param2" in histogram_names[i]) else 0.5, 
                    label=f'{df1_label} (n={len(data_class_numeric1)})',density=True)  # Plot histogram
        
        data_class2 = data_df2[histogram_names[i]] # Extract data for the current class

        # Filter out NaN and non-numeric values
        data_class_numeric2 = data_class2[pd.to_numeric(data_class2, errors='coerce').notnull()]
        

        axs[i].hist(data_class_numeric2, bins=20, alpha=0.8 if not ("param2" in histogram_names[i]) else 0.5, 
                    label=f'{df2_label} (n={len(data_class_numeric2)})',density=True)  # Plot histogram
        
        axs[i].set_ylabel('Frequency')
        axs[i].set_title(f'{labels[histogram_names[i]]}')
        axs[i].set_xlabel('Value Range', color='black')
        axs[i].tick_params(axis='x', labelcolor='black')
        axs[i].legend()
    
    fig.suptitle(title,fontsize = 24)
    plt.tight_layout()  # Adjust layout
    fig.patch.set_facecolor('white')

    plt.gca().set_facecolor('white')
    plt.savefig(f"{filename}.pdf")
    plt.savefig(f"{filename}.png",dpi = 300)
    plt.close(fig)  # Close the figure to free memory
    
def generate_parameter_comparison_histogram(model_id):
    mle_data_set = pd.read_csv(f"./outputs/{DATASET}/{DATASET}_mle_first_moments_only.csv")
    mle_data_set = mle_data_set.iloc[:,1:]
    sampled_data_set = pd.read_csv(f"./outputs/{DATASET}/{model_id}/{model_id}-sampled_dataset_first_moments_only.csv")
    plot_single_histogram(mle_data_set,sampled_data_set, filename=f"./outputs/{DATASET}/{model_id}/{model_id}_sampled_vs_mle_params", title =f"{model_id}- MLE and predictions from uniform samples in the latent space ")
    
def generate_mle_vs_predicted_histogram(model_id):
    mle_data_set = pd.read_csv(f"./outputs/{DATASET}/{DATASET}_mle_first_moments_only.csv")
    mle_data_set = mle_data_set.drop(columns=['session'])
    predicted_data_set = pd.read_csv(f"./outputs/{DATASET}/{model_id}/{model_id}_first_moments_only_predictions.csv")
    predicted_data_set = predicted_data_set.drop(columns=['session',"best_meu_z"])
    plot_single_histogram(mle_data_set,predicted_data_set, filename=f"./outputs/{DATASET}/{model_id}/{model_id}_predicted_vs_mle_params", title =f"{model_id}- MLE and model predictions",df2_label="Predicted")  

def calculate_figure_params(num_figures, subfig_size=(4, 4)):
    subfig_aspect_ratio = subfig_size[0] / subfig_size[1]

    best_num_rows = None
    best_num_cols = None
    min_empty_slots = float('inf')

    def get_closest_factors(num):
        factors = [(i, math.ceil(num / i)) for i in range(1, min(num + 1, 5)) if math.ceil(num / i) <= 6]
        closest_factors = min(factors, key=lambda x: (x[0] * x[1] - num, abs(x[0] - x[1])))
        return closest_factors

    best_num_rows, best_num_cols = get_closest_factors(num_figures)

    total_width = best_num_cols * subfig_size[0]
    total_height = best_num_rows * subfig_size[1]

    return best_num_rows, best_num_cols, total_width, total_height

def load_2d_models_and_plot_slices(model_path=None, output_path=None, df=None, kld_factor=None, latent_dim=None, lr=None, N=100, model_type="GP", use_activation=False, show_ids=False):
    # Set up model ID and output directory
    model_id = os.path.basename(model_path).rsplit("_", 1)[-1].split(".")[0]
    output_dir = f'{output_path}/{model_id}'
    os.makedirs(output_dir, exist_ok=True)

    # Load model and performance data
    model = load_trained_model(latent_dim=latent_dim, model_path=model_path)
    performance_file = f'./outputs/{DATASET}/{model_id}/{model_id}-performance.csv'
    performance_csv = pd.read_csv(performance_file)
    test_points_meu_z = np.array(performance_csv[performance_csv["session_type"] == "test"]["best_latent_pos"].apply(eval).map(np.array).tolist())

    # Choose metrics and plot
    metrics_list = VIS_ORDER_PREFERENCE_METRICS_MAIN_ONLY if VIS_ORDER_PREFERENCE_METRICS_MAIN_ONLY is not None else SUMMARIZED_METRICS

    # Figure setup for plot slices
    num_rows, num_cols, total_width, total_height = calculate_figure_params(len(metrics_list))
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(total_width, total_height))
    fig.suptitle(f'{model_id}{"-with activation" if use_activation else "- no activation"}', fontsize=16)
    fig.patch.set_facecolor('white')
    axs = axs.flatten()

    # Generate grid and compute model predictions
    grid = generate_grid(model, num_points=100).to(COMPUTE_DEVICE)
    model_outputs = compute_predictions_fom_latent_points(grid, model, model_output_dim, model_type="NN", with_activation=use_activation)
    x, y = grid[:, 0].cpu().numpy(), grid[:, 1].cpu().numpy()
    f = model_outputs.squeeze(0)

    
    for idx, metric in enumerate(metrics_list):
        metric_type, fidxs, _ = get_summarized_metric_details(metric)
        dist_params = f[:, fidxs]

        for k, metric_idx in enumerate(fidxs):
            c = dist_params[:, k].cpu().numpy().flatten()
            scatter = axs[idx].scatter(x, y, c=c, cmap='jet')
            axs[idx].set_title(ALL_METRICS_MOMENTS_LABEL_DICT.get(f"{metric}_param{k + 1}", f"{metric}_param{k + 1}"))

            meu_z_values = model.meu_z.cpu().numpy()
            axs[idx].scatter(meu_z_values[:, 0], meu_z_values[:, 1], color='white', edgecolor='black', marker='D', s=30)
            axs[idx].scatter(test_points_meu_z[:, 0], test_points_meu_z[:, 1], color='black', marker='X', s=20)

            if show_ids:
                for j, pos in enumerate(meu_z_values):
                    axs[idx].text(pos[0], pos[1], str(j), fontsize=7)

            plt.colorbar(scatter, ax=axs[idx])

            if metrics_list == VIS_ORDER_PREFERENCE_METRICS_MAIN_ONLY:
                break  # Only plot the first metric if using VIS_ORDER_PREFERENCE_METRICS

    plt.tight_layout()

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])  # Adjust subplot parameters to give the title space

    # Save figure in PDF and PNG formats
    file_name = f'{output_dir}/{model_id}-latent-slices{"-activation" if use_activation else "-no-activation"}'
    for ext in ['pdf', 'png']:
        plt.savefig(f'{file_name}.{ext}', format=ext, dpi=200 if ext == 'png' else None)

    plt.close()

    
def create_logprob_barplot(model_path, output_path, sort_by='meu_z_logprob', show_session_ids=True):
    # Extract the model ID
    filename = os.path.basename(model_path)  # Get the filename from the path
    model_id = filename.rsplit("_", 1)[-1].split(".")[0]  # Extract the model ID

    # Define output directory
    output_dir = f'{output_path}/{model_id}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load analysis DataFrame from CSV file
    analysis = pd.read_csv(f'{output_dir}/{model_id}-performance.csv')

    # Filter out 'outlier' rows
    analysis = analysis[analysis['session_type'] != 'outlier']

    # Calculate statistics
    meu_z_mean = analysis['meu_z_logprob'].mean()
    meu_z_std = analysis['meu_z_logprob'].std()
    training_logprob_mean = analysis.loc[analysis['session_type'] == 'train', 'best_logprob'].mean()
    training_logprob_std = analysis.loc[analysis['session_type'] == 'train', 'best_logprob'].std()
    testing_logprob_mean = analysis.loc[analysis['session_type'] != 'train', 'best_logprob'].mean()
    testing_logprob_std = analysis.loc[analysis['session_type'] != 'train', 'best_logprob'].std()

    # Create a dictionary of the calculated statistics
    stats_dict = {
        "model_id": [model_id],
        "meu_z_mean": [meu_z_mean],
        "meu_z_std": [meu_z_std],
        "training_logprob_mean": [training_logprob_mean],
        "training_logprob_std": [training_logprob_std],
        "testing_logprob_mean": [testing_logprob_mean],
        "testing_logprob_std": [testing_logprob_std]
    }

    # Create a DataFrame from the dictionary
    stats_df = pd.DataFrame(stats_dict)

    # Save the DataFrame as a CSV file in the output directory
    stats_df.to_csv(f'{output_dir}/{model_id}-performance_metrics.csv', index=False)

    # Define the labels for the legend
    legend_labels = [
        f'Minimum Logprob | Train Data\n(μ, σ) = [{training_logprob_mean:.3g}, {training_logprob_std:.3g}]',
        f'Minimum Logprob | Non-Train Data\n(μ, σ) = [{testing_logprob_mean:.3g}, {testing_logprob_std:.3g}]',
        f'Training Meu_Z Logprob | Train Data\n(μ, σ) = [{meu_z_mean:.3g}, {meu_z_std:.3g}]'
    ]

    # If sorting by magnitude of 'meu_z' in 'train' session_ids
    if sort_by == 'meu_z_mag':
        # Convert 'meu_z' from string to list and calculate the Euclidean norm
        analysis['meu_z_magnitude'] = analysis['meu_z'].apply(lambda x: norm(np.array(ast.literal_eval(x))) if isinstance(x, str) else np.nan)
        
        train_analysis = analysis[analysis['session_type'] == 'train'].sort_values(by='meu_z_magnitude', ascending=False)
        non_train_analysis = analysis[analysis['session_type'] != 'train']
        analysis_sorted = pd.concat([train_analysis, non_train_analysis], ignore_index=False)
        sort_title = "Meu_Z Distance from Origin"
    else:
        # Sort the DataFrame by 'sort_by', in descending order
        analysis_sorted = analysis.sort_values(by=sort_by, ascending=False)
        sort_title = 'Grid Minimum LogProb' if sort_by == 'best_logprob' else 'Training Meu_Z LogProb'

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(20, 10))

    # Create a color map based on the 'session_type' column
    colors = ['dodgerblue' if x == 'train' else 'skyblue' for x in analysis_sorted['session_type']]

    # Create overlapping bars for 'best_logprob' and 'meu_z_logprob'
    bar1 = ax.bar(range(len(analysis_sorted)), analysis_sorted['best_logprob'], alpha=0.5, color=colors)
    bar2 = ax.bar(range(len(analysis_sorted)), analysis_sorted['meu_z_logprob'].fillna(0), alpha=0.1, color='red')

    # Plot asterisks over the testing bars
    train_indices = np.where(analysis_sorted['session_type'] == 'test')[0]
    ax.scatter(train_indices, analysis_sorted.loc[analysis_sorted['session_type'] == 'test', 'best_logprob'] + 0.02, 
               marker='*', color='black')

    # Title, labels and legend
    title = f'({model_id}) Standardized LogProbs: Training Meu_Z vs Grid Miniumum LogProb'
    ax.set_title(title, fontsize=20)
    xlabel_text = f"Sessions Descending by {sort_title}"
    ax.set_xlabel(xlabel_text, fontsize=16)
    ax.set_ylabel('Standardized Logprob', fontsize=16)

    # If session ids are to be shown
    if show_session_ids:
        ax.set_xticks(range(len(analysis_sorted)))
        ax.set_xticklabels(analysis_sorted['ids'], rotation=90)

    # Create custom legend
    asterisk_line = Line2D([0], [0], marker='*', color='skyblue', label=legend_labels[1], markerfacecolor='black', markersize=14)
    legend_elements = [
                       Patch(facecolor='red', edgecolor='red', alpha=0.1, label=legend_labels[2]),
                       Patch(facecolor='dodgerblue', edgecolor='dodgerblue', alpha=0.5, label=legend_labels[0]),
                       asterisk_line]
    ax.legend(handles=legend_elements, fontsize=14)

    fig.tight_layout()
    fig.savefig(f'{output_dir}/{model_id}-{sort_by}.pdf', format='pdf')  # Save the figure
    
    # Save the figure as a PNG
    fig.savefig(f'{output_dir}/{model_id}-{sort_by}.png', format='png', dpi=200) 

    plt.close(fig)  # Close the figure to free memory

def drop_task_data_and_predict(model,update_w_data,obs_to_retain = 0, use_primer_sequence = False):
    predictions_dict = {}
    primer_sequence = {
    "binarySpan":{"lengths":[3,4,5,7], "n":int(obs_to_retain/4)}, # two data points per length
    "binary":{"n":obs_to_retain},
    "timing":{"n":obs_to_retain}}

    if use_primer_sequence:
        partial_update_w_data = update_w_data.copy()
        for metric in partial_update_w_data.keys():
            partial_update_w_data[metric] = []

    for metric in SUMMARIZED_METRICS:
        if not use_primer_sequence:
            partial_update_w_data = update_w_data.copy()
        metric_type, fidxs, length = get_summarized_metric_details(metric)
        
        if metric_type =="binarySpan": #these metrics have sub-metrics at each length 2-10
            obs_to_retain_found = False
            for metric_key in partial_update_w_data.keys():
                if metric in metric_key:
                    if obs_to_retain == 0: # drop all
                        partial_update_w_data[metric_key]=[] # drop data for this task
                    
                    elif use_primer_sequence: #special case to handle primer sequence
                        sub_metric_length = CURR_METRICS_DICT[metric_key]["length"]
                        
                        if obs_to_retain < 60: #use primer only
                            if sub_metric_length in primer_sequence['binarySpan']["lengths"]:
                                n_to_sample = primer_sequence['binarySpan']["n"]
                                partial_update_w_data[metric_key] = random.sample(update_w_data[metric_key], min(n_to_sample, len(update_w_data[metric_key])))
                                # print("length",sub_metric_length,"n",len(partial_update_w_data[metric_key]))
                        else: # use all data
                            partial_update_w_data[metric_key] = update_w_data[metric_key]
                    else:
                        if len(partial_update_w_data[metric_key])>0 and (not obs_to_retain_found): 
                            partial_update_w_data[metric_key] = random.sample(update_w_data[metric_key], min(obs_to_retain, len(update_w_data[metric_key])))
                            obs_to_retain_found = True
                        else:
                            partial_update_w_data[metric_key] =[] #set all other metric to 0 data
        else:
            if obs_to_retain <= 0: # drop all
                partial_update_w_data[metric]=[] # drop data for this task
            elif use_primer_sequence:
                n_to_sample = primer_sequence[metric_type]["n"]
                partial_update_w_data[metric] = random.sample(update_w_data[metric], min(n_to_sample, len(update_w_data[metric]))) # randomly sample obs_to_retain to keep and drop the rest
            else:
                partial_update_w_data[metric] = random.sample(update_w_data[metric], min(obs_to_retain, len(update_w_data[metric]))) # randomly sample obs_to_retain to keep and drop the rest
        # print("retain",obs_to_retain,metric, metric_key,partial_update_w_data)
        if not use_primer_sequence:
            lpl_result, best_meu_z = run_grid_search(model=model, update_w_data=partial_update_w_data, num_points=100)
            predictions_dict_metric = sample_predictions_in_latent_space(best_meu_z, model,file_name=None) # make predictions based on the meu_z
            predictions_dict[metric] = predictions_dict_metric.iloc[0].to_dict()[f"{metric}"] # get the parameters of the dropped task
            predictions_dict["best_meu_z"] = best_meu_z.cpu()[0].numpy().tolist()
    if use_primer_sequence:
        # for metric in update_w_data.keys():
        #     if torch.sum(torch.tensor(partial_update_w_data[metric])) != torch.sum(torch.tensor(update_w_data[metric])):
        #         print(metric, torch.sum(torch.tensor(partial_update_w_data[metric])),"==", torch.sum(torch.tensor(update_w_data[metric])), "len",len(partial_update_w_data[metric]), len(update_w_data[metric]) )
        lpl_result, best_meu_z = run_grid_search(model=model, update_w_data=partial_update_w_data, num_points=100)
        predictions_dict_metric = sample_predictions_in_latent_space(best_meu_z, model,file_name=None) # make predictions based on the meu_z
        predictions_dict = predictions_dict_metric.iloc[0].to_dict() # get the parameters of the dropped task
        result = predict_parameters_from_data(partial_update_w_data, model)
        predictions_dict["best_meu_z"] = best_meu_z.cpu()[0].numpy().tolist()
    
    return predictions_dict

def generate_analysis_csv(model_path, latent_dim, N, output_path, specific_holdout=None,base_path ="../", use_full_primer_range = False, plot_scatters = False):
    if specific_holdout is None:
        specific_holdout = ['']

    # Extract the model ID
    filename = os.path.basename(model_path)  # Get the filename from the path
    model_id = filename.rsplit("_", 1)[-1].split(".")[0]  # Extract the model ID

    # Load dale model using the extracted model ID
    dale_model = load_trained_model(model_path=model_path, latent_dim=latent_dim)
    
   

    # Define output directory
    output_dir = f'{output_path}/{model_id}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # sample uniformly from latent space and get predictions
    grid = generate_grid(dale_model, num_points=10 if latent_dim>2 else 100).to(COMPUTE_DEVICE)
    sample_predictions_in_latent_space(grid,dale_model, file_name=f"{output_dir}/{model_id}-sampled_dataset_first_moments_only.csv")
    
    #generate histogram to compare MLE params with learned parameters
    generate_parameter_comparison_histogram(model_id)

    # Load the CSV file
    participant_ids = pd.read_csv(f'{base_path}/data/{DATASET}/participant_ids_not_normed.csv')

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
    default_holdout = OUTLIER_HELDOUT_SESSIONS 

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
    
    dropped_task_preds = []
    retain_1obs_task_preds=[]
    # Either use full range of primer lengths or just a single value
    if use_full_primer_range:
        retain_primer_task_preds = {0:[],4:[], 8:[], 12:[],16:[],20:[],24:[],28:[],32:[],36:[],40:[],44:[],48:[],52:[],56:[], 60:[]}
    else:
        retain_primer_task_preds = {8:[]} # Use single primer length of 20
    # retain_primer_task_preds ={100:[]} 
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

            if ("binary" in CURR_METRICS_DICT[metric]["type"]) and len(data) > 0:
                data_list = []
                for _ in range(int(data[0])):
                    data_list.append(1.0)
                for _ in range(int(counts[0] - data[0])):
                    data_list.append(0.0)
                data = data_list
            ORACLE_update_w_data[metric] += data
            
        sessions_dropped_task_preds = drop_task_data_and_predict(dale_model,ORACLE_update_w_data) # drop data for each metric and predict
        dropped_task_preds.append(sessions_dropped_task_preds)

        sessions_retain_1_task_preds = drop_task_data_and_predict(dale_model,ORACLE_update_w_data,obs_to_retain = 1) # drop data for each metric and predict
        retain_1obs_task_preds.append(sessions_retain_1_task_preds)

        for count in retain_primer_task_preds.keys():
            sessions_retain_primer_task_preds = drop_task_data_and_predict(dale_model,ORACLE_update_w_data,obs_to_retain = count, use_primer_sequence=True) # only keep the primer sequence
            retain_primer_task_preds[count].append(sessions_retain_primer_task_preds)
        
        lpl_result, meu_z = compute_loss_landscape(model=dale_model, update_w_data=ORACLE_update_w_data, num_points=100)

        meu_z_cpu = meu_z.cpu()
        lpl_result_cpu = lpl_result.cpu()
        
        

        # Define the percentage to keep
        percentile_to_keep = 95  # keep smallest n% of lpl_result values

        # Calculate the cutoff for the smallest n% of lpl_result values
        cutoff = np.percentile(lpl_result_cpu.numpy(), percentile_to_keep)

        # Create masks for the values and coordinates that meet the cutoff
        values_mask = lpl_result_cpu <=cutoff
        meu_z_cpu_flattened = meu_z_cpu.view(meu_z_cpu.shape[0], -1)
        coords_mask = values_mask.unsqueeze(-1).expand_as(meu_z_cpu_flattened)

        # Apply the masks
        filtered_values = lpl_result_cpu[values_mask]
        filtered_coords = meu_z_cpu_flattened[coords_mask].reshape(-1, meu_z_cpu.shape[-1])
        # if values_mask.sum()==0:
        #     import pdb
        #     pdb.set_trace()
        # Find the minimum value in the scattered set of dots and corresponding coordinates
        min_value_index = np.argmin(filtered_values)
        min_value_coord = filtered_coords[min_value_index]

        # Check and update maximum value if needed
        max_value_session = torch.max(filtered_values).item()
        if max_value_session > max_value:
            max_value = max_value_session

        # Check and update minimum value if needed
        min_value_session = torch.min(filtered_values).item()
        if min_value_session <=min_value:
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
    
    #compute parameters predictions for the different sessions
  
    numpy_arrays = analysis["best_latent_pos"].apply(lambda x: np.array(eval(x)))

    # Convert the list of NumPy arrays to a 2D NumPy array
    numpy_matrix = np.vstack(numpy_arrays)
    
    ids_list = analysis["ids"] # extract IDs
    
    # Save the parameters learned by dropping tasks data
    
    dropped_task_df = pd.DataFrame(dropped_task_preds)
    dropped_task_df.insert(0,"session",ids_list)
    dropped_task_df.to_csv(f'{output_dir}/{model_id}_0obs_task_data_first_moments_only_predictions.csv')
    
    # save parameters learned by using single observation from task
    dropped_task_df = pd.DataFrame(retain_1obs_task_preds)
    dropped_task_df.insert(0,"session",ids_list)
    dropped_task_df.to_csv(f'{output_dir}/{model_id}_1obs_task_data_first_moments_only_predictions.csv')
    
    # Convert the NumPy array to a PyTorch tensor
    meu_z = torch.tensor(numpy_matrix).float()
    pred_filename = f'{output_dir}/{model_id}_first_moments_only_predictions.csv'
    preds_csv = sample_predictions_in_latent_space(meu_z,dale_model,file_name = pred_filename,model_type="NN", ids = ids_list)
    preds_csv.insert(0,"session",ids_list)
    preds_csv["best_meu_z"] = numpy_matrix.tolist()
    preds_csv.to_csv(pred_filename, index=False)
    

    if plot_scatters:
        #plot Scatter for predictions will all data
        preds_file = f'./outputs/{DATASET}/{model_id}/{model_id}_first_moments_only_predictions.csv'
        prepare_scatter_data_and_plot(model_id,y_preds_file=preds_file,figure_name ="all_task_data_corr_plot",
                                    title=f"{model_id}-Predicted and MLE parameters")
        
        #plot Scatter with data for each task dropped
        preds_file = f'./outputs/{DATASET}/{model_id}/{model_id}_0obs_task_data_first_moments_only_predictions.csv'
        prepare_scatter_data_and_plot(model_id,y_preds_file=preds_file, figure_name ="0obs_task_data_corr_plot",
                                    title=f"{model_id}-Using data from only other tasks")
        
        #plot Scatter with data for each task retaining single obs
        preds_file = f'./outputs/{DATASET}/{model_id}/{model_id}_1obs_task_data_first_moments_only_predictions.csv'
        prepare_scatter_data_and_plot(model_id,y_preds_file=preds_file, figure_name ="1obs_task_data_corr_plot",
                                    title=f"{model_id}-1 observation + data from only other tasks")
    
    
    image_paths_primer_mle = [] # store corr plots of primer vs TB-IMLE
    image_paths_primer_dlvm = [] # store corr plots of primer vs vs full DLVM
    image_paths_meu_z_scatter = []
    image_paths_meu_z_shifts = []
    for count in retain_primer_task_preds.keys():
        # save parameters learned by using single observation from task
        dropped_task_df = pd.DataFrame(retain_primer_task_preds[count])
        dropped_task_df.insert(0,"session",ids_list)
        dropped_task_df.to_csv(f'{output_dir}/{model_id}-k{count}_primer_task_data_first_moments_only_predictions.csv')
        #plot Scatter with data for each task retaining only the primer sequence
        preds_file = f'./outputs/{DATASET}/{model_id}/{model_id}-k{count}_primer_task_data_first_moments_only_predictions.csv'
        prepare_scatter_data_and_plot(model_id,y_preds_file=preds_file, figure_name =f"k{count}_primer_task_data_corr_plot",
                                title=f"{model_id}-Using fixed primer sequence (k={count})",y_label="ML-DLVM (PRIMER ONLY)",x_label="TB-IMLE (ALL DATA)")
        #plot Scatter with data for each task retaining only the primer sequence
        primer_preds_file = f'./outputs/{DATASET}/{model_id}/{model_id}-k{count}_primer_task_data_first_moments_only_predictions.csv'
        full_preds_file = f'./outputs/{DATASET}/{model_id}/{model_id}_first_moments_only_predictions.csv'
        
        prepare_scatter_data_and_plot(model_id,y_preds_file=primer_preds_file,x_preds_file = full_preds_file, figure_name =f"k{count}_primer_vs_full_task_data_corr_plot",
                                title=f"{model_id}-Using fixed primer sequence (k={count})",y_label="ML-DLVM (PRIMER ONLY)",x_label="ML-DLVM (ALL DATA)")

        meu_shifts_path = f"./outputs/{DATASET}/{model_id}/{model_id}-k{count}_primer_to_full_task_data_shifts"
        plot_line_segment(dropped_task_df["best_meu_z"].tolist(),preds_csv["best_meu_z"].tolist(), file_name =meu_shifts_path, title=f"Meu_z shift from primer (n={count}) to full data")

        image_paths_meu_z_shifts.append(f"{meu_shifts_path}.png") 
        image_paths_primer_mle.append(f'./outputs/{DATASET}/{model_id}/{model_id}-k{count}_primer_task_data_corr_plot.png')
        image_paths_primer_dlvm.append(f'./outputs/{DATASET}/{model_id}/{model_id}-k{count}_primer_vs_full_task_data_corr_plot.png')
        image_paths_meu_z_scatter.append(f'./outputs/{DATASET}/{model_id}/{model_id}-k{count}_primer_task_data_corr_plot_scatter.png')

    output_path_primer_mle= f'./outputs/{DATASET}/{model_id}/{model_id}_primer_task_data_corr_plot_gif.gif'
    output_path_primer_dlvm = f'./outputs/{DATASET}/{model_id}/{model_id}_primer_vs_full_task_data_corr_plot_gif.gif'
    output_path_meu_z_scatter = f'./outputs/{DATASET}/{model_id}/{model_id}_primer_task_data_corr_plot_scatter_gif.gif'
    output_path_meu_z_shifts = f'./outputs/{DATASET}/{model_id}/{model_id}_primer_task_data_meu_z_shifts_gif.gif'
    
    if latent_dim == 2:
        create_gif(image_paths_meu_z_scatter, output_path_meu_z_scatter, duration=1000)
        create_gif(image_paths_meu_z_shifts, output_path_meu_z_shifts, duration=1000)

    create_gif(image_paths_primer_mle, output_path_primer_mle, duration=1000)
    create_gif(image_paths_primer_dlvm, output_path_primer_dlvm, duration=1000)
    # plot histom grams
    
    generate_mle_vs_predicted_histogram(model_id)

    return analysis, min_value, max_value

def create_gif(image_paths, output_path, duration=500):
    # Open the images and store them in a list
    images = [Image.open(image) for image in image_paths]
    
    # Save the images as a GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )

def create_histograms(model_path, output_path, max_value, min_value, plot_meu_z=True, plot_session_id=False, sort_by_diff=True,base_path ="../"):

    def format_tick(val):
        if val == 1 or val == 0:
            return f'{val:.0f}'
        else:
            return f'{val:.2f}'.lstrip('0')

    filename = os.path.basename(model_path)
    model_id = filename.rsplit("_", 1)[-1].split(".")[0]
    output_dir = f'{output_path}/{model_id}'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    analysis = pd.read_csv(f'{output_dir}/{model_id}-performance.csv')
    analysis = analysis[analysis['session_type'] != 'outlier']

    # Counter for training sessions
    train_counter = 0
    for i, row in analysis.iterrows():
        if row['session_type'] == 'train':
            analysis.at[i, 'original_order'] = train_counter
            train_counter += 1

    if sort_by_diff:
        analysis['logprob_diff'] = analysis['meu_z_logprob'] - analysis['best_logprob']
        analysis = analysis.sort_values(by='logprob_diff', ascending=False)

    filtered_values_dict = pickle.load(open(f'{output_dir}/{model_id}-logprob_dict.pkl', 'rb'))
    all_hist_frequencies = []

    for index, row in analysis.iterrows():
        session_id = row['ids']
        filtered_values = filtered_values_dict[session_id]
        hist, bins = np.histogram(filtered_values.numpy(), bins=500)
        all_hist_frequencies.extend(hist)

    max_hist_frequency = np.percentile(all_hist_frequencies, 95)
    num_plots = len(analysis)
    N = int(np.ceil(np.sqrt(num_plots)))
    fig, axes = plt.subplots(N, N, figsize=(N*6, N*6))
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    for idx, (index, row) in enumerate(analysis.iterrows()):
        ax = axes.flatten()[idx]
        session_id = row['ids']
        filtered_values = filtered_values_dict[session_id]
        color = 'dodgerblue' if row['session_type'] == 'train' else 'skyblue'
        weights = np.ones_like(filtered_values.numpy()) / max_hist_frequency
        n, bins, patches = ax.hist(filtered_values.numpy(), bins=500, alpha=0.7, color=color, weights=weights)
        
        if plot_meu_z and row['session_type'] == 'train':
            meu_z_value = row['meu_z_logprob']
            ax.axvline(x=meu_z_value, color='red', linestyle='--', linewidth = 4)

        ax.set_xlim([min_value - 0.01*min_value, max_value])
        ax.set_ylim([0, 1])

        if plot_session_id:
            if row['session_type'] == 'train':
                original_order = int(row['original_order'])  # Convert to integer
                text = f'{session_id} ({original_order})'
            else:
                text = f'{session_id}'
            ax.text(0.95, 0.95, text, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=48)
            
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks(np.arange(0, np.floor(max_value) + 1, max(1, np.floor(max_value) // 4)))
        
        y_ticks = np.linspace(0, 1, 5)
        y_ticklabels = [format_tick(i) for i in y_ticks]
        ax.yaxis.set_ticks(y_ticks)
        ax.set_yticklabels(y_ticklabels)

        ax.tick_params(axis='both', which='major', labelsize=42)

        if idx % N != 0:
            ax.set_yticklabels([])
        if idx < num_plots - N:
            ax.set_xticklabels([])

    for idx in range(num_plots, N*N):
        fig.delaxes(axes.flatten()[idx])

    fig.suptitle(f'{model_id} LogProb Distribution Throughout Grid', fontsize=72)
    fig.text(0.5, 0.01, 'Standardized LogProb Values', ha='center', va='center', fontsize=60)
    fig.text(0.01, 0.5, 'Normalized Frequency', ha='center', va='center', rotation='vertical', fontsize=60)

    legend_elements = [Line2D([0], [0], color='red', lw=4, linestyle='--', label='LogProb(Meu_Z | Training Data)'),
                       Patch(facecolor='dodgerblue', edgecolor='dodgerblue', label='LogProb(Grid | Training Data)'),
                       Patch(facecolor='skyblue', edgecolor='skyblue', label='LogProb(Grid | Testing Data)')]

    fig.legend(handles=legend_elements, loc='upper right', fontsize=42)

    file_name = f'{output_dir}/{model_id}'
    if sort_by_diff:
        file_name += '-sorted'
    if plot_session_id:
        file_name += '-ids'
    file_name += '-logprob_spread.pdf'

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])
    plt.savefig(file_name, format='pdf')
    plt.savefig(file_name.replace('.pdf', '.png'), format='png', dpi=100)
    plt.close()

    return file_name

def create_logprob_barplot(model_path, output_path, sort_by='meu_z_logprob', show_session_ids=True, base_path =  "../"):
    # Extract the model ID
    filename = os.path.basename(model_path)  # Get the filename from the path
    model_id = filename.rsplit("_", 1)[-1].split(".")[0]  # Extract the model ID

    # Define output directory
    output_dir = f'{output_path}/{model_id}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load analysis DataFrame from CSV file
    analysis = pd.read_csv(f'{output_dir}/{model_id}-performance.csv')

    # Filter out 'outlier' rows
    analysis = analysis[analysis['session_type'] != 'outlier']

    # Calculate statistics
    meu_z_mean = analysis['meu_z_logprob'].mean()
    meu_z_std = analysis['meu_z_logprob'].std()
    training_logprob_mean = analysis.loc[analysis['session_type'] == 'train', 'best_logprob'].mean()
    training_logprob_std = analysis.loc[analysis['session_type'] == 'train', 'best_logprob'].std()
    testing_logprob_mean = analysis.loc[analysis['session_type'] != 'train', 'best_logprob'].mean()
    testing_logprob_std = analysis.loc[analysis['session_type'] != 'train', 'best_logprob'].std()

    # Create a dictionary of the calculated statistics
    stats_dict = {
        "model_id": [model_id],
        "meu_z_mean": [meu_z_mean],
        "meu_z_std": [meu_z_std],
        "training_logprob_mean": [training_logprob_mean],
        "training_logprob_std": [training_logprob_std],
        "testing_logprob_mean": [testing_logprob_mean],
        "testing_logprob_std": [testing_logprob_std]
    }

    # Create a DataFrame from the dictionary
    stats_df = pd.DataFrame(stats_dict)

    # Save the DataFrame as a CSV file in the output directory
    stats_df.to_csv(f'{output_dir}/{model_id}-performance_metrics.csv', index=False)

    # Define the labels for the legend
    legend_labels = [
        f'Minimum Logprob | Train Data\n(μ, σ) = [{training_logprob_mean:.3g}, {training_logprob_std:.3g}]',
        f'Minimum Logprob | Non-Train Data\n(μ, σ) = [{testing_logprob_mean:.3g}, {testing_logprob_std:.3g}]',
        f'Training Meu_Z Logprob | Train Data\n(μ, σ) = [{meu_z_mean:.3g}, {meu_z_std:.3g}]'
    ]


    # If sorting by magnitude of 'meu_z' in 'train' session_ids
    if sort_by == 'meu_z_mag':
        # Convert 'meu_z' from string to list and calculate the Euclidean norm
        analysis['meu_z_magnitude'] = analysis['meu_z'].apply(lambda x: norm(np.array(ast.literal_eval(x))) if isinstance(x, str) else np.nan)
        
        train_analysis = analysis[analysis['session_type'] == 'train'].sort_values(by='meu_z_magnitude', ascending=False)
        non_train_analysis = analysis[analysis['session_type'] != 'train']
        analysis_sorted = pd.concat([train_analysis, non_train_analysis], ignore_index=False)
        sort_title = "Meu_Z Distance from Origin"
    else:
        # Sort the DataFrame by 'sort_by', in descending order
        analysis_sorted = analysis.sort_values(by=sort_by, ascending=False)
        sort_title = 'Grid Minimum LogProb' if sort_by == 'best_logprob' else 'Training Meu_Z LogProb'

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(20, 10))

    # Create a color map based on the 'session_type' column
    colors = ['dodgerblue' if x == 'train' else 'skyblue' for x in analysis_sorted['session_type']]

    # Create overlapping bars for 'best_logprob' and 'meu_z_logprob'
    bar1 = ax.bar(range(len(analysis_sorted)), analysis_sorted['best_logprob'], alpha=0.5, color=colors)
    bar2 = ax.bar(range(len(analysis_sorted)), analysis_sorted['meu_z_logprob'].fillna(0), alpha=0.1, color='red')

    # Plot asterisks over the testing bars
    train_indices = np.where(analysis_sorted['session_type'] == 'test')[0]
    ax.scatter(train_indices, analysis_sorted.loc[analysis_sorted['session_type'] == 'test', 'best_logprob'] + 0.02, marker='*', color='black')

    # Title, labels and legend
    title = f'({model_id}) Standardized LogProbs: Training Meu_Z vs Grid Miniumum LogProb'
    ax.set_title(title, fontsize=20)
    xlabel_text = f"Sessions Descending by {sort_title}"
    ax.set_xlabel(xlabel_text, fontsize=16)
    ax.set_ylabel('Standardized Logprob', fontsize=16)

    # If session ids are to be shown
    if show_session_ids:
        ax.set_xticks(range(len(analysis_sorted)))
        ax.set_xticklabels(analysis_sorted['ids'], rotation=90)

    # Create custom legend
    asterisk_line = Line2D([0], [0], marker='*', color='skyblue', label=legend_labels[1], markerfacecolor='black', markersize=14)
    legend_elements = [
                       Patch(facecolor='red', edgecolor='red', alpha=0.1, label=legend_labels[2]),
                       Patch(facecolor='dodgerblue', edgecolor='dodgerblue', alpha=0.5, label=legend_labels[0]),
                       asterisk_line]
    ax.legend(handles=legend_elements, fontsize=14)

    fig.tight_layout()
    fig.savefig(f'{output_dir}/{model_id}-{sort_by}.pdf', format='pdf')  # Save the figure
    
    # Save the figure as a PNG
    fig.savefig(f'{output_dir}/{model_id}-{sort_by}.png', format='png', dpi=200) 

    plt.close(fig)  # Close the figure to free memory


def sample_predictions_in_latent_space(meu_z,model, file_name,model_type="NN",ids = None):
    key_metrics =VIS_ORDER_PREFERENCE_METRICS_MAIN_ONLY if VIS_ORDER_PREFERENCE_METRICS_MAIN_ONLY else SUMMARIZED_METRICS
    
    f = compute_predictions_fom_latent_points(meu_z,model,model_output_dim, model_type =model_type, with_activation=True)
        
    simulated_data ={}
    simulated_data_all_moments = {}

    for metric in key_metrics:

        metric_type, fidxs, _ = get_summarized_metric_details(metric)

        dist_params = f[:, :, fidxs].squeeze(0) # get the relevant parameters
        simulated_data[metric] =dist_params[:,0].cpu().numpy().flatten() # get the first moment
        if metric_type in ["timing"]:
            simulated_data_all_moments[f"{metric}_param1"] =dist_params[:,0].cpu().numpy().flatten()
            simulated_data_all_moments[f"{metric}_param2"] =dist_params[:,1].cpu().numpy().flatten()
        elif metric_type == "binary":            
            simulated_data_all_moments[f"{metric}_param1"] =dist_params[:,0].cpu().numpy().flatten()
        elif metric_type == "binarySpan": 
            simulated_data_all_moments[f"{metric}_param1"]=dist_params[:,0].cpu().numpy().flatten()
            simulated_data_all_moments[f"{metric}_param2"]=dist_params[:,1].cpu().numpy().flatten()
        else:    
            print("Unknown type ignored")
        
    sampled_data_set_all= pd.DataFrame.from_dict(simulated_data_all_moments)
    sampled_data_set_first_only= pd.DataFrame.from_dict(simulated_data)
    
    if file_name: # save if filename provided
        if not (ids is None):
            sampled_data_set_all.insert(0,"session",ids)
        sampled_data_set_all.to_csv(file_name.replace("first_moments_only","all_moments"),index=False)
        sampled_data_set_first_only.to_csv(file_name,index=False)
    
    return sampled_data_set_first_only
        
def load_model_configs(csv_path=None, time_stamp=None, latent_dim=None, held_out_session_ids=None, 
                      lr=None, kld_factor=None, model_path=None, model_id=None):
    """Load and filter model configurations from CSV or direct model path."""
    if model_path:
        return [os.path.basename(model_path)], [held_out_session_ids]
    
    df = pd.read_csv(csv_path)
    
    # Apply filters
    filters = {
        'time_stamp': time_stamp,
        'latent_dim': latent_dim,
        'held_out_session_ids': held_out_session_ids,
        'model_id': model_id,
        'lr': lr,
        'kld_factor': kld_factor
    }
    
    for key, value in filters.items():

        if value is not None:
            if key == 'held_out_session_ids':
                df = df[df[key].isin(value)]
            else:
                df = df[df[key] == value]
    
    return df['file_name'].tolist(), df['held_out_session_ids'].tolist()

def setup_mle_parameters():
    """Extract and save MLE parameters."""
    mle_first_moments, mle_all_moments = extract_mle_parameters()
    output_dir = f"./outputs/{DATASET}/"
    os.makedirs(output_dir, exist_ok=True)
    
    mle_first_moments.to_csv(f"{output_dir}/{DATASET}_mle_first_moments_only.csv", index=False)
    mle_all_moments.to_csv(f"{output_dir}/{DATASET}_mle_all_moments.csv", index=False)

def analyze_single_model(model_path, output_path, latent_dim, specific_holdout, base_path=parent_dir, 
                        use_full_primer_range=False, plot_scatters=False):
    """Analyze a single model and generate visualizations."""
    # Setup training data
    _, _, train_data_ids = prepare_data(
        heldout_obs_ids=DEFAULT_HELDOUT_SET + specific_holdout,
        remove_outliers=True
    )
    N = len(train_data_ids)
    print("Training set Size", N)

    # Generate analysis and plots
    analysis, min_value, max_value = generate_analysis_csv(
        model_path=model_path,
        latent_dim=latent_dim,
        N=N,
        output_path=output_path,
        specific_holdout=specific_holdout,
        base_path=base_path,
        use_full_primer_range=use_full_primer_range,
        plot_scatters=plot_scatters
    )

    # Create visualizations
    create_model_visualizations(
        model_path=model_path,
        output_path=output_path,
        min_value=min_value,
        max_value=max_value,
        latent_dim=latent_dim,
        N=N,
        base_path=base_path
    )

def create_model_visualizations(model_path, output_path, min_value, max_value, latent_dim, N, base_path="../"):
    """Create all visualizations for a model."""
    # Create histograms
    create_histograms(
        model_path=model_path,
        output_path=output_path,
        min_value=min_value,
        max_value=max_value,
        plot_meu_z=True,
        plot_session_id=True,
        sort_by_diff=True,
        base_path=base_path
    )

    # Create bar plots with different sorting
    for sort_by in ['best_logprob', 'meu_z_logprob', 'meu_z_mag']:
        create_logprob_barplot(
            model_path=model_path,
            output_path=output_path,
            sort_by=sort_by,
            show_session_ids=True
        )

    # Create 2D visualizations if applicable
    if latent_dim == 2:
        for use_activation in [False, True]:
            load_2d_models_and_plot_slices(
                model_path=model_path,
                output_path=output_path,
                latent_dim=latent_dim,
                show_ids=False,
                use_activation=use_activation,
                N=N
            )

def analyze_models(csv_path=None, time_stamp=None, base_path="../", latent_dim=None, 
                  held_out_session_ids=None, lr=None, kld_factor=None, model_path=None, 
                  model_id=None, use_full_primer_range=False, plot_scatters=False):
    """Main function to analyze one or more models."""

    # Load model configurations
    file_names, heldout_ids = load_model_configs(
        csv_path, time_stamp, latent_dim, held_out_session_ids, 
        lr, kld_factor, model_path, model_id
    )

    # Setup output path and MLE parameters
    output_path = f"./outputs/{DATASET}"
    setup_mle_parameters()

    # Analyze each model
    for idx, file_name in enumerate(file_names):
        # Save heldout IDs
        with open(f"{output_path}/heldout_ids.txt", "w") as f:
            f.write(str(heldout_ids[idx]))

        # Setup model path and specific holdout
        model_path = f"{base_path}/saved_models/{DATASET}/heldout_obsmulti{file_name}"
        specific_holdout = ast.literal_eval(heldout_ids[idx])
        print(f"Sending {model_path} to `analyze_single_model`")
        # Analyze the model
        analyze_single_model(
            model_path=model_path,
            output_path=output_path,
            latent_dim=latent_dim,
            specific_holdout=specific_holdout,
            base_path=base_path,
            use_full_primer_range=use_full_primer_range,
            plot_scatters=plot_scatters
        )

# Update the original function to use the new structure
def parse_csv_and_generate_analysis(csv_path, time_stamp=None, base_path="../", latent_dim=None,
                                  held_out_session_ids=None, lr=None, kld_factor=None, model_path=None,
                                  model_id=None, use_full_primer_range=False, plot_scatters=False):
    """Legacy function maintained for backward compatibility."""
    return analyze_models(
        csv_path, time_stamp, base_path, latent_dim, held_out_session_ids,
        lr, kld_factor, model_path, model_id, use_full_primer_range, plot_scatters
    )

def prepare_scatter_data_and_plot(model_id,y_preds_file=None,x_preds_file = None,title="Correlation Plot",figure_name="correlation_plot",x_label ="TB-IMLE",y_label ="M:-DLVM"):
    if x_preds_file is None: #assume MLE 
        x_preds_file = f'./outputs/{DATASET}/{DATASET}_mle_first_moments_only.csv'
    
    performance_file = f'./outputs/{DATASET}/{model_id}/{model_id}-performance.csv' # has information about train/test IDs
    
    x_preds_csv = pd.read_csv(x_preds_file)
    y_pred_csv = pd.read_csv(y_preds_file)
    performance_csv = pd.read_csv(performance_file)
    
    train_test_ids = performance_csv[["ids","session_type"]]
    
    # Select the "ids" and "session_type" columns
    train_test_ids = performance_csv[["ids", "session_type"]]

    # Convert to dictionary
    train_test_dict = dict(zip(train_test_ids["ids"], train_test_ids["session_type"]))
    
    param_cols = VIS_ORDER_PREFERENCE_METRICS_MAIN_ONLY if VIS_ORDER_PREFERENCE_METRICS_MAIN_ONLY else SUMMARIZED_METRICS
    plot_data = {param: {"dlvm": [], "mle": [],"point_labels":[]} for param in param_cols}  # Initialize plot_data properly
    sessions_to_remove = OUTLIER_HELDOUT_SESSIONS
    sessions = x_preds_csv["session"]
    for param in param_cols:
        for session_id in sessions:
            mle_value = x_preds_csv.loc[x_preds_csv["session"] == session_id, param].iloc[0]  # Extract single value
            pred_value = y_pred_csv.loc[y_pred_csv["session"] == session_id, param].iloc[0]  # Extract single value
            if pd.notnull(mle_value) and not (session_id in sessions_to_remove):
                plot_data[param]["mle"].append(mle_value)
                plot_data[param]["dlvm"].append(pred_value)
                plot_data[param]["point_labels"].append(train_test_dict[session_id])
    
    plot_scatter_grid(plot_data, param_cols, label=f"{model_id}-{figure_name}", 
                      xlabel=x_label, ylabel=y_label, 
                      col_to_title=SUMMARIZED_METRICS_MAIN_MOMENTS_LABEL_DICT, 
                      plot_output_path=f"./outputs/{DATASET}/{model_id}/",
                      id_list=sessions,
                      key_x="mle", key_y="dlvm",
                      point_colors = {"train":'green',"test":'orange'},
                      marker_symbols={"train":'^',"test":'x'},
                      plot_title=f"{title}", annotate=False, alpha=0.5)

    #plot meu_z
    merged_df = pd.merge(train_test_ids, y_pred_csv[["session","best_meu_z"]], left_on='ids', right_on='session')

    plot_meu_z(merged_df,file_name=f"./outputs/{DATASET}/{model_id}/{model_id}-{figure_name}_scatter",title=f"{title}")


def plot_line_segment(start_meu_z, final_meu_z, file_name, title=""):
    """
    Plots line segments from start_meu_z to final_meu_z for each session and colors/symbols based on session_type,
    then saves the plot to the specified file.

    Parameters:
    merged_df (pd.DataFrame): The merged DataFrame containing 'ids', 'session_type'.
    start_meu_z (list): List of starting meu_z coordinates.
    final_meu_z (list): List of final meu_z coordinates.
    file_name (str): The file name to save the plot (e.g., 'output.png').

    Returns:
    None
    """
    # Define markers and colors for 'train' and 'test' session types
    marker_map = {'train': '^', 'test': 'x'}
    color_map = {'train': 'green', 'test': 'orange'}

    # import pdb
    # pdb.set_trace()
    # print("start",start_meu_z)
    # print("final",final_meu_z)
    # Convert start_meu_z and final_meu_z to numpy arrays
    start_coordinates = start_meu_z #np.array([ast.literal_eval(coord) for coord in start_meu_z])
    final_coordinates = final_meu_z #np.array([ast.literal_eval(coord) for coord in final_meu_z])

    # Ensure that start and final coordinates are of the same length
    assert len(start_coordinates) == len(final_coordinates), "Start and final coordinates must have the same length"

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot line segments
    def get_color(i):
        hash_object = hashlib.md5(str(i).encode())
        hex_color = hash_object.hexdigest()[:6]
        return tuple(int(hex_color[j:j+2], 16)/255.0 for j in (0, 2, 4))

    for start, end in zip(start_coordinates, final_coordinates):
        color = get_color(hash(tuple(start)))  # Generate color based on start coordinates
        
        # Handle 1D coordinates by setting y-values to 0
        if len(start) == 1:
            x_values, y_values = [start[0], end[0]], [0, 0]
        else:
            x_values, y_values = [start[0], end[0]], [start[1], end[1]]
        
        # Plot the line segment and end points
        plt.plot(x_values, y_values, color=color, alpha=0.6)
        plt.scatter(end[0], y_values[1], color=color, marker="D")
        plt.scatter(start[0], y_values[0], color=color, alpha=0.8, marker="o")


    # Add labels and title
    plt.xlabel('Latent Variable 1')
    plt.ylabel('Latent Variable 2')
    plt.title(title)

    # Create and show legend
    plt.legend(handles=[plt.Line2D([0], [0], marker='D', label='Full data'),
                        plt.Line2D([0], [0], marker='o', label='Primer')],
               title='Session Type')

    # Save the plot to the specified file
    plt.savefig(f"{file_name}.png", dpi=300)
    plt.savefig(f"{file_name}.pdf")

    # Show the plot
    plt.close()

def plot_meu_z(merged_df, file_name, title = ""):
    """
    Plots best_meu_z for each session and colors/symbols based on session_type,
    then saves the plot to the specified file.

    Parameters:
    merged_df (pd.DataFrame): The merged DataFrame containing 'ids', 'session_type', and 'best_meu_z'.
    file_name (str): The file name to save the plot (e.g., 'output.png').

    Returns:
    None
    """
    # Define markers and colors for 'train' and 'test' session types
    marker_map = {'train': '^', 'test': 'x'}
    color_map = {'train': 'green', 'test': 'orange'}

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot each session type with a different marker and color
    for session_type in marker_map.keys():
        subset = merged_df[merged_df['session_type'] == session_type]
        # print("Meu_z",subset['best_meu_z'].shape, subset['best_meu_z'])
        subset['best_meu_z'] = subset['best_meu_z'].apply(ast.literal_eval)
        coordinates = np.array(subset['best_meu_z'].tolist())
        if coordinates.shape[1] == 1:
            plt.scatter(coordinates[:, 0], np.zeros_like(coordinates[:, 0]), 
                marker=marker_map[session_type], 
                color=color_map[session_type], 
                label=session_type)
        else:
            plt.scatter(coordinates[:, 0], coordinates[:, 1], 
                marker=marker_map[session_type], 
                color=color_map[session_type], 
                label=session_type)

    # Add labels and title
    plt.xlabel('Latent Variable 1')
    plt.ylabel('Latent Variable 2')
    plt.title(title)

    # Create and show legend
    plt.legend(title='Session Type')

    # Save the plot to the specified file
    plt.savefig(f"{file_name}.png",dpi = 300)

     # Save the plot to the specified file
    plt.savefig(f"{file_name}.pdf")

    # Show the plot
    plt.close()

def plot_scatter_grid(plot_data, param_cols,label, xlabel, ylabel, col_to_title, plot_output_path,id_list=None,
                      key_x ="tb",key_y="ml",plot_title="Correlation Plot", annotate=False, alpha=1, point_colors={"point_label1":'blue',"point_label2":'red'}, marker_symbols={"point_label1":'o',"point_label2":'x'}):
    
    num_rows, num_cols, total_width, total_height =calculate_figure_params(num_figures=len(param_cols))
    fig, axs = plt.subplots(num_rows, num_cols, figsize=( total_width, total_height))
    
    axs = axs.flatten()

    # Create a scatter plot for each parameter
    for i,param in enumerate(plot_data.keys()):
        # axs[i].scatter(plot_data[param][key_x], plot_data[param][key_y], alpha=alpha, color=color, marker=marker)
        label1_values = []
        label2_values = []
        for point_label, x, y in zip(plot_data[param]["point_labels"], 
                                       plot_data[param][key_x], plot_data[param][key_y]):
            marker = marker_symbols.get(point_label, 'o')  # 'o' as default marker if point_label is not found in symbols
            color = point_colors.get(point_label, 'blue')  # 'blue' as default color if point_label is not found in color
            
            axs[i].scatter(x, y, label=point_label, marker=marker,alpha=alpha,color=color)
            if point_label == list(point_colors.keys())[0]:
                label1_values.append([x,y])
            else:
                label2_values.append([x,y])
        
        if len(label2_values) ==0 or len(label1_values) ==0:
            continue
        label1_values = np.array(label1_values)
        label2_values = np.array(label2_values)
        
        # Plot regression line and correlation coefficient
        m, b = np.polyfit(plot_data[param][key_x], plot_data[param][key_y], 1)
        
        r_value  = compute_correlation_coeff([plot_data[param][key_x], plot_data[param][key_y]],use_icc = True) # corr coefficient
        r_test_only = compute_correlation_coeff([label2_values[:,0],label2_values[:,1]],use_icc = True)
        r_train_only = compute_correlation_coeff([label1_values[:,0],label1_values[:,1]],use_icc = True)
        
        axs[i].plot(plot_data[param][key_x], m*np.array(plot_data[param][key_x]) + b, color='red')
        
        #correlation coefficient
        axs[i].text(0.1, 0.9, f'r = {r_value:.3f}', transform=axs[i].transAxes,fontsize = 18)
        axs[i].text(0.1, 0.7, f'r = {r_test_only:.3f}', transform=axs[i].transAxes,fontsize = 18, color=point_colors.get(list(point_colors.keys())[1]))
        axs[i].text(0.1, 0.8, f'r = {r_train_only:.3f}', transform=axs[i].transAxes,fontsize = 18, color=point_colors.get(list(point_colors.keys())[0]))
        
        # Plot line of perfect equality
        lims = [
            np.min([axs[i].get_xlim(), axs[i].get_ylim()]),  # min of both axes
            np.max([axs[i].get_xlim(), axs[i].get_ylim()]),  # max of both axes
        ]
        
        axs[i].plot(lims, lims, color='black', alpha=0.75, zorder=0)
        
        axs[i].set_xlim(lims)
        axs[i].set_ylim(lims)
        
        # Add annotations if required
        if annotate:
            for k, txt in enumerate(id_list):
                # print(txt)
                axs[i].annotate(txt[5:], (plot_data[param][key_x][k], plot_data[param][key_y][k]))
        
        n = len(plot_data[param][key_x]) # Count of dots
        axs[i].set_title(f'{col_to_title[param]} (n = {n})',fontsize = 14)
        axs[i].set_xlabel(xlabel,fontsize = 18)
        axs[i].set_ylabel(ylabel,fontsize = 18)
        axs[i].tick_params(axis='both', which='major', labelsize=14)
    
    # Create the directory if it does not exist
    output_dir = plot_output_path
    os.makedirs(output_dir, exist_ok=True)
    fig.suptitle(plot_title)
    fig.tight_layout()
    # Save as save_file_type
    plt.savefig(output_dir + '/{}.{}'.format(label,"png"), format= "png")
    plt.savefig(output_dir + '/{}.{}'.format(label, "pdf"), format= "pdf")
    plt.close(fig)  # Close the figure to free memory
    
def compute_correlation_coeff(data, use_icc = True): # TODO validate ICC computation
    
    if len(data[0])<5:
        return np.nan
    
    if not use_icc:
        return np.corrcoef(data[0], data[1])[0,1] # pearson co-efficient
    else:
        ratings_a = data[0]
        ratings_b = data[1]

        # Combine the ratings arrays into a single column
        combined_ratings = np.concatenate((ratings_a, ratings_b))

        # Create a DataFrame with the combined ratings
        df = pd.DataFrame({'ratings': combined_ratings})

        # Create a new column 'rater' with values 'A' and 'B' for the first and second set of ratings, respectively
        df['rater'] = np.where(df.index < len(ratings_a), 'A', 'B')

        # Add a new column 'target' representing the index
        df['target'] = np.concatenate((range(0,len(ratings_a)), range(0,len(ratings_a))))

        # import pdb
        # pdb.set_trace()
        # print(df)
        icc_value = pg.intraclass_corr(data=df,targets='target',  raters='rater',    ratings="ratings",  nan_policy='omit').loc[0, 'ICC']
    
        return icc_value

def visualize_latent_space_to_parameter_mapping(model, validation_meu_z, colormap='Grays', use_activation=False, show_ids=False,
                                                plot_title=None, show_all_legends=False, figure_width=None):
    """
    Visualize the mapping from latent space to parameter space with both training and validation meu_z points.

    Args:
        model: Trained model with meu_z attribute
        validation_meu_z: Validation latent points (numpy array or tensor)
        colormap: Colormap for the parameter visualization (default: 'grayscale')
        use_activation: Whether to use activation in model predictions
        show_ids: Whether to show IDs on training meu_z points

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    if torch.is_tensor(validation_meu_z):
        validation_meu_z = validation_meu_z.cpu().numpy()

    metrics_list = VIS_ORDER_PREFERENCE_METRICS_MAIN_ONLY if VIS_ORDER_PREFERENCE_METRICS_MAIN_ONLY is not None else SUMMARIZED_METRICS

    num_rows, num_cols, total_width, total_height = calculate_figure_params(len(metrics_list))

    if figure_width is not None and figure_width > 0:
        width_scale = figure_width / total_width
        figure_width = float(figure_width)
    else:
        width_scale = 1.0
        figure_width = float(total_width)

    figure_height = max(1.5, total_height * width_scale)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(figure_width, figure_height))

    font_scale = max(0.25, width_scale)
    min_fontsize = 4
    suptitle_fontsize = max(min_fontsize, 24 * font_scale)

    if plot_title is not None:
        fig.suptitle(plot_title, fontsize=suptitle_fontsize)
    else:
        fig.suptitle(
            f'Latent Space to Parameter Mapping{" - with activation" if use_activation else " - no activation"}',
            fontsize=suptitle_fontsize
        )

    fig.patch.set_facecolor('white')
    axs = np.atleast_1d(axs).flatten()

    grid = generate_grid(model, num_points=100).to(COMPUTE_DEVICE)
    model_outputs = compute_predictions_fom_latent_points(grid, model, model_output_dim, model_type="NN", with_activation=use_activation)
    x, y = grid[:, 0].cpu().numpy(), grid[:, 1].cpu().numpy()
    f = model_outputs.squeeze(0)

    base_marker_area = plt.rcParams.get('lines.markersize', 6) ** 2
    size_scale = max(0.25, width_scale)
    area_scale = max(0.08, size_scale ** 2)
    scatter_area = base_marker_area * area_scale * 0.4
    highlight_area = max(4.0, 18 * area_scale)

    axis_title_fontsize = max(min_fontsize, 12 * font_scale)
    axis_tick_fontsize = max(min_fontsize, 8 * font_scale)
    legend_fontsize = max(6.0, 8 * font_scale)
    colorbar_tick_fontsize = max(min_fontsize, 8 * font_scale)

    for idx, metric in enumerate(metrics_list):
        metric_type, fidxs, _ = get_summarized_metric_details(metric)
        dist_params = f[:, fidxs]

        for k, metric_idx in enumerate(fidxs):
            c = dist_params[:, k].cpu().numpy().flatten()
            if metric_type == "timing" and use_activation:
                c = np.exp(c)
            scatter = axs[idx].scatter(x, y, c=c, cmap=colormap, s=scatter_area)
            axs[idx].set_title(
                ALL_METRICS_MOMENTS_LABEL_DICT.get(f"{metric}_param{k + 1}", f"{metric}_param{k + 1}"),
                fontsize=axis_title_fontsize
            )

            meu_z_values = model.meu_z.cpu().numpy()
            axs[idx].scatter(
                meu_z_values[:, 0],
                meu_z_values[:, 1],
                color='white',
                edgecolor='black',
                marker='D',
                s=highlight_area,
                label=f'Train (n = {meu_z_values.shape[0]})'
            )

            axs[idx].scatter(
                validation_meu_z[:, 0],
                validation_meu_z[:, 1],
                color='red',
                marker='o',
                s=highlight_area,
                label=f'Validation (n = {validation_meu_z.shape[0]})'
            )

            if show_ids:
                for j, pos in enumerate(meu_z_values):
                    axs[idx].text(pos[0], pos[1], str(j), fontsize=max(min_fontsize, 7 * font_scale))

            cbar = plt.colorbar(scatter, ax=axs[idx])
            cbar.ax.tick_params(labelsize=colorbar_tick_fontsize)

            legend = axs[idx].legend(
                fontsize=legend_fontsize,
                markerscale=max(0.4, size_scale * 0.8),
                handlelength=max(0.6, size_scale),
                handletextpad=0.25,
                borderpad=0.2,
                labelspacing=0.25,
                frameon=True
            )
            legend.get_frame().set_linewidth(0.5)
            legend.get_frame().set_alpha(0.8)

            if metrics_list == VIS_ORDER_PREFERENCE_METRICS_MAIN_ONLY:
                break

        if not show_all_legends and idx < len(metrics_list) - 1:
            legend = axs[idx].get_legend()
            if legend is not None:
                legend.remove()

        axs[idx].tick_params(labelsize=axis_tick_fontsize)

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])

    for ax in axs:
        ax.set_aspect('equal', adjustable='box')

    return fig
