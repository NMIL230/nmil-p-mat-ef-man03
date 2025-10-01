# This module contains utility functions for working with data distributions, including loading data, trained models
import glob
import sys
import os
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
from torch.nn.functional import softplus
import random
import pandas as pd
import pdb
from utils.set_seed import set_seed
import json
import importlib.util
import re
from dataset_configurations.generate_configs_file import create_config_file # to create config file if not exists
from utils.data_preprocessing_utils import generate_full_matrix # to create config file if not exists
from utils.variational_NN import variationalNN # definition of the NN model

import time


########################### DEFINE ALL GLOBAL CONSTANTS HERE ############################################################

# Whether to use two-parameter sigmoid functions (control learning of parameters)
TWO_PARAM_SIGMOIDS = True  # controlls if Sigmoid lambda and gamma should fixed - only alpha and beta are learned when true
RELEVANT_METRICS_ONLY = True  # Ignore metrics that are not of interest -- foreample complex span reaction time
LOG_NORMAL_NORMALIZATION_CONST = (
    500  # used to normalize the log probilities from LogNormal distributions
)
MAX_REACTION_TIME = 10000 # 10 seconds
RANDOM_SEED = 0
set_seed(RANDOM_SEED) #Set Random Seed for reproductibility 

# Get the home path of the project.
PROJECT_HOME_PATH = f"{os.path.dirname(os.path.realpath(__file__))}/../"
PROJECT_HOME = Path(PROJECT_HOME_PATH).resolve()

# Load configuration settings from the configurations.json file.
with open(f'{PROJECT_HOME_PATH}/configurations.json', 'r') as config_file:
    config = json.load(config_file)

# Set computation device: GPU or CPU based on config.
if config["USE_GPU_DEVICE"]=="yes":
    COMPUTE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    COMPUTE_DEVICE = "cpu"
    
DATASET = config["DATASET"] # dataset to use

########################### GLOBAL CONSTANTS END HERE ######################################################################

########################### DATASET SPECIFIC GLOBAL CONSTANTS START HERE ######################################################################
# Load the module file for the selected dataset, dynamically importing it.
module_file = f"{DATASET}.py"
try:
    # Create config file if it doesn't exist.
    if not os.path.exists(f"{PROJECT_HOME_PATH}/dataset_configurations/config_files/{module_file}"):
        create_config_file(DATASET)
        
    # Dynamically load the dataset module.
    spec = importlib.util.spec_from_file_location(DATASET, f"{PROJECT_HOME_PATH}/dataset_configurations/config_files/{module_file}")
    dataset_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset_module)
except FileNotFoundError:
    # Exit if the module file does not exist.
    print(f"FAILED and EXITED: Module file '{module_file}' not found for dataset '{DATASET}' in 'dataset_configurations' folder ")
    sys.exit(0)


def _create_placeholder_dataset(data_dir: Path, metrics_dict: dict, relevant_metrics) -> None:
    """Create minimal placeholder assets so figure scripts can run without raw COLL10 data."""
    data_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = data_dir / "full_data_matrix_not_normed.pt"
    participants_path = data_dir / "participant_ids_not_normed.csv"
    csv_path = data_dir / "all_data_time_not_normed.csv"

    placeholder_session = "placeholder_session"

    metrics = list(relevant_metrics)
    matrix = torch.zeros((1, 1, len(metrics)), dtype=torch.float32)
    for idx, metric in enumerate(metrics):
        metric_info = metrics_dict.get(metric, {})
        metric_type = metric_info.get("type", "binary")
        if metric_type == "timing":
            matrix[0, 0, idx] = 1.0
        elif metric_type == "beta":
            matrix[0, 0, idx] = 0.5
        else:
            matrix[0, 0, idx] = 1.0

    torch.save(matrix, matrix_path)
    pd.DataFrame({"ids": [placeholder_session]}).to_csv(participants_path, index=False)

    if not csv_path.exists():
        rows = []
        for idx, metric in enumerate(metrics):
            metric_info = metrics_dict.get(metric, {})
            rows.append(
                {
                    "user_session": placeholder_session,
                    "task_label": metric_info.get("summary_metric_label", metric),
                    "data_type": metric_info.get("type", "binary"),
                    "metric": metric,
                    "result": float(matrix[0, 0, idx]),
                    "presentation_time": float(idx),
                }
            )
        pd.DataFrame(rows).to_csv(csv_path, index=False)


def _ensure_minimal_data_assets(metrics_dict: dict, relevant_metrics) -> None:
    """Ensure required tensor/CSV artifacts exist, synthesizing placeholders if necessary."""
    data_dir = PROJECT_HOME / "data" / DATASET
    data_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = data_dir / "full_data_matrix_not_normed.pt"
    participants_path = data_dir / "participant_ids_not_normed.csv"
    csv_path = data_dir / "all_data_time_not_normed.csv"

    if matrix_path.exists() and participants_path.exists():
        return

    if csv_path.exists():
        try:
            generate_full_matrix(DATASET, relevant_metrics, normalize_times=False)
        except Exception as exc:
            print(
                "Warning: Failed to build full data matrix from raw data. "
                f"Falling back to placeholder assets. Details: {exc}"
            )
        if matrix_path.exists() and participants_path.exists():
            return

    _create_placeholder_dataset(data_dir, metrics_dict, relevant_metrics)

# Import variables from the imported module
required_variables = ["METRICS_DICT", "RELEVANT_METRICS", "SUMMARIZED_METRICS",
                      "SUMMARIZED_METRICS_MAIN_MOMENTS_LABEL_DICT", "ALL_METRICS_MOMENTS_LABEL_DICT",
                      "ALL_METRICS", "BLACKLIST_SET", "DEFAULT_HELDOUT_SET","SUMMARIZED_METRICS_METRIC_TYPES"]

# Check that all required variables are present in the dataset module.
for var_name in required_variables:
    if not hasattr(dataset_module, var_name):
        sys.exit(f"Variable '{var_name}' not found in {module_file} for dataset '{DATASET}'")

# Assign variables from the dataset module to be used later.
CURR_METRICS_DICT = dataset_module.METRICS_DICT
RELEVANT_METRICS = dataset_module.RELEVANT_METRICS # full list of metrics
SUMMARIZED_METRICS = dataset_module.SUMMARIZED_METRICS # 2 corsi task combined
SUMMARIZED_METRICS_MAIN_MOMENTS_LABEL_DICT = dataset_module.SUMMARIZED_METRICS_MAIN_MOMENTS_LABEL_DICT
SUMMARIZED_METRICS_METRIC_TYPES = dataset_module.SUMMARIZED_METRICS_METRIC_TYPES
ALL_METRICS_MOMENTS_LABEL_DICT = dataset_module.ALL_METRICS_MOMENTS_LABEL_DICT
# ALL_METRICS = dataset_module.ALL_METRICS
OUTLIER_HELDOUT_SESSIONS = dataset_module.BLACKLIST_SET
DEFAULT_HELDOUT_SET = dataset_module.DEFAULT_HELDOUT_SET

VIS_ORDER_PREFERENCE_METRICS_MAIN_ONLY = dataset_module.VIS_ORDER_PREFERENCE_METRICS_MAIN_ONLY if hasattr(dataset_module, "VIS_ORDER_PREFERENCE_METRICS_MAIN_ONLY") else None
VIS_ORDER_PREFERENCE_METRICS_ALL = dataset_module.VIS_ORDER_PREFERENCE_METRICS_ALL if hasattr(dataset_module, "VIS_ORDER_PREFERENCE_METRICS_ALL") else None

# Ensure downstream utilities have at least minimal data assets available.
_ensure_minimal_data_assets(CURR_METRICS_DICT, RELEVANT_METRICS)

RAW_DATA_AVAILABLE = (PROJECT_HOME / "data" / DATASET / "all_data_time_not_normed.csv").exists()
FULL_MATRIX_AVAILABLE = (PROJECT_HOME / "data" / DATASET / "full_data_matrix_not_normed.pt").exists()

########################### DATASET SPECIFIC GLOBAL CONSTANTS END HERE ######################################################################

########################### # CHECK THERE IS THE FULL DATA MATRIX FILE ######################################################################
# Path to the full data matrix file.
data_matrix_path = os.path.join(PROJECT_HOME_PATH, f"data/{DATASET}/full_data_matrix_not_normed.pt")

########################### END CHECK FOR MATRIX DATASET

# Helper function to create a folder if it does not exist.
def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
        
# Create folders for data storage, model training analysis, and visualization outputs.
create_folder(f"{PROJECT_HOME_PATH}/data/{DATASET}/")
create_folder(f"{PROJECT_HOME_PATH}/model_training_analysis/{DATASET}/")
create_folder(f"{PROJECT_HOME_PATH}/visualization/presentations/{DATASET}/")
create_folder(f"{PROJECT_HOME_PATH}/visualization/outputs/{DATASET}/")
create_folder(f"{PROJECT_HOME_PATH}/visualization/mle/{DATASET}/")
    
# Dictionary of distributions used for different metrics.
dist_dict = {
    "binary": torch.distributions.binomial.Binomial,
    # "binary": torch.distributions.bernoulli.Bernoulli,
    "timing": torch.distributions.log_normal.LogNormal,
    "beta": torch.distributions.beta.Beta,
    "binarySpan": torch.distributions.binomial.Binomial,
}


# Dictionary of prior distributions for the parameters used in the Bayesian model.
prior_dist_dict = {
    "binary": [torch.distributions.beta.Beta(torch.tensor(1.0, device=COMPUTE_DEVICE), torch.tensor(3.0, device=COMPUTE_DEVICE))],
    # "binary": torch.distributions.bernoulli.Bernoulli,
    "timing": [
        torch.distributions.Normal(torch.tensor(5, device=COMPUTE_DEVICE), torch.tensor(5, device=COMPUTE_DEVICE)),
        torch.distributions.Normal(torch.tensor(1, device=COMPUTE_DEVICE), torch.tensor(5, device=COMPUTE_DEVICE))
    ],  # [mean, std]
    "beta": [
        torch.distributions.uniform.Uniform(torch.tensor(0, device=COMPUTE_DEVICE), torch.tensor(10, device=COMPUTE_DEVICE)),
        torch.distributions.uniform.Uniform(torch.tensor(0, device=COMPUTE_DEVICE), torch.tensor(10, device=COMPUTE_DEVICE))
    ],  # [alpha, beta]
    "binarySpan": [
        torch.distributions.Normal(torch.tensor(6, device=COMPUTE_DEVICE), torch.tensor(3, device=COMPUTE_DEVICE)),
        torch.distributions.Normal(torch.tensor(10, device=COMPUTE_DEVICE), torch.tensor(5, device=COMPUTE_DEVICE))
    ]  # [alpha, beta]
}

# Function to activate the binarySpan task metrics using a parameterized sigmoid function.
def binarySpan_activation(fs, counts, length):

    # Apply softplus activation to the first two parameters of the input tensor and set the lower bound to 0
    a_, b_ = fs[:, :, 0] , fs[:, :, 1]
    if length is None: # no specific length is given just return the sigmoid parameters i.e a_ and b_
        a_, b_ = clamp_sigmoid_params(a_,b_)
        return [a_,b_]
    
    # Compute the gamma and lambda parameters of the sigmoid function
    if TWO_PARAM_SIGMOIDS:
        g, l = torch.tensor(0.02), torch.tensor(0.02)
    else:
        g, l = torch.sigmoid(fs[:, :, 2]) * 0.5, torch.sigmoid(fs[:, :, 3]) * 0.5
    
    # Compute the predicted probabilities using the parameterized sigmoid function and the computed parameters
    pred_probs = get_differentiable_sigmoid(a_, b_, g, l)(length)
    
    # Return the predicted probabilities and input counts as a list
    return [counts, pred_probs]

def clamp_sigmoid_params(a,b):
    a_ = torch.clamp(softplus(a), min=0, max=20)
    b_ = torch.clamp(softplus(b), min=0, max=20)
    return a_, b_

# Function to return a parameterized sigmoid function for binarySpan tasks.
def get_differentiable_sigmoid(a, b, gamma, lambda_):
    def sigmoid(x):
        # Clamp the transformed values to a maximum value of 20
        a_, b_ = clamp_sigmoid_params(a,b)
        
        # Compute the value of the sigmoid function for a given input
        return gamma + ((1 - gamma - lambda_) / (1 + torch.exp((10/(b_+1))*(x-a_)))) 

    # Return the sigmoid function
    return sigmoid

def convert_sigmoid_params(threshold_param, spread_param, gamma=0.02, lambda_=0.02, to="presentable"):
    """
    Converts sigmoid parameters between differentiable form and presentable form.

    Args:
    - threshold_param (torch.Tensor): The threshold parameter of the sigmoid function.
    - spread_param (torch.Tensor): The spread parameter of the sigmoid function.
    - gamma (torch.Tensor): The gamma parameter of the sigmoid function.
    - lambda_ (torch.Tensor): The lambda parameter of the sigmoid function.
    - to (str): The direction of conversion. Can be "presentable" to convert to presentable form or "differentiable" to convert to differentiable form.

    Returns:
    - tuple: The converted parameters (psiTheta, psiSigma, psiGamma, psiLambda) in presentable form or (threshold_param, spread_param, gamma, lambda_) in differentiable form.
    """

    if to == "presentable":
        threshold_param, spread_param = clamp_sigmoid_params(threshold_param, spread_param) # ensure the parameters are within the valid range
        psiTheta = threshold_param
        psiSigma = (spread_param + 1) / 10
        psiGamma = gamma
        psiLambda = lambda_
        return psiTheta, psiSigma, psiGamma, psiLambda
    elif to == "differentiable":
        # throw error if the parameters are less than 0
        if threshold_param < 0 or spread_param < 0:
            raise ValueError("psiTheta and psiSigma must be greater than or equal to 0.")
        def inverse_softplus(y):
            return torch.where(y > 20, y, torch.log(torch.expm1(y)))
        threshold_param = inverse_softplus(threshold_param)
        spread_param = inverse_softplus(spread_param * 10 - 1)
        psiGamma = gamma
        psiLambda = lambda_
        return threshold_param, spread_param, psiGamma, psiLambda
    else:
        raise ValueError("Invalid conversion direction. Use 'presentable' or 'differentiable'.")


# Function to generate a presentable sigmoid function with specified parameters.
def get_presentable_sigmoid(psiTheta, psiSigma, gamma=0.02, lambda_=0.02, transform_params=False):
    """
    Generate a presentable sigmoid function with specified parameters.

    This function returns a sigmoid function intended for presentation purposes and is not differentiable.
    It should not be used for differentiation or model building.

    Parameters:
    psiTheta (float): The theta parameter for the sigmoid function.
    psiSigma (float): The sigma parameter for the sigmoid function. Must be greater than 0.
    gamma (float, optional): The gamma parameter for the sigmoid function. Default is 0.02.
    lambda_ (float, optional): The lambda parameter for the sigmoid function. Default is 0.02.
    transform_params (bool, optional): Whether to transform the parameters to presentable form. Default is True.

    Returns:
    function: A sigmoid function with the specified parameters.

    Raises:
    ValueError: If psiSigma is not greater than 0.
    """
    if transform_params:
        psiTheta, psiSigma, gamma, lambda_ = convert_sigmoid_params(psiTheta, psiSigma, gamma, lambda_, to="presentable")
        
    if psiSigma <= 0:
        raise ValueError("psiSigma must be greater than 0.")
    
    def sigmoid(x):
        # Compute the value of the sigmoid function for a given input
        return gamma + ((1 - gamma - lambda_) / (1 + torch.exp((x - psiTheta) / psiSigma)))
    
    return sigmoid

# Function to compute the prior log probability for different metric types.
def prior_log_prob(params, metric_type):
    if metric_type =="binary":
        params = [torch.clamp(torch.sigmoid(params[:,0]).squeeze(), min=1e-2, max=1 - 1e-2)]
        return prior_dist_dict[metric_type][0].log_prob(params[0])
    elif metric_type =="timing":
        params = [torch.clamp(params[:,0],min=0, max = torch.log(torch.tensor(MAX_REACTION_TIME))), torch.clamp(params[:,1],min=0, max = 10)]
        return prior_dist_dict[metric_type][0].log_prob(params[0])+prior_dist_dict[metric_type][1].log_prob(params[1])
    elif metric_type =="binarySpan":
        params = [torch.clamp(softplus(params[:,0]), min=0, max=20) , torch.clamp(softplus(params[:,1]), min=0, max=20)]
        return prior_dist_dict[metric_type][0].log_prob(params[0])+prior_dist_dict[metric_type][1].log_prob(params[1])

# Activation functions for different metric types
# These functions transform the raw parameters (fs) to valid distributional parameters used in the model.

# Activation for binary metrics (e.g., 0-1 outcomes like correct/incorrect responses)
def binary_activation(fs, counts, length):
    return [counts, torch.clamp(torch.sigmoid(fs).squeeze(), min=1e-2, max=1 - 1e-2)] # preventing p =0 or p = 1
    # return [counts, torch.sigmoid(fs).squeeze()] # preventing p =0 or p = 1

# Activation for timing metrics (e.g., reaction times modeled by a log-normal distribution)
def timing_activation(fs, counts, length):
    return [torch.clamp(softplus(fs[:, :, 0]),min=0, max = torch.log(torch.tensor(MAX_REACTION_TIME))), softplus(fs[:, :, 1])]

# Activation for beta distributions (used for modeling probabilities between 0 and 1)
def beta_activation(fs, counts, length):
    return [softplus(fs[:, :, 0]), softplus(fs[:, :, 1])]

# Activation for Bernoulli distribution (similar to binary activation, but using Bernoulli outcomes)
def bernoulli_activation(fs,counts, length):
    return [torch.clamp(torch.sigmoid(fs).squeeze(), min=1e-2, max=1 - 1e-2)]
    
# Dictionary of activation functions for each metric type
activation_dict = {
    "binary": binary_activation,  # 0 - 1 range for prob
    # "binary": bernoulli_activation,.
    "timing": timing_activation,  # any num for mean of log normal, scale must be positive tho...
    "beta": beta_activation,  # positive nums for alpha and beta for beta dist
    "binarySpan": binarySpan_activation,
}


# Function to compute the log-probability of a given distribution
def compute_dist_log_prob(dist, data, metric_type):
    # Placeholder function: Not yet implemented
    # if metric_type == "binary":
    return None


# Create a dictionary mapping each metric to its corresponding index in the model output
# get index of model output f needed for each metric, store in dict 
def create_metrics_dict(use_relevant_metrics_only = True):
    # This function calculates which parameters from the model output correspond to which metrics.

    if TWO_PARAM_SIGMOIDS:
        sigmoid_n_params = 2  # Only two parameters for sigmoid (alpha, beta)
    else:
        sigmoid_n_params = 4  # Four parameters if using a more complex sigmoid (alpha, beta, gamma, lambda)
        
    if use_relevant_metrics_only:
        metrics_to_use = RELEVANT_METRICS  # Use only metrics marked as relevant
    else:
        metrics_to_use = CURR_METRICS_DICT.keys() # Use all available metrics
    
    idx = 0 #Initialize index for model outputs
    previous_task = None #it's assumed that binary span metrics are in order i.e following each other # To track the previous task (important for binarySpan tasks)
   
    # Loop through each metric and assign corresponding model output indices
    for ix, metric in enumerate(metrics_to_use):
        type_ = CURR_METRICS_DICT[metric]["type"] # Get the type of the current metric
        CURR_METRICS_DICT[metric]["length"] = None # Initialize length for the metric
        if previous_task: # None should be ignored # For binarySpan tasks, check if it's a new task or continuation of the previous
            if previous_task["type"] =="binarySpan":
                if previous_task["summary_metric_label"] != CURR_METRICS_DICT[metric]["summary_metric_label"]: # new task following previou span task
                    idx += sigmoid_n_params # Shift indices if a new task follows a previous span task

        # Assign indices for binary metrics (require one parameter)
        if type_ == "binary":  # needs one param
            CURR_METRICS_DICT[metric]["f_idxs"] = [idx]
            idx += 1
        
        # Assign indices for binarySpan tasks (require multiple parameters and a length parameter)
        elif (
            type_ == "binarySpan"
        ):  # span tasks need 4 parameters, also need length param added
            CURR_METRICS_DICT[metric]["f_idxs"] = [idx + i for i in range(sigmoid_n_params)]
            length_param = int(re.search(r'\d+$', metric).group()) # get the number in the metric as the length
            CURR_METRICS_DICT[metric]["length"] = length_param

        # Assign indices for timing or beta metrics (require two parameters)
        elif (type_ == "timing") or (type_ == "beta"):  # need 2 params
            CURR_METRICS_DICT[metric]["f_idxs"] = [idx, idx + 1]
            idx += 2

        # Error handling for missing task types
        else:
            raise RuntimeError("missing task type")
        
        previous_task = CURR_METRICS_DICT[metric] # Track the current task for the next iteration
        
    # Finalize the model output dimension
    if previous_task["type"] =="binarySpan":
        model_output_dim = idx+sigmoid_n_params
    else:
        model_output_dim = idx   
        

    # Assign original indices for each metric (useful for backtracking original values)
    # also get og fixs to use when needed
    idx = 0
    previous_task = None #it's assumed that binary span metrics are in order i.e following each other
    for ix, metric in enumerate(CURR_METRICS_DICT.keys()):
        type_ = CURR_METRICS_DICT[metric]["type"]
        if previous_task: # None should be ignored
            if previous_task["type"] =="binarySpan":
                if previous_task["summary_metric_label"] != CURR_METRICS_DICT[metric]["summary_metric_label"]: # new task following previou span task
                    idx += sigmoid_n_params # shift indices
                    
        # Same process for assigning original indices for different types
        if type_ == "binary":  # needs one param
            CURR_METRICS_DICT[metric]["f_idxs_og"] = [idx]
            idx += 1
        elif (
            type_ == "binarySpan"
        ):  # span tasks need 4 parameters, also need length param added
            CURR_METRICS_DICT[metric]["f_idxs_og"] = [idx + i for i in range(sigmoid_n_params)]
        elif (type_ == "timing") or (type_ == "beta"):  # need 2 params
            CURR_METRICS_DICT[metric]["f_idxs_og"] = [idx, idx + 1]
            idx += 2
        else:
            raise RuntimeError("missing task type")
        previous_task = CURR_METRICS_DICT[metric]
    
            
    return CURR_METRICS_DICT, model_output_dim # Return updated dictionary and model output dimension

# Create the metrics dictionary and calculate model output dimensions.
CURR_METRICS_DICT, model_output_dim = create_metrics_dict(RELEVANT_METRICS_ONLY)

# Function to transform the raw data in the full data matrix
def transform_data(full_data_matrix, metrics=None):
    """
    Transforms the raw data in the full data matrix into a format that can be used for training/testing machine learning models.
    This includes handling boundary values for certain metrics, such as 0 or 1 values for beta metrics, and 0 values for timing metrics.
    
    Args:
    - full_data_matrix (torch.Tensor): A tensor of shape (N, T, M) containing the raw data for N participants, T trials,
        and M metrics.
    - metrics_dict (dict): A dictionary containing information about the metrics, including their types.

    Returns:
    - data_dict (dict): A dictionary containing the transformed data for each metric, including:
        - data: A tensor of shape (N, T) containing the raw data with boundary values handled.
        - tmp_data: A tensor of shape (T, N, 1) containing the transformed data used for computing log-likelihoods.
        - counts: A tensor of shape (N,) containing the number of valid trials for each participant/session.
        - mask: A tensor of shape (N, T) containing a binary mask indicating which observations are valid.
    """
    data_dict = {}  # Initialize dictionary to store transformed data for each metric

    for metric_num, metric in enumerate(metrics):
        metric_type = CURR_METRICS_DICT[metric]["type"]  # Get the type of the current metric
        data = full_data_matrix[:, :, metric_num]  # Select the data for the current metric

        # Compute the number of valid trials for each participant, excluding NaN values
        counts = torch.where(
            torch.logical_not(torch.isnan(data)), torch.ones_like(data), data
        ).nansum(dim=-1)

        # Handle boundary values for beta metrics (replacing exact 0s and 1s)
        if metric_type == "beta":
            data = torch.where(data == 0.0, torch.ones_like(data) * 0.00001, data)  # Replace 0 values with small positive values
            data = torch.where(data == 1.0, torch.ones_like(data) * 0.99999, data)  # Replace 1 values with slightly smaller values
            tmp_data = (
                torch.where(data.isfinite(), data, torch.ones_like(data) * 0.5)  # Replace NaN values with 0.5
                .transpose(0, 1)
                .unsqueeze(-2)
            )

        # Handle timing metrics (replacing exact 0 values)
        elif metric_type == "timing":
            data = torch.where(data == 0.0, torch.ones_like(data) * 1e-20, data)  # Replace 0 values with small positive values
            tmp_data = (
                torch.where(data.isfinite(), data, torch.ones_like(data))  # Replace NaN values with 1
                .transpose(0, 1)
                .unsqueeze(-2)
            )

        # Handle binary and binarySpan metrics
        elif metric_type == "binary" or metric_type == "binarySpan":
            data = data.nansum(dim=-1).unsqueeze(-1)  # Sum the binary values across trials and add a new dimension
            data = torch.where(
                counts.unsqueeze(-1) == 0.0, torch.ones_like(data) * torch.nan, data
            )  # Set NaN values where there are no valid trials for a participant
            tmp_data = (
                torch.where(data.isfinite(), data, torch.zeros_like(data))  # Replace NaN values with 0
                .transpose(0, 1)
                .unsqueeze(-2)
            )
        # Raise an error for unsupported metric types
        else:
                raise RuntimeError("missing task")
        # create a mask for valid data (non-NaN values)
        mask = torch.logical_not(torch.isnan(data))  #  N x T
        data_dict[metric] = [
            data,
            tmp_data,
            counts,
            mask,
        ]
        
    return data_dict

# Function to prepare the data for model training or evaluation
def prepare_data(heldout_obs_ids=[], get_heldout_instead= False, normalize_times = True, remove_outliers = False):
    """
    Prepares the dataset by removing held-out observations or outliers and transforms the data into a suitable format 
    for machine learning models.
    
    Args:
    - heldout_obs_ids (list): List of observation IDs to hold out for validation or testing.
    - get_heldout_instead (bool): If True, returns held-out observations instead of the training set.
    - normalize_times (bool): Whether or not to normalize reaction times in the data.
    - remove_outliers (bool): If True, removes predefined outlier sessions from the dataset.
    
    Returns:
    - data_dict (dict): Transformed data for each metric, ready for use in ML models.
    - metrics (list): List of metrics used in the dataset.
    - participant_ids (list): List of participant IDs remaining after any removals.
    """
    # # prep data to get counts and handle boundary conditions
    data_dict = {}

    # Load the list of outlier sessions that should be held out if 'remove_outliers' is set to True.
    default_must_holdout = OUTLIER_HELDOUT_SESSIONS # outlier sessions that should not be used in training

     # If removing outliers, append them to the list of held-out sessions.
    if remove_outliers: heldout_obs_ids = heldout_obs_ids +default_must_holdout # must hold out if we are not returning the heldout set
    
    # Get the full data matrix, participant IDs, and metrics.
    full_data_matrix, participant_ids, metrics = get_session_data_matrix(heldout_obs_ids=heldout_obs_ids, get_heldout_instead= get_heldout_instead,normalize_times=normalize_times)
    
    # Transform the raw data into a format that the model can use.
    data_dict = transform_data(full_data_matrix,metrics=metrics) # transform data to prepare it for use in ML models
    return data_dict, metrics, participant_ids
   
# Function to get the full data matrix and hold out specific sessions if needed
def get_session_data_matrix(heldout_obs_ids=[], get_heldout_instead= False, normalize_times=True):
    """
    Retrieves the full data matrix, and if necessary, holds out specific sessions or returns only the held-out data.
    
    Args:
    - heldout_obs_ids (list): List of session IDs to be held out from the training data.
    - get_heldout_instead (bool): If True, returns only the held-out data.
    - normalize_times (bool): Whether or not to normalize reaction times in the data.

    Returns:
    - full_data_matrix (torch.Tensor): The data matrix with all sessions, possibly excluding held-out sessions.
    - remaining_ids (list): List of participant IDs for the remaining observations.
    - metrics (list): The metrics used in the data.
    """
    
    # get the full data matrix, metrics, and participant IDs
    full_data_matrix, metrics, participant_ids = get_full_data_matrix(normalize_times=normalize_times)

    # create a list of indices for observations to be held out
    heldout_idx = [idx for idx, id in enumerate(participant_ids) if id in heldout_obs_ids]

    # select the observations to be held out
    heldout_matrix = full_data_matrix[heldout_idx, :, :]

    # create a mask of indices for the remaining observations
    remaining_mask = torch.tensor([idx not in heldout_idx for idx in range(len(participant_ids))])

    # select the remaining observations using the mask
    remaining_matrix = full_data_matrix[remaining_mask, :, :]

    # create a list of participant IDs for the remaining observations using the mask
    remaining_ids = [id for idx, id in enumerate(participant_ids) if remaining_mask[idx]]
    
    if get_heldout_instead:
        # return the heldout matrix, remaining IDs, and metrics
        return heldout_matrix, heldout_obs_ids, metrics
    else:
        # return the remaining matrix, remaining IDs, and metrics
        return remaining_matrix, remaining_ids, metrics

def get_model_filepath(held_out_session_ids):
    """
    Generates a filepath for saving the model based on which sessions are held out during training.
    
    Args:
    - held_out_session_ids (list): List of session IDs that were held out during model training.
    
    Returns:
    - filenm (str): The filepath where the trained model will be saved.
    """
    if len(held_out_session_ids) ==0:
        heldout_id = "_none" # holding out no session
    elif len(held_out_session_ids) ==1:
        heldout_id = held_out_session_ids[0]
    else:
        heldout_id = "multi" # holding out more than 1 session
    
    filenm = f"../saved_models/{DATASET}/heldout_obs{heldout_id}"
    return filenm


def transform_boundary_value(metric, value):
    """
    Transforms boundary values for certain metrics, such as replacing exact 0s and 1s for beta metrics 
    or 0 values for timing metrics.
    
    Args:
    - metric (str): The name of the metric being transformed.
    - value (float): The raw value to be transformed.
    
    Returns:
    - (float): The transformed value, adjusted for boundary conditions.
    """
    metric_type = CURR_METRICS_DICT[metric]["type"]
    if metric_type == "beta":  # Exact 0 and 1's cause error for beta dist...
        if value == 0:
            return 0.00001 #Replace 0 with a small positive value
        elif value == 1:
            return 0.99999 #Replace 1 with a slightly smaller value
        else:
            return value
    elif metric_type == "timing":  # 0 inputs cause error in LogNormal/ inf lob prob
        if value == 0:
            return 1e-20 #Replace 0 with a small positive value 
        else:
            return value
    else:
        return value #No Transformation needed


def extract_update_w_data_dict(data_dict, metrics): 
    """
    Generates a data structure similar to 'update_w_data', preparing the transformed data for use in ML models.
    
    Args:
    - data_dict (dict): A dictionary containing transformed data for each metric.
    - metrics (list): List of metrics used in the dataset.
    
    Returns:
    - relevant_data_dict (dict): A dictionary containing the data for each metric.
    - relevant_data_dict_counts (dict): A dictionary containing the count of valid trials for each metric.
    """
    relevant_data_dict = {}  # Store the transformed data for each metric.
    relevant_data_dict_counts = {}  # Store the count of valid trials for each metric.

    # Initialize the lists for each metric.
    for metric in metrics:
        relevant_data_dict[metric] = []
        relevant_data_dict_counts[metric] = 0

    # Loop through each metric and process the data.
    for metric in metrics:
        data, _, counts, _ = data_dict[metric]
        data = data[torch.logical_not(torch.isnan(data))].tolist()

        # Special handling for binary metrics (converting to binary outcomes).
        if ("binary" in CURR_METRICS_DICT[metric]["type"]) and len(data) > 0:
            data_list = []
            for _ in range(int(data[0])):
                data_list.append(1.0) #Adds 1s for valid trials
            for _ in range(int(counts[0] - data[0])):
                data_list.append(0.0) #Adds 0s for invalid trials
            random.shuffle(data_list) #Shuffle data
            data = data_list
        relevant_data_dict[metric] += data  # Add the processed data to the relevant data dictionary.
        relevant_data_dict_counts[metric] = counts  # Update the count for each metric.

    return relevant_data_dict, relevant_data_dict_counts #Return the processed data and counts


def get_full_data_matrix(normalize_times=False):
    """
    Loads the full data matrix and participant IDs from the saved files.
    
    Args:
    - normalize_times (bool): Whether or not to normalize reaction times in the data.
    
    Returns:
    - full_data_matrix (torch.Tensor): The loaded data matrix.
    - RELEVANT_METRICS (list): List of relevant metrics.
    - ALL_PARTICIPANT_IDS (list): List of all participant IDs in the dataset.
    """
     # Load the full data matrix from the saved file.
    data_dir = PROJECT_HOME / "data" / DATASET
    matrix_path = data_dir / "full_data_matrix_not_normed.pt"
    participants_path = data_dir / "participant_ids_not_normed.csv"

    if not matrix_path.exists() or not participants_path.exists():
        _ensure_minimal_data_assets(CURR_METRICS_DICT, RELEVANT_METRICS)

    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Full data matrix file not found in '{matrix_path}'. "
            "Provide raw data or keep the generated placeholders available."
        )

    full_data_matrix = torch.load(matrix_path)

    if not participants_path.exists():
        raise FileNotFoundError(
            f"Participant IDs file not found in '{participants_path}'. "
            "Provide raw data or keep the generated placeholders available."
        )

    ALL_PARTICIPANT_IDS = pd.read_csv(participants_path)["ids"].tolist()
    return full_data_matrix, RELEVANT_METRICS, ALL_PARTICIPANT_IDS # Return the loaded data and participant IDs.

def mle_params_to_dist(
    metric, mle_dist_params, metric_type, counts=1, metrics_dict=CURR_METRICS_DICT, use_differentiable_sigmoid=False
):
    """
    Converts MLE (Maximum Likelihood Estimate) parameters to the distributional form required by each metric type.
    
    Args:
    - metric (str): The metric being processed.
    - mle_dist_params (list): The raw parameters output from the model.
    - metric_type (str): The type of the metric (e.g., binary, timing, beta).
    - counts (int): The count of valid trials (defaults to 1).
    - metrics_dict (dict): Dictionary containing metric-specific details.
    
    Returns:
    - (torch.distributions.Distribution): The distribution object for the given metric and parameters.
    """
     
    # assuming counts=1 when we want to try a single data point at a time
    # Note, we save raw dist params rather than fs that need activation for MLE
    # For binarySpan metrics, use a parameterized sigmoid function to transform the parameters.
    if metric_type == "binarySpan":
        psiTheta, psiSigma = torch.tensor(mle_dist_params[0]), torch.tensor(mle_dist_params[1])
        g, l = torch.tensor(0.02), torch.tensor(0.02)  # Default values for gamma and lambda.


        if use_differentiable_sigmoid:
            a, b,_,_= convert_sigmoid_params(psiTheta, psiSigma, gamma=g, lambda_=l, to="differentiable")
            print(f"Converted params from presentable to differentiable sigmoids params:(psiTheta ={psiTheta}, psiSigma ={psiSigma}) to (a={a}, b={b})")
            a,b,_,_ = convert_sigmoid_params(psiTheta, psiSigma, gamma=g, lambda_=l, to="differentiable")
            pred_probs = get_differentiable_sigmoid(a, b, g, l)(
                metrics_dict[metric]["length"]
            )
        else:
            pred_probs = get_presentable_sigmoid(psiTheta,psiSigma, g, l)(
                metrics_dict[metric]["length"]
            )
        mle_dist_params = [counts, pred_probs]
    # For binary metrics, return the distribution with the MLE parameters.
    elif metric_type == "binary":
        mle_dist_params = [counts, mle_dist_params[0]]

    elif metric_type == "timing":
        mle_dist_params = [torch.clamp(torch.tensor(param), min=1e-20) for param in mle_dist_params]

    return dist_dict[metric_type](*mle_dist_params)

def load_trained_model(latent_dim=3, model_path=None):
    """
    Loads a trained model for inference.
    Args:
        latent_dim (int): Dimension of the latent space (must match the trained model).
        model_path (str): Path to the trained model file (if not provided, inferred using a general file pattern).
    Returns:
        model: The loaded trained model.
    Raises:
        FileNotFoundError: If the model file is not found.
        ValueError: If the model path cannot be inferred or is not provided.
    """
    
    if model_path is None:  # No model path provided, infer the model path using a general file pattern
        def get_most_recent_file(file_paths):
            # Use max function to return the file path with the latest modification time
            return max(file_paths, key=lambda x: os.path.getmtime(x))

        # Infer the model path from a general file pattern
        filename = f"variationalNN_relevant_only_latentdim{latent_dim}*.pt"
        model_paths = glob.glob(f"{PROJECT_HOME_PATH}/saved_models/{DATASET}/heldout_obsmulti/{filename}")  # Assuming models are saved in a fixed folder

        if not model_paths:
            raise FileNotFoundError(f"No model files found matching pattern: {filename}")

        # If multiple models exist, select the most recent one
        model_path = get_most_recent_file(model_paths)
        print(f"Inferred model path: {model_path}")
    else:
        # Model path provided, use it directly
        print(f"Using provided model path: {model_path}")

    # Load the state_dict from the model path
    state_dict = torch.load(model_path, map_location=torch.device(COMPUTE_DEVICE))

    # Infer N from the state_dict (shape of 'meu_z' parameter)
    meu_z_shape = state_dict['meu_z'].shape  # This should give us a shape like (N, latent_dim)
    N = meu_z_shape[0]  # N is the first dimension of the tensor
    print(f"Inferred N (number of participants): {N}")

    # Infer model_output_dim from the state_dict (size of the output layer fc3's weight)
    fc3_weight_shape = state_dict['fc3.weight'].shape  # This should give us a shape like (model_output_dim, out_sz)
    model_output_dim = fc3_weight_shape[0]  # model_output_dim is the first dimension
    print(f"Inferred model_output_dim: {model_output_dim}")

    # Initialize the model using the inferred N and model_output_dim
    model = variationalNN(N, latent_dim, model_output_dim)

    # Load the model state
    model.load_state_dict(state_dict)
    model = model.eval()  # Set the model to evaluation mode (no training)

    # Make sure model params remain frozen
    for param in model.parameters():
        param.requires_grad = False

    model.to(COMPUTE_DEVICE)
    
    return model

def get_summarized_metric_details(metric):
    
    if metric in CURR_METRICS_DICT.keys():
        metric_type = CURR_METRICS_DICT[metric]["type"]
        # get the output indices of the model that are relevant for the metric
        fidxs = CURR_METRICS_DICT[metric]["f_idxs"]
        length = CURR_METRICS_DICT[metric]["length"]
    else: # if metric is not found in the metrics_dict i.e generalized metrics e.g Complex Span without a specific length
        metric_type = CURR_METRICS_DICT[f"{metric}_correct_w_len_2"]["type"]
        # get the output indices of the model that are relevant for the metric
        fidxs = CURR_METRICS_DICT[f"{metric}_correct_w_len_2"]["f_idxs"]
        length = None
        
    return metric_type, fidxs, length

def generate_model_name():
    nouns = [
        "Falcon", "Panther", "Hawk", "Tiger", "Eagle", "Phoenix", "Dragon", "Leopard", "Knight", "Viper",
        "Griffin", "Wolf", "Raven", "Cheetah", "Lion", "Ocelot", "Cobra", "Jaguar", "Serpent", "Bear",
        "Stallion", "Mustang", "Raptor", "Puma", "Tornado", "Cyclone", "Tsunami", "Thunder", "Blizzard", "Tempest",
        "Shark", "Orca", "Piranha", "Barracuda", "Manta", "Kraken", "Hydra", "Minotaur", "Centaur", "Pegasus",
        "Basilisk", "Chimera", "Sphinx", "Wyvern", "Cerberus", "Golem", "Yeti", "Mammoth", "Saber", "Scorpion",
        "Anaconda", "Komodo", "Gryphon", "Lynx", "Cougar", "Wolverine", "Badger", "Fox", "Hound", "Mongoose",
        "Panther", "Jaguar", "Lynx", "Cougar", "Wolverine", "Badger", "Fox", "Hound", "Mongoose", "Otter",
        "Beaver", "Ferret", "Weasel", "Mink", "Stoat", "Ermine", "Marten", "Fisher", "Sable", "Civet",
        "Genet", "Fossa", "Meerkat", "Hyena", "Jackal", "Dingo", "Coyote", "Wolf", "Bear", "Panda"
    ]
    
    verbs = [
        "Strike", "Run", "Fly", "Roar", "Charge", "Blaze", "Dash", "Glide", "Soar", "Dive",
        "Sprint", "Pounce", "Surge", "Flash", "Zoom", "Leap", "Rage", "Bolt", "Thrust", "Sweep",
        "Ascend", "Rush", "Storm", "Lunge", "Blast", "Thrive", "Ambush", "Pierce", "Unleash", "Ignite",
        "Climb", "Swoop", "Gallop", "Hurdle", "Skim", "Skate", "Slide", "Trek", "Vault", "Whirl",
        "Whiz", "Whip", "Whisk", "Whirlwind", "Zigzag", "Zoom", "Barrel", "Bound", "Burst", "Cannon",
        "Catapult", "Charge", "Chase", "Cruise", "Dart", "Dash", "Flee", "Gallop", "Hasten", "Hurtle",
        "Jet", "Lunge", "Plunge", "Race", "Rocket", "Rush", "Scamper", "Scurry", "Shoot", "Skim",
        "Sprint", "Pounce", "Surge", "Flash", "Zoom", "Leap", "Rage", "Bolt", "Thrust", "Sweep",
        "Ascend", "Rush", "Storm", "Lunge", "Blast", "Thrive", "Ambush", "Pierce", "Unleash", "Ignite",
        "Climb", "Swoop", "Gallop", "Hurdle", "Skim", "Skate", "Slide", "Trek", "Vault", "Whirl",
        "Whiz", "Whip", "Whisk", "Whirlwind", "Zigzag", "Zoom", "Barrel", "Bound", "Burst", "Cannon"
    ]
    
    # Generate a 4-digit time-based number (e.g., last 4 digits of current time in seconds)
    timestamp = int(time.time())  # Current time in seconds
    number = timestamp % 10000  # Get the last 4 digits
    # Create a new random generator instance
    local_random = random.Random()

    # Use the local_random instance to make choices
    noun = local_random.choice(nouns)
    verb = local_random.choice(verbs)

    model_name = f"{noun}-{verb}-{number:04d}".lower()  # Ensure the number is always 4 digits
    return model_name
