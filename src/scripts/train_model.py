import logging
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # Add parent directory to path for module access
import torch
import torch.nn as nn
from torch.utils.data import random_split
from utils.data_distribution_utils import (
    RELEVANT_METRICS,
    COMPUTE_DEVICE,
    DATASET,
    DEFAULT_HELDOUT_SET,
    RANDOM_SEED,
    generate_model_name,
    get_model_filepath,
    load_trained_model,
    prepare_data,
    create_metrics_dict,
    activation_dict,
    CURR_METRICS_DICT,
    activation_dict,
    dist_dict,
    prior_log_prob,
    extract_update_w_data_dict
)
from utils.grid_search_utils import compute_loss_landscape
from utils.variational_NN import variationalNN
from gpytorch.priors import NormalPrior
import numpy as np
import argparse
from utils.set_seed import set_seed
import warnings
import datetime
warnings.filterwarnings("ignore")  # Suppress warnings
import csv
import torch.optim.lr_scheduler as lr_scheduler
import pdb
import matplotlib.pyplot as plt
from torch.nn.functional import softplus
from utils.grid_search_utils import run_grid_search
CUR_LOCAL_FILE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))


def compute_log_prob(f, data_dict, metrics, metrics_dict, mean_log_probs, std_log_probs, 
                    log_timing_data=False, clamp_log_prob=False, max_log_prob=-1000, 
                    z_score_log_prob=False, param_prior_loss_factor=0.0):
    """
    Compute the log probability of the data given the model outputs.

    Args:
        f: Model outputs with shape [grid_size, batch_size, output_dim], where grid_size is 1 for training
           and can be larger (e.g., 10000) for validation/grid search.
        data_dict: Dictionary containing data tensors for each metric.
        metrics: List of metrics included in the model.
        metrics_dict: Dictionary with metric properties.
        mean_log_probs: Precomputed mean log probabilities for each metric.
        std_log_probs: Precomputed standard deviation of log probabilities for each metric.
        log_timing_data (bool): Whether to log transform timing data.
        clamp_log_prob (bool): Whether to clamp log probabilities.
        max_log_prob (float, optional): Maximum value for log probability clamping.
        z_score_log_prob (bool): Whether to z-score the log probabilities.

    Returns:
        total_log_prob_data: The total log probability of the data.
    """
    total_log_prob_data = torch.tensor(0.0)
    prior_param_loss = torch.tensor(0.0)
    for metric in metrics:
        metric_type = metrics_dict[metric]["type"]
        fidxs = metrics_dict[metric]["f_idxs"]
        _, tmp_data, counts, mask = data_dict[metric]
        
        # Skip if no valid data or zero standard deviation
        if (counts == 0).all() or std_log_probs[metric] == 0:
            continue
        
        # Skip binary span tasks with length > 10
        if metric_type == "binarySpan" and metrics_dict[metric]["length"] > 10:
            continue
        
        if log_timing_data and metric_type == "timing":
            # Apply log transformation to timing data if specified
            tmp_data = torch.log(tmp_data)
        
        # Apply activation functions to the model outputs to get valid distribution parameters
        dist_params = activation_dict[metric_type](
            f[:, :, fidxs], counts, metrics_dict[metric]["length"]
        )
        
        # Create the distribution object for the metric (e.g., Normal, Binomial)
        dist = dist_dict[metric_type](*dist_params)
        
        # Compute the log probability of the observed data under the model's predicted distribution
        log_prob_task = (
            dist.log_prob(tmp_data.to(COMPUTE_DEVICE)).transpose(0, 1).transpose(1, 2)
        )
        
        if clamp_log_prob and max_log_prob is not None:
            # Clamp log probabilities to prevent numerical issues (e.g., log(0))
            log_prob_task = torch.clamp(log_prob_task, min=max_log_prob)
        
        log_prob_task = log_prob_task.mean(axis=0) # average across samples
        log_prob_task = log_prob_task[mask] # remove masked values FIRST
        if z_score_log_prob:
            # Z-score the log probabilities to normalize them
            std = std_log_probs[metric] + 0.001  # avoid dividing by zero or very small values
            z_scored_lg_prob_task = (
                log_prob_task - mean_log_probs[metric]
            ) / std
            
            z_scored_lg_prob_task = z_scored_lg_prob_task.mean()
            # Accumulate the normalized log probability
            total_log_prob_data = (
                total_log_prob_data + z_scored_lg_prob_task
            )  # normalize log probabilities
        else:
            log_prob_task = log_prob_task.mean() # take mean AFTER masking
            total_log_prob_data = total_log_prob_data + log_prob_task
        # log_prob_task = log_prob_task.mean(axis=0)  # Average the log probabilities across samples new shape (N, 1)

        # if z_score_log_prob:
        #     # Z-score the log probabilities to normalize them
        #     std = std_log_probs[metric] + 0.001  # Avoid dividing by zero or very small values
        #     z_scored_lg_prob_task = (log_prob_task - mean_log_probs[metric]) / std
        #     log_prob_task = z_scored_lg_prob_task

        # log_prob_task = torch.nanmean(log_prob_task, dim=0) # shape (N, 1)
        # log_prob_task = log_prob_task.mean() # shape (1,)
        # total_log_prob_data += log_prob_task

        
        if param_prior_loss_factor>0: # compute the log probability of the parameter under the prior to the loss
            prior_param_log_prob = -prior_log_prob(softplus(f[:, :, fidxs]), metric_type).mean() #only positive parameters
            if prior_param_log_prob==torch.inf:
                print(metric_type,prior_param_log_prob)
            prior_param_loss+=prior_param_log_prob
             
    return total_log_prob_data, prior_param_loss


def compute_loss(model, data_dict, metrics, metrics_dict, mean_log_probs, std_log_probs, args, prior_x, is_training=True):
    """
    Compute the loss for the model.

    Args:
        model: The variational neural network model.
        data_dict: Dictionary containing data tensors for each metric.
        metrics: List of metrics included in the model.
        metrics_dict: Dictionary with metric properties.
        mean_log_probs: Precomputed mean log probabilities for each metric.
        std_log_probs: Precomputed standard deviation of log probabilities for each metric.
        args: Command-line arguments parsed by argparse.
        prior_x: Prior distribution over the latent variables.
        is_training: Boolean indicating if the computation is for training or validation.

    Returns:
        total_loss: The computed loss.
        total_log_prob_data: The total log probability of the data.
    """
    model.eval() if not is_training else model.train()  # Set model mode
    latent_dist = torch.distributions.Normal(
        model.meu_z, nn.functional.softplus(model.sigma_z)
    )
    latent_points = latent_dist.rsample((args.n_samples,))  # Sample latent variables shape (n_samples, N, latent_dim)
    f = model(latent_points.to(COMPUTE_DEVICE))  # Forward pass through the model
    kld = 0
    prior_param_loss = 0
    
    if is_training:
        total_log_prob_data, prior_param_loss = compute_log_prob(f, data_dict, metrics, metrics_dict, mean_log_probs, std_log_probs, log_timing_data=args.log_timing_data,
                                               clamp_log_prob=args.clamp_log_prob, max_log_prob=args.max_log_prob, z_score_log_prob=args.z_score_log_prob, param_prior_loss_factor=args.param_prior_loss_factor)
        # import pdb; pdb.set_trace()
        kld = torch.distributions.kl_divergence(latent_dist, prior_x).mean()  # Compute KL divergence
        total_loss = -total_log_prob_data + args.kld_factor * kld  # Compute total loss
        l2_norm = sum(p.pow(2).sum() for p in model.parameters())  # L2 regularization
        total_loss += args.l2_lambda * l2_norm
        total_loss += args.param_prior_loss_factor * prior_param_loss #add the prior loss to the total loss
        
    else:
        total_log_prob_data_sum = 0.0
        count = 0
        kld = torch.tensor(-1.0)

        for update_w_data in data_dict:
            log_prob_data,meu_z = run_grid_search(model, update_w_data, num_points=10)
            total_log_prob_data_sum += log_prob_data
            count += 1

        # Compute the average log probability
        average_log_prob_data = total_log_prob_data_sum / count if count > 0 else 0.0
        total_loss = average_log_prob_data

    return total_loss,kld, prior_param_loss



CUR_LOCAL_FILE_DIR =  os.path.abspath(os.path.join(os.path.dirname(__file__)))

def train_model(args):
    """
    Function to train the Distributional Latent Variable Model (DLVM) using variational inference.

    Args:
        args: Command-line arguments parsed by argparse.
    """
    # Load precomputed mean and standard deviation of log probabilities for each metric
    mean_log_probs = torch.load(os.path.join(CUR_LOCAL_FILE_DIR, f"../data/{DATASET}/mean_log_probabilities.pt"), weights_only=False)
    std_log_probs = torch.load(os.path.join(CUR_LOCAL_FILE_DIR, f"../data/{DATASET}/std_log_probabilities.pt"), weights_only=False)

    filenm = get_model_filepath(args.held_out_session_ids)
    
    if not os.path.exists(filenm):
        os.makedirs(filenm)
    
    # Prepare the training data, optionally holding out certain sessions (held_out_session_ids)
    data_dict, metrics, participant_ids = prepare_data(heldout_obs_ids=args.held_out_session_ids, remove_outliers=True) 

    metrics = RELEVANT_METRICS
    # Create a dictionary with metric properties and determine the output dimension of the model
    metrics_dict, model_output_dim = create_metrics_dict(use_relevant_metrics_only=True)
    # Get the number of participants (N) from the data
    N = list(data_dict.values())[0][0].shape[0]
    print("Training set Size", N)
    total_n_samples = args.n_samples

    # Split the heldout set into validation and test sets
    _, _, heldout_participant_ids = prepare_data(heldout_obs_ids=args.held_out_session_ids, get_heldout_instead=True)
    heldout_size = len(heldout_participant_ids)
    val_size = int(0.5 * heldout_size)
    test_size = heldout_size - val_size
    val_indices, test_indices = random_split(range(heldout_size), [val_size, test_size])
    vals_update_w_data_list = []

    # log the Ns used for training and heldout to the log file
    logging.info(f"Training set Size: {N}, Validation set Size: {val_size}, Test set Size: {test_size}")

    # Ensure participant_ids is defined and matches heldout_participant_ids
    participant_ids = heldout_participant_ids  # Assuming participant_ids should be the same as heldout_participant_ids

    # Use only the validation indices for validation
    for idx in val_indices:
        session = participant_ids[idx]
        # print(f"Session {session} is being used for validation")
        
        # Prepare data for the current session
        held_data_dict, metrics, _ = prepare_data(heldout_obs_ids=[session], get_heldout_instead=True)
        
        # Preprocess the data
        update_w_data, _ = extract_update_w_data_dict(data_dict=held_data_dict, metrics=metrics)

        vals_update_w_data_list.append(update_w_data)

    # Initialize the variational neural network model
    model = variationalNN(N, args.latent_dim, model_output_dim  ).to(COMPUTE_DEVICE)
  
    # Define the optimizer (Adam optimizer) for training the model parameters
    optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=args.lr)
    # Define the prior distribution over the latent variables (standard normal prior)
    prior_x = NormalPrior(
        torch.zeros(N, args.latent_dim), torch.ones(N, args.latent_dim)
    ).to(COMPUTE_DEVICE)

    string_id = args.model_id
    loss_list = []  # List to keep track of loss values during training
    val_loss_list = []  # List to keep track of validation loss values
    lowest_val_loss = np.inf  # Initialize the lowest loss encountered
    # Define a learning rate scheduler to reduce the learning rate at specified milestones
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[args.n_epochs*0.5, args.n_epochs*0.75], gamma=0.5)
    best_model = model.state_dict()

    # Training loop over epochs
    patience = int(args.n_epochs*0.5)  # Number of epochs to wait for improvement
    patience_counter = 0  # Counter to track epochs without improvement

    for e in range(args.n_epochs):
        optimizer.zero_grad()  # Zero out gradients from previous iteration
        train_loss, kld, prior_param_loss = compute_loss(model, data_dict, metrics, metrics_dict, mean_log_probs, std_log_probs, args, prior_x, is_training=True)
        train_loss.backward()  # Backpropagate the loss to compute gradients
        optimizer.step()  # Update the model parameters
        scheduler.step()  # Update the learning rate if using a scheduler
        loss_list.append(train_loss.item())  # Append the current loss to the loss list

        # Validation
        val_loss, _,_= compute_loss(model, vals_update_w_data_list, metrics, metrics_dict, mean_log_probs, std_log_probs, args, prior_x, is_training=False)
        val_loss_list.append(val_loss.item())  # Append the validation loss

        if args.verbose:
            logging.info(f"epoch: {e + 1}, train_loss: {train_loss.item():.4f}, kld: {kld.item():.4f},param_prior_prob: {prior_param_loss.item():.4f}, val_loss: {val_loss.item():.4f}")

        if val_loss.item() < lowest_val_loss:
            lowest_val_loss = val_loss.item()
            best_model = model.state_dict()
            patience_counter = 0  # Reset the counter if there is an improvement
        else:
            patience_counter += 1  # Increment the counter if no improvement

        # Check if patience has been exceeded
        if patience_counter >= patience:
            logging.info(f"Early stopping at epoch {e + 1} due to no improvement in validation loss for {patience} epochs.")
            break

    pt_file_name = f"/variationalNN_relevant_only_latentdim{args.latent_dim}_" + string_id + ".pt"
    
    torch.save(
        best_model,
        filenm + pt_file_name
    )
    csv_file_path = os.path.join(CUR_LOCAL_FILE_DIR, f'../model_training_analysis/{DATASET}/{args.run_mode}s_data.csv')

    new_record = {
        "model_id": string_id,
        "time_stamp": datetime.datetime.now(),
        'latent_dim': args.latent_dim,
        'param_prior_factor': args.param_prior_loss_factor,
        'held_out_session_ids': participant_ids,
        'val_held_out_session_ids': [participant_ids[idx] for idx in val_indices],
        'dataset': DATASET,
        'lr': args.lr,
        'kld_factor': args.kld_factor,
        'file_name': pt_file_name,
        'final_avg_log_prob': train_loss.item(),
        'final_avg_log_prob_val': val_loss.item(),
        'total_count': N,
        'final_kld': kld.item(),
        "all_runtime_args": vars(args)
    }

    update_csv_file(csv_file_path, new_record)
    save_loss_plot(loss_list, val_loss_list, new_record)
    
    if args.verbose:
        print(f"Training complete! Model saved to {pt_file_name}")

def save_data_to_csv(file_path, data):
    """
    Save data to a CSV file.
    If the file doesn't exist, create a new file and save the data.

    Args:
        file_path (str): Path to the CSV file.
        data (list): List of dictionaries containing data to be saved.
    """
    fieldnames = data[0].keys() if data else []
    file_exists = os.path.exists(file_path)
    
    if not file_exists:
        # Create a new CSV file and write the header
        with open(file_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
    with open(file_path, 'a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerows(data)

def update_csv_file(file_path, new_record):
    """
    Update a CSV file with a new record.
    If the file doesn't exist, create a new file and save the new record.
    If the file exists, append the new record to the existing data.

    Args:
        file_path (str): Path to the CSV file.
        new_record (dict): Dictionary containing the new record to be added.
    """
    if not os.path.isfile(file_path):
        save_data_to_csv(file_path, [new_record])
    else:
        save_data_to_csv(file_path, [new_record])

def save_loss_plot(loss_list, val_loss_list, training_params):
    """
    Save plots of the training and validation loss on both natural and log scales.

    Parameters:
    - loss_list: List of loss values recorded during training.
    - val_loss_list: List of validation loss values recorded during training.
    - training_params: Dictionary containing training parameters and details.
    """
    model_id = training_params.get("model_id", "")
    latent_dim = training_params.get('latent_dim', 'N/A')
    held_out_session_ids = training_params.get('held_out_session_ids', 'N/A')
    dataset = training_params.get('dataset', 'N/A')
    lr = training_params.get('lr', 'N/A')
    kld_factor = training_params.get('kld_factor', 'N/A')
    
    folder_path = os.path.join(CUR_LOCAL_FILE_DIR, f"../visualization/outputs/{DATASET}/{model_id}")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    file_name_natural = f"{folder_path}/training_loss_natural"
    file_name_log = f"{folder_path}/training_loss_log"

    # Plot on natural scale
    plt.figure(figsize=(10, 6))
    plt.plot(loss_list, label=f"Training Loss ({loss_list[-1]:.4f})")
    plt.plot(val_loss_list, label=f"Validation Loss ({val_loss_list[-1]:.4f})")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(
        f'{model_id} - Training and Validation Loss (Natural Scale)\n'
        f'Dataset: {dataset}, Latent Dim: {latent_dim}, LR: {lr}, KLD Factor: {kld_factor}\n',
        fontsize=14
    )
    plt.legend()
    plt.grid(True)
    plt.savefig(file_name_natural + ".pdf")
    plt.savefig(file_name_natural + ".png")
    plt.close()

    # Plot on log scale
    plt.figure(figsize=(10, 6))
    plt.plot(loss_list, label=f"Training Loss ({loss_list[-1]:.4f})")
    plt.plot(val_loss_list, label=f"Validation Loss ({val_loss_list[-1]:.4f})")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')  # Set y-axis to log scale
    plt.title(
        f'{model_id} - Training and Validation Loss (Log Scale)\n'
        f'Dataset: {dataset}, Latent Dim: {latent_dim}, LR: {lr}, KLD Factor: {kld_factor}\n',
        fontsize=14
    )
    plt.legend()
    plt.grid(True)
    plt.savefig(file_name_log + ".pdf")
    plt.savefig(file_name_log + ".png")
    plt.close()

def run(args):
    """
    Main function to run the training process.

    Args:
        args: Command-line arguments parsed by argparse.
    """
    print("Heldout IDS", args.held_out_session_ids, "Latent Dim", args.latent_dim, "KLD factor", args.kld_factor, "lr", args.lr)
    if args.run_mode == "debug":
        args.n_epochs = 100
        train_model(args)
        print("Looks like things are working! Go ahead and turn off debug mode")
        sys.exit()
    if args.verbose:
        print(f"Starting model {args.model_id} with heldout session IDs: {args.held_out_session_ids}")
    train_model(args)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--held_out_session_ids", nargs='*', type=str, help='List of strings', default=DEFAULT_HELDOUT_SET)
    parser.add_argument("--latent_dim", type=int, default=3)
    parser.add_argument("--n_epochs", type=int, default=8_000)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.008)
    parser.add_argument("--kld_factor", type=float, default=0.01)
    parser.add_argument('--run_mode', type=str, default="debug", choices=['debug', 'run'])
    parser.add_argument("--save_best_model", type=bool, default=True)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--z_score_log_prob", type=bool, default=True)
    parser.add_argument("--log_timing_data", type=bool, default=False)
    parser.add_argument("--clamp_log_prob", type=bool, default=False)
    parser.add_argument("--max_log_prob", type=int, default=-3000)
    parser.add_argument("--use_param_prior_loss", type=bool, default=False)
    parser.add_argument("--param_prior_loss_factor", type=float, default=0.0)
    parser.add_argument("--l2_lambda", type=float, default=0.0)
    parser.add_argument("--n_hidden_layers", type=int, default=2)
    args = parser.parse_args()
    args.model_id = generate_model_name()
    set_seed(RANDOM_SEED)
    args.ramdom_seed = RANDOM_SEED
    
    args.vis_output_dir = os.path.join(CUR_LOCAL_FILE_DIR, f"../visualization/outputs/{DATASET}/{args.model_id}")
    if not os.path.exists(args.vis_output_dir):
        os.makedirs(args.vis_output_dir)

    log_file = f"{args.vis_output_dir}/{args.model_id}_training_log.txt"
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args.init_from_scratch = True if args.latent_dim != 3 else False
    logging.info("Training configuration:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")
    
    run(args)
    
