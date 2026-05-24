from sympy import per
import os, sys, torch, math, re
import numpy as np
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

# --- Project-specific Imports & Constants ---
from nmil_dlvm.paths import ANALYSIS_ARTIFACTS_ROOT
from nmil_dlvm.utils.set_seed import set_seed
from nmil_dlvm.utils.data_distribution_utils import (
    DATASET, RANDOM_SEED, COMPUTE_DEVICE, load_trained_model, CURR_METRICS_DICT,
    activation_dict, dist_dict
)

set_seed(RANDOM_SEED)

# --- Build paths to data files ---
synthetic_data_dir = ANALYSIS_ARTIFACTS_ROOT / "dlvm_imle_comparison" / "synthetic_data" / DATASET

def load_simulated_data(file_path: str) -> dict:
    """Load data from a .pt file."""
    if not os.path.exists(file_path):
        print(f"Data file not found at {file_path}")
        raise FileNotFoundError(f"Data file not found at {file_path}")
    try:
        data = torch.load(file_path, weights_only=False, map_location=COMPUTE_DEVICE)
        print(f"Successfully loaded data from '{file_path}' with {len(data)} runs.")
        return data
    except Exception as e:
        print(f"Failed to load data from '{file_path}': {e}")
        raise

def run_optimization_search(model, update_w_data, num_restarts = 10):
    """
    Runs an optimization search to find the best latent points for the data in update_w_data.
    Uses update_latent_dist_from_data to optimize the latent position.
    
    Args:
        model: The DLVM model used for optimization
        update_w_data (dict): A dictionary containing the data for each metric
        num_restarts (int): Number of initial points (used for initialization)
    
    Returns:
        tuple: (optimized_loss, best_meu_z) - The final loss and optimized latent position
    """
    # Initialize latent parameters 
    latent_dim = model.meu_z.shape[-1]
    
    # Get dimensions and current distribution parameters
    mean_meu_z = model.meu_z.mean(dim=0) # shape (latent_dim)
    mean_sigma_z = model.sigma_z.mean(dim=0) # shape (latent_dim)

    # Initialize parameters based on number of restarts - ensure they're on COMPUTE_DEVICE
    meu_z_init = torch.zeros(max(1, num_restarts), latent_dim, device=COMPUTE_DEVICE)
    sigma_z_init = torch.ones(max(1, num_restarts), latent_dim, device=COMPUTE_DEVICE) * 0.545

    # First position always uses current mean
    meu_z_init[0] = mean_meu_z.to(COMPUTE_DEVICE)
    sigma_z_init[0] = mean_sigma_z.to(COMPUTE_DEVICE)

    # For multiple restarts, sample remaining positions uniformly from model's learned range
    if num_restarts > 1:
        min_meu_z = model.meu_z.min(dim=0)[0].to(COMPUTE_DEVICE)
        max_meu_z = model.meu_z.max(dim=0)[0].to(COMPUTE_DEVICE)
        meu_z_init[1:] = min_meu_z.unsqueeze(0) + (max_meu_z - min_meu_z).unsqueeze(0) * torch.rand(num_restarts - 1, latent_dim, device=COMPUTE_DEVICE)

    # Make parameters require gradients for optimization - ensure they're on COMPUTE_DEVICE
    meu_z = torch.nn.Parameter(meu_z_init.clone().to(COMPUTE_DEVICE))
    sigma_z = torch.nn.Parameter(sigma_z_init.clone().to(COMPUTE_DEVICE))
    
    # Optimization parameters
    max_epochs = 500
    lr = 0.01
    max_n_progress_fails = 200
    n_samples = 20
    grad_clip = 0.2
    min_allowed_log_prob = -3000
    
    # Run the optimization
    try:
        optimized_dist, best_meu_z, best_sigma_z, lowest_loss, per_metric_losses = update_latent_dist_from_data_with_restarts(
            update_w_data=update_w_data,
            max_epochs=max_epochs,
            lr=lr,
            model=model,
            max_n_progress_fails=max_n_progress_fails,
            meu_z=meu_z,
            sigma_z=sigma_z,
            n_samples=n_samples,
            grad_clip=grad_clip,
            min_allowed_log_prob=min_allowed_log_prob,
            metrics_dict=CURR_METRICS_DICT
        )

        agg_losses = get_mean_across_lengths(per_metric_losses)

        return lowest_loss, best_meu_z, agg_losses

    except Exception as e:
        print(f"Optimization failed: {e}")
        # Return original position and a high loss if optimization fails
        return float('inf'), None, None
    
def get_mean_across_lengths(per_metric_losses):
    """
    Aggregate losses across different lengths for metrics that have multiple lengths.
    
    Args:
        per_metric_losses (dict): Dictionary of metric names and their losses.
        
    Returns:
        dict: Aggregated losses across lengths for each metric.
    """
    task_loss_lists = defaultdict(list)

    for mname, loss in per_metric_losses.items():
        loss = float(loss)

        # --- CorsiComplex: aggregate across lengths -----------------
        if re.match(r"CorsiComplex_correct_w_(?:len|length)_\d+$", mname):
            task_loss_lists["CorsiComplex"].append(loss)
            continue

        # --- SimpleSpan: aggregate across lengths -------------------
        if re.match(r"SimpleSpan_correct_w_(?:len|length)_\d+$", mname):
            task_loss_lists["SimpleSpan"].append(loss)
            continue

        task_loss_lists[mname].append(loss)

    return {
        task: (sum(vals) / len(vals)) if len(vals) > 1 else vals[0]
        for task, vals in task_loss_lists.items()
    }

def update_latent_dist_from_data_with_restarts(
    update_w_data,  # dictionary of metric names and data
    max_epochs,  # maximum number of epochs to run the optimization for
    lr,  # learning rate for the Adam optimizer
    model,  # function that maps latent points to outputs
    max_n_progress_fails,  # maximum number of epochs where loss does not improve before stopping optimization
    meu_z,  # tensor representing the initial mean(s) of the latent normal distribution(s)
    sigma_z,  # tensor representing the initial standard deviation(s) of the latent normal distribution(s)
    n_samples=10,  # number of samples to draw from the latent distribution at each optimization step
    grad_clip=0.2,  # maximum allowed magnitude of gradients before clipping
    min_allowed_log_prob=-1000,  # minimum allowed log probability before clamping
    metrics_dict=None,  # dictionary of metric names and information dictionaries
):
    """
    Update latent normal distribution(s) using data with support for multiple random restarts.
    
    This function extends update_latent_dist_from_data to handle multiple restarts in parallel.
    When meu_z and sigma_z have shape [num_restarts, latent_dim], it runs num_restarts 
    optimizations in parallel and returns the best result. For backward compatibility,
    if inputs have shape [latent_dim], they are reshaped to [1, latent_dim].
        
    INPUTS:
    - update_w_data: dictionary of metric names and data
    - max_epochs: maximum number of epochs to run the optimization for
    - lr: learning rate for the Adam optimizer
    - model: function that maps latent points to outputs
    - max_n_progress_fails: maximum number of epochs where loss does not improve before stopping optimization
    - meu_z: tensor representing the initial mean(s) of the latent normal distribution(s)
             Shape: [num_restarts, latent_dim] or [latent_dim] (will be reshaped)
    - sigma_z: tensor representing the initial standard deviation(s) of the latent normal distribution(s)
               Shape: [num_restarts, latent_dim] or [latent_dim] (will be reshaped)
    - n_samples: number of samples to draw from each latent distribution at each optimization step
    - grad_clip: maximum allowed magnitude of gradients before clipping
    - min_allowed_log_prob: minimum allowed log probability before clamping
    - metrics_dict: dictionary of metric names and information dictionaries

    OUTPUTS:
    - latent_dist: updated latent normal distribution (best across all restarts)
    - best_meu_z: tensor representing the mean of the best latent normal distribution
    - best_sigma_z: tensor representing the standard deviation of the best latent normal distribution
    - lowest_loss: loss value of the best restart
    """
    
    # Handle input shape compatibility - reshape if needed
    if meu_z.dim() == 1:
        meu_z = meu_z.unsqueeze(0)  # [latent_dim] -> [1, latent_dim]
    if sigma_z.dim() == 1:
        sigma_z = sigma_z.unsqueeze(0)  # [latent_dim] -> [1, latent_dim]
    
    
    num_restarts, latent_dim = meu_z.shape
    assert sigma_z.shape == (num_restarts, latent_dim), f"sigma_z shape {sigma_z.shape} doesn't match meu_z shape {meu_z.shape}"
    
    # Convert to parameters for optimization
    meu_z_param = torch.nn.Parameter(meu_z.clone().detach())
    sigma_z_param = torch.nn.Parameter(sigma_z.clone().detach())
    
    # Create optimizer for all restarts
    optimizer = torch.optim.Adam([meu_z_param, sigma_z_param], lr=lr)
    
    # Track progress for each restart
    progress_fails = torch.zeros(num_restarts, dtype=torch.int)
    lowest_losses = torch.full((num_restarts,), np.inf)
    best_meu_zs = meu_z_param.clone()
    best_sigma_zs = sigma_z_param.clone()
    
    # Global best tracking
    global_best_loss = np.inf
    global_best_restart_idx = 0
    global_best_metric_losses = {}       # NEW: {metric: float}
    
    # Training loop
    for e in range(max_epochs):
        optimizer.zero_grad()

        per_metric_losses_this_epoch = {} # NEW: Track losses per metric for this epoch
        
        # Create latent normal distributions for all restarts
        latent_dists = torch.distributions.Normal(
            meu_z_param, torch.nn.functional.softplus(sigma_z_param)
        )
        
        # Draw samples from all latent distributions
        # latent_points shape: [n_samples, num_restarts, latent_dim]
        latent_points = latent_dists.rsample((n_samples,))
        
        # Reshape for model forward pass
        batch_size = n_samples * num_restarts
        latent_points_flat = latent_points.reshape(batch_size, latent_dim)
        
        # Pass through model
        f_flat = model(latent_points_flat.to(COMPUTE_DEVICE))
        output_dim = f_flat.shape[-1]
        
        # Reshape back to [n_samples, num_restarts, output_dim]
        f = f_flat.reshape(n_samples, num_restarts, output_dim)
        
        # Initialize log probabilities for each restart
        total_log_prob_data = torch.zeros(num_restarts, device=COMPUTE_DEVICE)
        norm_const = 0
        
        # Process each metric
        for metric, data in update_w_data.items():
            if len(data) > 0:
                # Convert data to tensor
                data = torch.tensor(data).float()
                
                # Get metric information
                metric_type = metrics_dict[metric]["type"]
                fidxs = metrics_dict[metric]["f_idxs"]
                
                # Update normalization constant
                norm_const += len(data)
                
                # Get distribution parameters for all restarts
                counts = torch.tensor(data.shape[0]).reshape(1)
                dist_params = activation_dict[metric_type](
                    f[:, :, fidxs], counts, metrics_dict[metric]["length"]
                )
                
                # Create distributions for all restarts
                dist = dist_dict[metric_type](*dist_params)
                
                # Handle binary and binarySpan metrics the same way
                if metric_type.startswith("binary"):
                    data_sum = data.sum()
                    data_expanded = data_sum.expand(num_restarts)
                    
                    # Compute log probabilities for all restarts
                    probs = dist.log_prob(data_expanded.to(COMPUTE_DEVICE))
                    
                    # Clamp probabilities
                    probs = torch.clamp(probs, min=min_allowed_log_prob)
                    
                    # For binary metrics, handle different distribution batch shapes
                    if probs.dim() == 2:
                        # binarySpan case: probs has shape [n_samples, num_restarts]
                        log_prob_task = probs.mean(dim=0)
                    elif probs.dim() == 1 and probs.shape[0] == n_samples:
                        # pure binary case: probs has shape [n_samples], average to get scalar
                        log_prob_task = probs.mean()
                        # Ensure it has shape [num_restarts] for broadcasting
                        log_prob_task = log_prob_task.expand(num_restarts)
                    else:
                        # Fallback: assume it's already the right shape
                        log_prob_task = probs.mean(dim=0)
                
                elif metric_type == "timing":
                    # For timing metrics, we need to handle the batch shape correctly
                    # The distribution has batch_shape [n_samples, num_restarts]
                    # For each data point, we need to evaluate against all samples and restarts
                    
                    total_log_prob = 0
                    for data_point in data:
                        # Evaluate this data point against all samples and restarts
                        # data_point is scalar, expand to [n_samples, num_restarts]
                        data_expanded = data_point.expand(n_samples, num_restarts)
                        
                        # Compute log probabilities - dist expects [n_samples, num_restarts] 
                        point_probs = dist.log_prob(data_expanded.to(COMPUTE_DEVICE))
                        
                        # Clamp probabilities
                        point_probs = torch.clamp(point_probs, min=min_allowed_log_prob)
                        
                        # Sum over this data point: [n_samples, num_restarts] -> [num_restarts]
                        total_log_prob += point_probs.sum(dim=0)
                    
                    # Average over samples: [num_restarts]
                    log_prob_task = total_log_prob / n_samples
                    
                else:
                    # For other metric types (e.g., beta), handle similarly to timing
                    # The distribution has batch_shape [n_samples, num_restarts]
                    # For each data point, we need to evaluate against all samples and restarts
                    
                    total_log_prob = 0
                    for data_point in data:
                        # Evaluate this data point against all samples and restarts
                        # data_point is scalar, expand to [n_samples, num_restarts]
                        data_expanded = data_point.expand(n_samples, num_restarts)
                        
                        # Compute log probabilities - dist expects [n_samples, num_restarts] 
                        point_probs = dist.log_prob(data_expanded.to(COMPUTE_DEVICE))
                        
                        # Clamp probabilities
                        point_probs = torch.clamp(point_probs, min=min_allowed_log_prob)
                        
                        # Sum over this data point: [n_samples, num_restarts] -> [num_restarts]
                        total_log_prob += point_probs.sum(dim=0)
                    
                    # Average over samples: [num_restarts]
                    log_prob_task = total_log_prob / n_samples

                per_metric_losses_this_epoch[metric] = -log_prob_task # shape [num_restarts]

                # Add to total log probability for each restart
                total_log_prob_data += log_prob_task
        
        # Normalize by number of data points
        norm_const = max(1, norm_const)
        total_log_prob_data = total_log_prob_data / norm_const
        
        per_metric_losses_this_epoch = {
            metric: loss / norm_const for metric, loss in per_metric_losses_this_epoch.items()
        }

        # Compute losses for all restarts
        losses = -total_log_prob_data
        
        # Backward pass (sum all losses for gradient computation)
        total_loss = losses.sum()
        total_loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_([meu_z_param, sigma_z_param], grad_clip)
        
        # Update parameters
        optimizer.step()
        
        # Update best parameters for each restart
        current_losses = losses.detach().cpu().numpy()
        
        for restart_idx in range(num_restarts):
            current_loss = current_losses[restart_idx]
            
            if current_loss < lowest_losses[restart_idx]:
                # Update best for this restart
                lowest_losses[restart_idx] = torch.tensor(current_loss)
                best_meu_zs[restart_idx] = meu_z_param[restart_idx].clone()
                best_sigma_zs[restart_idx] = sigma_z_param[restart_idx].clone()
                progress_fails[restart_idx] = 0
                
                # Update global best if this is the best overall
                if current_loss < global_best_loss:
                    global_best_loss = float(current_loss)
                    global_best_restart_idx = restart_idx
                    
                    # NEW:
                    global_best_metric_losses = {
                        metric: per_metric_losses_this_epoch[metric][restart_idx]
                        for metric in per_metric_losses_this_epoch
                    }
            else:
                progress_fails[restart_idx] += 1
        
        # Check if all restarts have failed to improve for too long
        if torch.all(progress_fails > max_n_progress_fails):
            break
    
    # Return the globally best result
    best_meu_z_final = best_meu_zs[global_best_restart_idx]
    best_sigma_z_final = best_sigma_zs[global_best_restart_idx]
    
    # Ensure output shapes are consistent with original function [1, latent_dim]
    if best_meu_z_final.dim() == 1:
        best_meu_z_final = best_meu_z_final.unsqueeze(0)  # [latent_dim] -> [1, latent_dim]
    if best_sigma_z_final.dim() == 1:
        best_sigma_z_final = best_sigma_z_final.unsqueeze(0)  # [latent_dim] -> [1, latent_dim]
    
    return (
        torch.distributions.Normal(
            best_meu_z_final, torch.nn.functional.softplus(best_sigma_z_final)
        ),
        best_meu_z_final,
        best_sigma_z_final,
        global_best_loss,
        global_best_metric_losses
    )
