# this file contains functions for generating a grid of latent points and computing the log probability loss for those points
import os
import numpy as np
import torch
import sys
from utils.data_distribution_utils import (
    COMPUTE_DEVICE, DATASET, SUMMARIZED_METRICS, convert_sigmoid_params, RELEVANT_METRICS, RANDOM_SEED,
    get_summarized_metric_details,CURR_METRICS_DICT, activation_dict, dist_dict, SUMMARIZED_METRICS_METRIC_TYPES, set_seed
)

set_seed(RANDOM_SEED)

# create global path to this file i.e grid_search_utils.py
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))

def generate_grid(model, num_points):
    """
    Generates a grid of points within the range of the model's meu_z parameter.
    Args:
        model (torch.nn.Module): The model containing the meu_z parameter.
        num_points (int): The number of points to generate along each dimension.
    Returns:
        torch.Tensor: A tensor of shape (num_points^d, d) containing the grid points,
                      where d is the dimensionality of meu_z.
    """
    with torch.no_grad():
        # Ensure meu_z is on CPU and in float format for numpy operations
        meu_z_cpu = model.meu_z.cpu()
        
        # Compute min and max values along each dimension
        min_meuz = torch.min(meu_z_cpu, dim=0)[0]
        max_meuz = torch.max(meu_z_cpu, dim=0)[0]
        
        # Get dimensionality of the latent space
        d = min_meuz.shape[0]
        
        # Create coordinate tensors more efficiently using list comprehension
        coordinate_tensors = [
            torch.linspace(min_meuz[i], max_meuz[i], num_points)
            for i in range(d)
        ]

        # Create meshgrid with indexing='ij' for consistent behavior
        grid_tensors = torch.meshgrid(*coordinate_tensors, indexing='ij')
        
        # Stack and reshape in one operation
        grid_reshaped = torch.stack(grid_tensors, dim=-1).reshape(-1, d)
        
        return grid_reshaped

def run_grid_search(model, update_w_data, num_points = 100):
    """
    Runs a grid search to find the best latent position for the given data.
    
    Args:
        model: The DLVM model
        update_w_data (dict): Data for each metric
        num_points (int): Number of grid points per dimension
    
    Returns:
        tuple: (best_loss, best_meu_z) - The best loss and corresponding latent position
    """
    meu_z = generate_grid(model, num_points = num_points).to(COMPUTE_DEVICE).unsqueeze(1)
    # randomize the order of the points
    meu_z = meu_z[torch.randperm(meu_z.shape[0])] # randomize the order of the points to avoid any bias in the order of the points
    loss = log_prob_loss(meu_z, model=model, update_w_data=update_w_data)
    
    # Find the best loss and corresponding position
    best_idx = torch.argmin(loss)
    best_loss = loss[best_idx]
    best_meu_z = meu_z[best_idx]
    
    return best_loss, best_meu_z

def optimize_latent_position(model, update_w_data, **kwargs):
    """
    Convenience function to optimize latent position using update_latent_dist_from_data.
    
    Args:
        model: The DLVM model
        update_w_data (dict): Data for each metric
        **kwargs: Optional parameters for optimization (max_epochs, lr, etc.)
    
    Returns:
        tuple: (final_loss, optimized_meu_z, optimized_sigma_z, latent_distribution)
    """
    return run_optmization_search(model, update_w_data, **kwargs)

def compute_loss_landscape(model, update_w_data, num_points =100, custom_grid = None,randomize_order = True):
    if custom_grid is None:
        meu_z = generate_grid(model, num_points = num_points).to(COMPUTE_DEVICE).unsqueeze(1)
    else:
        meu_z = custom_grid.to(COMPUTE_DEVICE).unsqueeze(1)
    # randomize the order of the points
    if randomize_order:
        meu_z = meu_z[torch.randperm(meu_z.shape[0])] # randomize the order of the points to avoid any bias in the order of the points
    loss = log_prob_loss(meu_z, model =model, update_w_data = update_w_data)
    return loss, meu_z

def log_prob_loss(meu_z, model = None, update_w_data = None, min_allowed_log_prob=-3000, model_type ="NN", model_output_dim=12):
    
    mean_log_probs = torch.load(os.path.join(BASE_PATH, f"../data/{DATASET}/mean_log_probabilities.pt"),weights_only=False)
    std_log_probs = torch.load(os.path.join(BASE_PATH, f"../data/{DATASET}/std_log_probabilities.pt"),weights_only=False)
    
    f = compute_predictions_fom_latent_points(meu_z,model,model_output_dim, model_type =model_type)  
    
    if f.dim() == 2:
        f = f.unsqueeze(0)  # Add a dimension at index 0 if f is 2D - this is for convinience to work with code below
    
    # initialize log probability of data under the latent distribution to 0
    total_log_prob_data =torch.zeros(f.shape[1]).to(f.device)
    # import pdb

    # for each metric in the update data
    for metric in update_w_data.keys():
        # get data for the metric
        data = update_w_data[metric]

        # if there is data for the metric
        if len(data) > 0:
            
            # convert data to a tensor
            data = torch.tensor(data).float().unsqueeze(-1).unsqueeze(-1)

            # get the type of distribution to use for the metric
            metric_type = CURR_METRICS_DICT[metric]["type"]

            # get the output indices of the model that are relevant for the metric
            fidxs = CURR_METRICS_DICT[metric]["f_idxs"]

            # get the parameters for the distribution for the metric
            counts = torch.tensor(data.shape[0]).reshape(1)
            
            dist_params = activation_dict[metric_type](f[:, :, fidxs], counts, CURR_METRICS_DICT[metric]["length"])

            # create the distribution for the metric using the parameters
            dist = dist_dict[metric_type](*dist_params)
            
            # if the metric is binary, sum the data
            if metric_type.startswith("binary"):
                data = data.sum()

            # compute the log probability of the data under the distribution
            probs = dist.log_prob(data.to(COMPUTE_DEVICE))

            # # clamp the log probabilities to the minimum allowed value
            # probs = torch.clamp(probs, min=min_allowed_log_prob)

            # compute the log probability of the data for this metric
            if probs.dim()==2:
                log_prob_task = (probs.sum(axis=0) - mean_log_probs[metric])/(std_log_probs[metric]+0.001)
            else:
                log_prob_task = (probs - mean_log_probs[metric])/(std_log_probs[metric]+0.001)

            if not metric_type.startswith("binary"): 
                log_prob_task = log_prob_task.mean(0).squeeze(0) #average over data points
            
            # add the log probability of the data for this metric to the total log probability
            total_log_prob_data += log_prob_task
    return -total_log_prob_data # return the negative log probability to convert to loss


def compute_predictions_fom_latent_points(meu_z,model,model_output_dim=12, model_type ="NN",with_activation = True):
    
    if meu_z.dim() == 2:
        meu_z = meu_z.unsqueeze(1) # Add a dimension at index 1 if meu_z is 2D - this is for convinience to work with code below

    if model_type =="GP": 
        n_svgp_samples = 10
        N = meu_z.shape[0]
        # draw samples from the latent distribution
        latent_points = meu_z.squeeze(1)
        
        svgp_dist = model(latent_points.to(COMPUTE_DEVICE)) 
        svgp_samples_per_sample = [] 
        for _ in range(n_svgp_samples):
            svgp_samples_per_sample.append(svgp_dist.rsample().unsqueeze(-3)) 
        # pdb.set_trace()
        f = torch.cat(svgp_samples_per_sample).reshape(n_svgp_samples, 1, N, 
                                                       model_output_dim).reshape(-1, N, model_output_dim).mean(axis =0)
    else: # NN
        
        # draw samples from the latent distribution
        latent_points = meu_z.squeeze(1) # Nsamps x latent_dim
        # pass samples through the model
        f = model(latent_points.to(COMPUTE_DEVICE))  # Nsamps x 1 x 23
    
    if f.dim() == 2: #TODO remove this dependency on f being 3D since 2D is enough to represent the output i.e Nsamps x output_dim
        f = f.unsqueeze(0)  # Add a dimension at index 0 if f is 2D - this is for convinience to work with code below

    
    if with_activation:
        for metric in SUMMARIZED_METRICS:

            metric_type, fidxs, length = get_summarized_metric_details(metric)

            dist_params = activation_dict[metric_type](
                            f[:, :, fidxs], 1, length 
                        )
            if metric_type == "binary":
                dist_params = [dist_params[1]] # ignore the count for binary metrics and focus on the logits
                # ensure dist_params has size (N, num_params)
                if dist_params[0].dim() == 0: # if it is a single value, add 2 dimensions i.e (1,1)
                    dist_params = [dist_params[0].unsqueeze(0).unsqueeze(0)]
            elif metric_type =="binarySpan":
                psiTheta,psiSigma,_,_ = convert_sigmoid_params(dist_params[0],dist_params[1],to = "presentable")
                dist_params = [psiTheta,psiSigma]

            # dist params comes in a a list - create a tensor from it
            dist_params = torch.stack(dist_params).squeeze(1).T # convert to shape (N, num_params)
            f[:, :, fidxs] = dist_params
        
    return f

def run_optmization_search(model, update_w_data, num_restarts = 10):
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
    from utils.active_learning_utils import update_latent_dist_from_data_with_restarts
    import torch
    
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

    print(f"Num Restarts {num_restarts}: Initial meu_z positions:", meu_z_init[0])
    print(f"Num Restarts {num_restarts}: Initial sigma_z positions:", sigma_z_init[0])
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
        optimized_dist, best_meu_z, best_sigma_z, lowest_loss = update_latent_dist_from_data_with_restarts(
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
        
        return lowest_loss, best_meu_z
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        # Return original position and a high loss if optimization fails
        return float('inf'), None

def predict_parameters_from_data(update_w_data, model, num_points = 200, use_grid_search = True):
    """
    Predicts the distributional parameters for the data in update_w_data.
    Args:
        update_w_data (dict): A dictionary containing the data for each metric.
        model: The DLVM model used for prediction.
        num_points (int): Number of points for grid search or optimization.
        use_grid_search (bool): If True, uses grid search. If False, uses optimization.
    Returns:
        dictionary: A dict containing the predicted parameters for the data.
    """
    # initialize an empty dict to store the predicted parameters
    predicted_parameters = {}
    if use_grid_search:
        loss, best_meu_z = run_grid_search(model, update_w_data, num_points = num_points)
    else:
        # Use optimization instead of just the model's current position
        loss, best_meu_z = run_optmization_search(model, update_w_data, num_restarts = num_points)
    predicted_f = compute_predictions_fom_latent_points(best_meu_z, model,with_activation = True)
    
    predicted_params = extract_model_params_from_predictions(predicted_f, update_w_data.keys())

    return predicted_params

def extract_model_params_from_predictions(predictions, metrics):
    """
    Extracts the model parameters from the predictions.
    Args:
        predictions (dict): A tensor of shape (N, 1, 12) containing the predictions for each metric.
        model_type (str): The type of model to use for extraction.
    Returns:
        dict: A dictionary containing the extracted parameters.
    """
    predicted_params = {}
    for metric in metrics:
        
        metric_type = CURR_METRICS_DICT[metric]["type"]
        fidxs = CURR_METRICS_DICT[metric]["f_idxs"]
        
        dist_params = predictions[:, :, fidxs].flatten()
            
        
        # detatch the parameters from the graph and convert to non-tensor float value
        dist_params = [param.detach().cpu().item() for param in dist_params]

        summarized_metric_label = CURR_METRICS_DICT[metric]["summary_metric_label"]
        predicted_params[summarized_metric_label] = dist_params

    return predicted_params

def get_predictions_dicts_from_latent_points(meu_z, model, model_type = "NN", with_activation = True):

    f = compute_predictions_fom_latent_points(meu_z, model, model_output_dim=12, model_type = model_type, with_activation = with_activation)
    # ensure the first dimension is the batch dimension i.e if 1xnxnum_params, make it nx1xnum_params
    f= f.reshape(-1,1,f.shape[-1])
    batch_size = f.shape[0]
    predicted_params_list = []
    
    for i in range(batch_size):
        predicted_params = extract_model_params_from_predictions(f[i:i+1], RELEVANT_METRICS)
        # import pdb; pdb.set_trace()
        predicted_params_list.append(predicted_params)
    
    return predicted_params_list