import sys,os

sys.path.append("../")
import numpy as np
import pdb
from torch import nn
import glob
import torch

from utils.data_distribution_utils import (
    RELEVANT_METRICS,
    CURR_METRICS_DICT,
    activation_dict,
    dist_dict,
    prepare_data,
    get_differentiable_sigmoid,
    COMPUTE_DEVICE,
    mle_params_to_dist
)
from utils.grid_search_utils import generate_grid, log_prob_loss

def get_KL_per_cognitive_test(
    current_posterior_beleif_latent_dist,
    model,
    T,
    m,
    update_w_data,
    num_grid_points=1000,
    use_grid_search=False,
):
    """
    Calculates the KL divergence between the posterior predictive distribution and the posterior belief 
    of the latent distribution for each cognitive test, using either grid search or direct sampling.

    Parameters:
    - current_posterior_beleif_latent_dist: Current posterior belief of the latent distribution (torch.distributions.Distribution)
    - model: Model used for making predictions (nn.Module)
    - T: Number of samples to draw from the latent distribution (int)
    - m: Number of samples to draw from the posterior predictive distribution (int) 
    - update_w_data: Dictionary containing observed data for each metric
    - num_grid_points: Number of points to generate in grid search (int, default=1000)
    - use_grid_search: Whether to use grid search vs direct sampling (bool, default=False)

    Returns:
    - KL_per_test: PyTorch tensor containing KL divergence between posterior predictive distribution
                   and posterior belief of latent distribution for each cognitive test
    """
    if use_grid_search:
        return get_KL_per_cognitive_test_NN_using_grid_search(
            current_posterior_beleif_latent_dist, model, T, m, update_w_data, num_grid_points
        )
    else:
        return get_KL_per_cognitive_test_NN(
            current_posterior_beleif_latent_dist, model, T, m
        )

def get_KL_per_cognitive_test_NN_using_grid_search(
    current_posterior_beleif_latent_dist,
    model,
    T,
    m,
    update_w_data,
    num_grid_points=1000,
):
    """
    Calculates the KL divergence between the posterior predictive distribution and the posterior belief of the latent distribution for each cognitive test.
    Uses grid search to select the most likely latent points based on the data.

    Parameters:
    - current_posterior_beleif_latent_dist: the current posterior belief of the latent distribution (torch.distributions.Distribution)
    - model: the model that is being used to make predictions (nn.Module)
    - T: the number of samples to draw from the latent distribution (int)
    - m: the number of samples to draw from the posterior predictive distribution (int)
    - update_w_data: dictionary containing the observed data for each metric
    - num_grid_points: number of points to generate in the grid search (int)

    Returns:
    - KL_per_test: a PyTorch tensor containing the KL divergence between the posterior predictive distribution and the posterior belief of the latent distribution for each cognitive test (torch.Tensor)
    """
    # Calculate the logarithm of the number of samples to draw from the latent distribution
    log_T = torch.log(torch.tensor(T))
    
    # Initialize an empty list to store the KL divergence for each metric
    KL_per_test = []
    
    # Generate grid of candidate latent points
    grid_points = generate_grid(model, num_grid_points)
    
    # Compute log probability loss for each grid point
    losses = log_prob_loss(grid_points.unsqueeze(1), model, update_w_data)
    
    # Select top 5% (lowest losses = highest log probabilities)
    num_to_select = max(1, int(0.05 * len(losses)))
    top_indices = torch.topk(losses, num_to_select, largest=False).indices
    
    # Use selected points as samples
    selected_points = grid_points[top_indices]
    
    # Repeat selected points to match T if needed
    if len(selected_points) < T:
        repeat_factor = T // len(selected_points) + 1
        zs = selected_points.repeat(repeat_factor, 1)[:T]
    else:
        zs = selected_points[:T]
    
    # Add missing dimension to match expected shape [T, 1, latent_dim]
    zs = zs.unsqueeze(1)
    
    # Pass the selected latent points through the model to get predictions
    fs = model(zs.to(COMPUTE_DEVICE))
    
    # Iterate over each metric
    for metric in RELEVANT_METRICS: 
        # Get the type of distribution to use for this metric
        metric_type = CURR_METRICS_DICT[metric]['type'] 
        
        # Calculate the parameters of the posterior predictive distribution for this metric using the model's predictions
        fidxs = CURR_METRICS_DICT[metric]['f_idxs']
        dist_params = activation_dict[metric_type](fs[:,:,fidxs], 1, CURR_METRICS_DICT[metric]['length']) 
        
        # Create a PyTorch distribution object for the posterior predictive distribution for this metric using the calculated parameters
        all_dists = dist_dict[metric_type](*dist_params) 
        
        # Sample from the posterior predictive distribution for this metric
        y_samples = all_dists.sample((m,)).to(COMPUTE_DEVICE)
        
        # Calculate the log probability of the samples under the posterior predictive distribution for this metric
        probs_y_samples_home_dist = all_dists.log_prob(y_samples)
        
        # Calculate the log probability of the samples under each latent distribution
        # Calculate the log probability of the samples under each latent distribution
        probs_y_samples_all_dists = all_dists.log_prob(y_samples.unsqueeze(-1) ) 
        
        # Calculate the log probability of the samples under each latent distribution minus the logarithm of the number of samples drawn from the latent distribution
        probs = probs_y_samples_all_dists.squeeze() - log_T 
        
        # Calculate the log sum of the probabilities under each latent distribution
        log_P_ys_vect = torch.logsumexp(probs, -1) 
        
        # Calculate the KL divergence between the posterior predictive distribution and each latent distribution for this metric
        KL_vect = ((1/m)*(probs_y_samples_home_dist.squeeze() - log_P_ys_vect.to(COMPUTE_DEVICE) )).sum(dim=-2) 
        
        # Average the KL divergence over the number of samples drawn from the latent distribution
        total_KL_for_metric = (1/T)* KL_vect.sum() 
        
        # Append the averaged KL divergence for this metric to the list of KL divergences
        KL_per_test.append(total_KL_for_metric)
    
    # Return the list of KL divergences as a PyTorch tensor
    return torch.tensor(KL_per_test).float()
    
    

def get_KL_per_cognitive_test_NN(current_posterior_beleif_latent_dist, model, T, m):
    """
    Calculates the KL divergence between the posterior predictive distribution and the posterior belief of the latent distribution for each cognitive test.

    Parameters:
    - current_posterior_beleif_latent_dist: the current posterior belief of the latent distribution (torch.distributions.Distribution)
    - model: the model that is being used to make predictions (nn.Module)
    - T: the number of samples to draw from the latent distribution (int)
    - m: the number of samples to draw from the posterior predictive distribution (int)

    Returns:
    - KL_per_test: a PyTorch tensor containing the KL divergence between the posterior predictive distribution and the posterior belief of the latent distribution for each cognitive test (torch.Tensor)
    """
    # Calculate the logarithm of the number of samples to draw from the latent distribution
    log_T = torch.log(torch.tensor(T))
    
    # Initialize an empty list to store the KL divergence for each metric
    KL_per_test = []
    
    # Sample from the latent distribution and pass the samples through the model to get the corresponding predictions
    zs = current_posterior_beleif_latent_dist.rsample((T,)) 
    fs = model(zs.to(COMPUTE_DEVICE))
    
    # Iterate over each metric
    for metric in RELEVANT_METRICS: 
        # Get the type of distribution to use for this metric
        metric_type = CURR_METRICS_DICT[metric]['type'] 
        
        # Calculate the parameters of the posterior predictive distribution for this metric using the model's predictions
        fidxs = CURR_METRICS_DICT[metric]['f_idxs']
        dist_params = activation_dict[metric_type](fs[:,:,fidxs], 1, CURR_METRICS_DICT[metric]['length']) 
        
        # Create a PyTorch distribution object for the posterior predictive distribution for this metric using the calculated parameters
        all_dists = dist_dict[metric_type](*dist_params) 
        
        # Sample from the posterior predictive distribution for this metric
        y_samples = all_dists.sample((m,)).to(COMPUTE_DEVICE)
        
        # Calculate the log probability of the samples under the posterior predictive distribution for this metric
        probs_y_samples_home_dist = all_dists.log_prob(y_samples)
        
        # Calculate the log probability of the samples under each latent distribution
        # Calculate the log probability of the samples under each latent distribution
        probs_y_samples_all_dists = all_dists.log_prob(y_samples.unsqueeze(-1) ) 
        
        # Calculate the log probability of the samples under each latent distribution minus the logarithm of the number of samples drawn from the latent distribution
        probs = probs_y_samples_all_dists.squeeze() - log_T 
        
        # Calculate the log sum of the probabilities under each latent distribution
        log_P_ys_vect = torch.logsumexp(probs, -1) 
        
        # Calculate the KL divergence between the posterior predictive distribution and each latent distribution for this metric
        KL_vect = ((1/m)*(probs_y_samples_home_dist.squeeze() - log_P_ys_vect.to(COMPUTE_DEVICE) )).sum(dim=-2) 
        
        # Average the KL divergence over the number of samples drawn from the latent distribution
        total_KL_for_metric = (1/T)* KL_vect.sum() 
        
        # Append the averaged KL divergence for this metric to the list of KL divergences
        KL_per_test.append(total_KL_for_metric)
    
    # Return the list of KL divergences as a PyTorch tensor
    return torch.tensor(KL_per_test).float()

def perform_active_learning_update(curr_posterior_beleif_latent_dist, model, update_w_data, args, metrics_dict, num_restarts=100, always_start_from_train_mean_meu_z=False, limit_sigma_z_to_unit_ball=False):
    """
    Core active learning update step that:
    1. Calculates information gain for each test
    2. Selects the next test
    3. Updates the posterior belief
    
    Args:
        curr_posterior_beleif_latent_dist: Current posterior belief distribution
        model: The DLVM model
        update_w_data (dict): Current collected data
        args: Command line arguments
        metrics_dict: Dictionary of metric configurations
        
    Returns:
        tuple: (new_posterior_dist, new_meu_z, new_sigma_z, lowest_loss, selected_test)
    """
    
    # check if the update_w_data is empty i.e every metric has no data
    if any(len(data) >0 for data in update_w_data.values()):
        
        # Get dimensions and current distribution parameters
        latent_dim = curr_posterior_beleif_latent_dist.mean.shape[1]
        mean_meu_z = curr_posterior_beleif_latent_dist.mean.clone().detach()
        mean_sigma_z = curr_posterior_beleif_latent_dist.stddev.clone().detach()

        # Initialize parameters based on number of restarts
        meu_z_init = torch.zeros(max(1, num_restarts), latent_dim).to(COMPUTE_DEVICE)
        sigma_z_init = torch.ones(max(1, num_restarts), latent_dim).to(COMPUTE_DEVICE)  

        # First position always uses current mean
        meu_z_init[0] = mean_meu_z
        sigma_z_init[0] = mean_sigma_z

        # For multiple restarts, sample remaining positions uniformly from model's learned range
        if num_restarts > 1:
            min_meu_z = model.meu_z.min(dim=0)[0]
            max_meu_z = model.meu_z.max(dim=0)[0]
            meu_z_init[1:] = min_meu_z.unsqueeze(0) + (max_meu_z - min_meu_z).unsqueeze(0) * torch.rand(num_restarts - 1,    latent_dim).to(COMPUTE_DEVICE)

        # Convert to parameters for optimization
        meu_z_param = torch.nn.Parameter(meu_z_init)
        sigma_z_param = torch.nn.Parameter(sigma_z_init)

        # Update posterior distribution
        new_posterior_dist, new_meu_z, new_sigma_z, lowest_loss = update_latent_dist_from_data_with_restarts(
            update_w_data, 
            args.max_epcohs, 
            args.lr, 
            model, 
            args.max_n_progress_fails, 
            meu_z_param,  # Pass the Parameter object
            sigma_z_param,  # Pass the Parameter object
            n_samples=20, 
            grad_clip=args.grad_clip,
            min_allowed_log_prob=args.min_allowed_log_prob, 
            metrics_dict=metrics_dict,
        )
    else:
        if always_start_from_train_mean_meu_z: # assumes the update starts from the mean of the trained model
            new_meu_z = model.meu_z.mean(dim=0).unsqueeze(0) # shape: (1, latent_dim)
            new_sigma_z = torch.ones_like(new_meu_z)
            new_posterior_dist = torch.distributions.Normal(new_meu_z, new_sigma_z)
        else: # uses the previous posterior belief of the latent distribution
            new_posterior_dist = curr_posterior_beleif_latent_dist
            new_meu_z = curr_posterior_beleif_latent_dist.mean
            new_sigma_z = curr_posterior_beleif_latent_dist.stddev
        lowest_loss = 0
        test_to_run_next = None
    
    # ensure that the stand dev is always 1 for the new posterior distribution
    if limit_sigma_z_to_unit_ball:
        new_sigma_z = torch.ones_like(new_sigma_z)*3 # set the standard deviation to 2
        new_posterior_dist = torch.distributions.Normal(new_meu_z, new_sigma_z)

    print("Mean and std of new posterior distribution: ", new_posterior_dist.mean, new_posterior_dist.stddev)

    # Calculate KL divergence (information gain) for each possible test
    KL_per_test = get_KL_per_cognitive_test(
        new_posterior_dist,
        model,
        args.T,
        args.M,
        update_w_data,
        args.num_grid_points,
        args.use_grid_search,
    )
    
    
    test_to_run_next = RELEVANT_METRICS[KL_per_test.argmax().item()]
    
    return new_posterior_dist, new_meu_z, new_sigma_z, lowest_loss, test_to_run_next



def compute_probs_data_and_KLD(
    curr_posterior_beleif_latent_dist,
    model, 
    data_dict, 
    all_metrics, 
    best_mle_params_obs, 
    n_samples=500, 
    metrics_dict=None,
    filename = None,
):
    """
    Calculates the probability of data given a model, the MLE log probability and the KLD divergence between MLE and Model distribution.

    Parameters:
    - curr_posterior_beleif_latent_dist: the current posterior belief of the latent distribution (torch.distributions.Distribution)
    - model: the model that is being used to make predictions (nn.Module)
    - data_dict: a dictionary containing the data that the log probabilities will be calculated for (dict)
    - all_metrics: a list of metrics for which the log probability of the data will be calculated (list of str)
    - best_mle_params_obs: a dictionary containing the maximum likelihood estimate (MLE) of the data for each metric (dict of str -> torch.Tensor)
    - n_samples: the number of samples to draw from the latent distribution (int, optional)
    - metrics_dict: a dictionary containing information about the metrics, such as the type of distribution to use for each metric and the indices of the features in the model's output that correspond to each metric (dict, optional)
    - filename: a file to save the model's predictions to (str, optional)

    Returns:
    - total_log_prob_data: the sum of the log probabilities of the data under the model's predictions (float)
    - mle_log_prob: the sum of the log probabilities of the data under the MLEs of the data (float)
    """
    # Sample from the latent distribution
    latent_points = curr_posterior_beleif_latent_dist.rsample((n_samples,)) 

    # Pass the latent samples through the model to get the corresponding predictions
    f = model(latent_points.to(COMPUTE_DEVICE))
    
    # Save the model's predictions to a file if specified
    if filename is not None:
        torch.save(f, filename)
    
    total_n_samples = n_samples 
    total_log_prob_data = 0 
    mle_log_prob = 0.0 
    total_kld = 0.0
    
    # Iterate over each metric
    for metric in all_metrics: 
        # Get the type of distribution to use for this metric
        metric_type = metrics_dict[metric]['type'] 
        
        # Get the indices of the features in the model's output that correspond to this metric
        fidxs = metrics_dict[metric]['f_idxs']
        
        # Get the data for this metric
        data, _, counts, _ = data_dict[metric] 
        
        # Calculate the parameters of the distribution for this metric using the model's predictions
        dist_params = activation_dict[metric_type](f[:,:,fidxs], counts, metrics_dict[metric]['length'])
        
        # Calculate the parameters of the distribution for this metric using the model's predictions
        dist_params = activation_dict[metric_type](f[:,:,fidxs], counts, metrics_dict[metric]['length'])
        
        
        # Create a PyTorch distribution object for this metric using the calculated parameters
        dist =  dist_dict[metric_type](*dist_params)  
        
        # Calculate the log probability of the data for this metric under the model's predictions
        prob_data = dist.log_prob(data[np.logical_not(np.isnan(data))].to(COMPUTE_DEVICE)).sum()/total_n_samples
        
        # Calculate the MLE of the data for this metric
        if "Complex" in metric: 
            mle_dist_params = best_mle_params_obs["CorsiComplex"] 
        elif "Simple" in metric:
            mle_dist_params = best_mle_params_obs["SimpleSpan"]
        else:
            mle_dist_params = best_mle_params_obs[metric] 
        
        
        mle_dist = mle_params_to_dist(metric, mle_dist_params, metric_type, counts=counts.item() ) 
        mle_prob = mle_dist.log_prob(data[np.logical_not(np.isnan(data))]).sum() 
        
        # Set the MLE of the data to the maximum of the MLE and the probability of the data under the model's predictions
        mle_prob = max(mle_prob, prob_data)
        
        # Accumulate the probability of the data under the model's predictions and the MLE of the data
        total_log_prob_data += prob_data.item() 
        mle_log_prob +=  mle_prob.item() 
        
        # compute the KL Divergence for the metric
        mle_dist = move_distribution_to_cuda(mle_dist)
        
        total_kld +=calculate_kld_given_metric(mle_dist, dist, metric) 

    return total_log_prob_data, mle_log_prob, total_kld

def calculate_kld_given_metric(predicted_dist, base_distribution, metric=None):
    # Calculate Kullback-Leibler divergence (KLD) between the two distributions
    metric_kld = torch.distributions.kl_divergence(predicted_dist, base_distribution) # KLD(P||Q) where P is the predicted distribution and Q is the base distribution
    if isinstance(base_distribution, torch.distributions.Binomial):
        # Handling cases where the KLD is infinite and the MLE probability is 1
        if torch.isclose(predicted_dist.probs, torch.tensor(1.0),atol=10**-4):
            metric_kld = torch.where(torch.isinf(metric_kld), torch.tensor(0.0).to(COMPUTE_DEVICE), metric_kld) # if the MLE probability is 1, set the KLD to 0
        # Handling cases where the KLD is infinite and the MLE probability is not 1
        elif torch.isinf(metric_kld.mean()):
            base_distribution.probs = torch.where(torch.isinf(metric_kld), torch.tensor(0.9999999).to(COMPUTE_DEVICE), base_distribution.probs)
            metric_kld = torch.distributions.kl_divergence(predicted_dist, base_distribution)
    return metric_kld.mean()

def move_distribution_to_cuda(distribution):
    # Check the type of the distribution
    if isinstance(distribution, torch.distributions.Binomial):
        # Get the total_count and probs parameters of the distribution
        total_count = distribution.total_count
        probs = distribution.probs

        # Move the parameters to the GPU
        total_count = total_count.to(COMPUTE_DEVICE)
        probs = probs.to(COMPUTE_DEVICE)

        # Create a new Binomial distribution with the GPU-based parameters
        cuda_distribution = torch.distributions.Binomial(total_count, probs)
    elif isinstance(distribution, torch.distributions.Normal):
        # Get the loc and scale parameters of the distribution
        loc = distribution.loc
        scale = distribution.scale

        # Move the parameters to the GPU
        loc = loc.to(COMPUTE_DEVICE)
        scale = scale.to(COMPUTE_DEVICE)

        # Create a new Normal distribution with the GPU-based parameters
        cuda_distribution = torch.distributions.Normal(loc, scale)
    # Check the type of the distribution
    elif isinstance(distribution, torch.distributions.LogNormal):
        # Get the loc and scale parameters of the distribution
        loc = distribution.loc
        scale = distribution.scale

        # Move the parameters to the GPU
        loc = loc.to(COMPUTE_DEVICE)
        scale = scale.to(COMPUTE_DEVICE)

        # Create a new LogNormal distribution with the GPU-based parameters
        cuda_distribution = torch.distributions.LogNormal(loc, scale)
    elif isinstance(distribution, torch.distributions.Beta):
        # Get the concentration1 and concentration0 parameters of the distribution
        concentration1 = distribution.concentration1
        concentration0 = distribution.concentration0

        # Move the parameters to the GPU
        concentration1 = concentration1.to(COMPUTE_DEVICE)
        concentration0 = concentration0.to(COMPUTE_DEVICE)

        # Create a new Beta distribution with the GPU-based parameters
        cuda_distribution = torch.distributions.Beta(concentration1, concentration0)
    else:
        # Raise an exception if the distribution type is not recognized
        raise ValueError(f'Unrecognized distribution type: {type(distribution)}. Modify this function to handle it if needed')

    return cuda_distribution





def update_latent_dist_from_data(
    update_w_data,  # dictionary of metric names and data
    max_epochs,  # maximum number of epochs to run the optimization for
    lr,  # learning rate for the Adam optimizer
    model,  # function that maps latent points to outputs
    max_n_progress_fails,  # maximum number of epochs where loss does not improve before stopping optimization
    meu_z,  # tensor representing the initial mean of the latent normal distribution
    sigma_z,  # tensor representing the initial standard deviation of the latent normal distribution
    n_samples=10,  # number of samples to draw from the latent distribution at each optimization step
    grad_clip=0.2,  # maximum allowed magnitude of gradients before clipping
    min_allowed_log_prob=-1000,  # minimum allowed log probability before clamping
    metrics_dict=None,  # dictionary of metric names and information dictionaries
):
    
    """
    Update a latent normal distribution using data.
        
        INPUTS:
        - update_w_data: dictionary of metric names and data
        - max_epochs: maximum number of epochs to run the optimization for
        - lr: learning rate for the Adam optimizer
        - model: function that maps latent points to outputs
        - max_n_progress_fails: maximum number of epochs where loss does not improve before stopping optimization
        - meu_z: tensor representing the initial mean of the latent normal distribution
        - sigma_z: tensor representing the initial standard deviation of the latent normal distribution
        - n_samples: number of samples to draw from the latent distribution at each optimization step
        - grad_clip: maximum allowed magnitude of gradients before clipping
        - min_allowed_log_prob: minimum allowed log probability before clamping
        - metrics_dict: dictionary of metric names and information dictionaries

        OUTPUTS:
        - latent_dist: updated latent normal distribution
        - best_meu_z: tensor representing the mean of the updated latent normal distribution
        - best_sigma_z: tensor representing the standard deviation of the updated latent normal distribution
    """

    # create Adam optimizer
    optimizer = torch.optim.Adam([meu_z, sigma_z], lr=lr)
    
    # counter for number of epochs where loss does not improve
    progress_fails = 0
    
    # initialize lowest loss to infinity
    lowest_loss = np.inf
    
    # for each epoch
    for e in range(max_epochs):
        # zero gradients
        optimizer.zero_grad()
        
        # create latent normal distribution with current mean and standard deviation
        latent_dist = torch.distributions.Normal(
            meu_z, torch.nn.functional.softplus(sigma_z)
        )
        
        # draw samples from the latent distribution
        latent_points = latent_dist.rsample((n_samples,))  # n_samples x N x Latent_dim
        
        # pass samples through the model
        f = model(latent_points.to(COMPUTE_DEVICE))  # Nsamps x 1 x 23  (Out size =23)
        
        # initialize log probability of data under the latent distribution to 0
        total_log_prob_data = 0
        
        # initialize normalization constant to 0
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
                probs = dist.log_prob(data.to(COMPUTE_DEVICE))
                
                # clamp the log probabilities to the minimum allowed value
                probs = torch.clamp(probs, min=min_allowed_log_prob)
                
                # compute the log probability of the data for this metric
                log_prob_task = probs.sum() / n_samples
                
                # add the log probability of the data for this metric to the total log probability
                total_log_prob_data += log_prob_task
        
        # ensure the normalization constant is at least 1
        norm_const = max(1, norm_const)
        
        # normalize the total log probability by the normalization constant
        total_log_prob_data = total_log_prob_data / (norm_const)
        
        # compute the loss as the negative of the total log probability
        loss = -total_log_prob_data
        
        # compute gradients
        loss.backward()
        
        # clip gradients to the maximum allowed magnitude
        torch.nn.utils.clip_grad_norm_([meu_z, sigma_z], grad_clip)
        
        # update model parameters
        optimizer.step()
        
        # if the current loss is less than the lowest loss so far
        if loss.item() < lowest_loss:
            # update the lowest loss
            lowest_loss = loss.item()
            
            # reset the progress fails counter
            progress_fails = 0
            
            # update the best mean and standard deviation found so far
            best_meu_z = meu_z
            best_sigma_z = sigma_z
        
        # otherwise
        else:
            # increment the progress fails counter
            progress_fails += 1
        
        # if the progress fails counter exceeds the maximum allowed
        if progress_fails > max_n_progress_fails:
            # break out of the epoch loop
            break

    # return a normal distribution object with the best mean and standard deviation found,
    # as well as the tensors representing the best mean and standard deviation
    return (
        torch.distributions.Normal(
            best_meu_z, torch.nn.functional.softplus(best_sigma_z)
        ),
        best_meu_z,
        best_sigma_z,
        lowest_loss
    )


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
    
    # Training loop
    for e in range(max_epochs):
        optimizer.zero_grad()
        
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
        for metric in update_w_data.keys():
            data = update_w_data[metric]
            
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
                
                # Add to total log probability for each restart
                total_log_prob_data += log_prob_task
        
        # Normalize by number of data points
        norm_const = max(1, norm_const)
        total_log_prob_data = total_log_prob_data / norm_const
        
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
        global_best_loss
    )




def compute_total_n_trials_actual_session_data(held_out_session_id):
    """
    Compute the total number of trials in a given session.
    
    Parameters:
    - held_out_session_num: The index of the session for which to compute the total number of trials.
    
    Returns:
    - The total number of trials in the given session.
    """
    # Initialize the total number of trials to 0
    total_n_trails = 0
    
    # Get the data for the given session
    data_dict, all_metrics, _= prepare_data(heldout_obs_ids =[held_out_session_id], get_heldout_instead = True)
    # Iterate through the cognitive tests in the session
    for metric in all_metrics:
        # Get the counts for the cognitive test
        _, _, counts, _ = data_dict[metric]
        
        # Add the counts to the total number of trials, ignoring NaN values
        total_n_trails += torch.nansum(counts)

    # Return the total number of trials as an integer
    return int(total_n_trails.item())

def evaluate_model_fit_performance(
    curr_posterior_belief_latent_dist,
    model,
    data_dict,
    all_metrics,
    best_mle_params_obs,
    n_samples=500,
    metrics_dict=None,
    mle_ratio_type="naive",
    filename=None,
):
    total_log_prob_data, mle_log_prob_data, total_kld = compute_probs_data_and_KLD(
        curr_posterior_belief_latent_dist,
        model,
        data_dict,
        all_metrics,
        best_mle_params_obs,
        n_samples=n_samples,
        metrics_dict=metrics_dict,
        filename=filename,
    )
    performance={}
    performance["mle_ratio_naive"] = compute_naive_mle_ratio(total_log_prob_data, mle_log_prob_data)
    performance["mle_ratio_correct"] = compute_correct_mle_ratio(total_log_prob_data, mle_log_prob_data)
    performance["model_log_prob_data"]=total_log_prob_data
    performance["mle_log_prob_data"]=mle_log_prob_data
    performance["total_kld"]=total_kld
    performance["meu_z"]=curr_posterior_belief_latent_dist.mean.tolist()
    performance["sigma_z"]=curr_posterior_belief_latent_dist.stddev.tolist()
    return performance
    

def get_mle_ratio(
    curr_posterior_belief_latent_dist,
    model,
    data_dict,
    all_metrics,
    best_mle_params_obs,
    n_samples=500,
    metrics_dict=None,
    mle_ratio_type="naive",
    filename=None,
):
    assert mle_ratio_type in ["naive", "correct"]
    total_log_prob_data, mle_log_prob_data, log_prob_dist_list = compute_probs_data_and_KLD(
        curr_posterior_belief_latent_dist,
        model,
        data_dict,
        all_metrics,
        best_mle_params_obs,
        n_samples=n_samples,
        metrics_dict=metrics_dict,
        filename=filename,
    )
    # Compute MLE Ratio
    if mle_ratio_type == "naive":
        mle_ratio = compute_naive_mle_ratio(total_log_prob_data, mle_log_prob_data)
    elif mle_ratio_type == "correct":
        mle_ratio = compute_correct_mle_ratio(total_log_prob_data, mle_log_prob_data)

    return mle_ratio, log_prob_dist_list


def compute_naive_mle_ratio(log_prob_data, mle_log_prob_data):
    mle_ratio = log_prob_data / mle_log_prob_data
    if (
        mle_log_prob_data < 0.0 and log_prob_data < 0.0
    ):  # if probs are negative, take inverse of ratio
        mle_ratio = mle_log_prob_data / log_prob_data
    mle_ratio = max(mle_ratio, 0.0)
    return mle_ratio




def compute_correct_mle_ratio(log_prob_data, mle_log_prob_data):
    ratio = torch.exp(torch.tensor(log_prob_data - mle_log_prob_data))
    return ratio


def get_data_list_from_mle_data_generator(
    metric, best_mle_params_obs, test_batch_size, verbose=False
):
    """
        Generate a list of simulated data points for a cognitive test using the maximum likelihood estimate (MLE) parameters for the cognitive test.
        
        INPUT:
        - metric: The name of the cognitive test for which to generate simulated data.
        - best_mle_params_obs: A dictionary of MLE parameters for each cognitive test.
        - test_batch_size: The number of simulated data points to generate for the cognitive test.
        - verbose: A boolean value indicating whether to print verbose output. (default: False)
        
        #Note: Give it a metric, metric type, N (number of observations we want), ground truth parameters (Note for myself for future function)
        #Note: should return list of sampled data points of size N 
        OUTPUT:
        - A list of simulated data points for the cognitive test.
    """
    # Get the type of distribution to use for the cognitive test
    metric_type = CURR_METRICS_DICT[metric]["type"]
    
    # Retrieve the MLE parameters for the cognitive test from the dictionary of MLE parameters
    if "Complex" in metric:
        mle_dist_params = best_mle_params_obs["CorsiComplex"]
    elif "Simple" in metric:
        mle_dist_params = best_mle_params_obs["SimpleSpan"]
    else:
        mle_dist_params = best_mle_params_obs[metric]
    
    # Create a PyTorch distribution object for the cognitive test using the retrieved MLE parameters and the distribution type
    mle_dist = mle_params_to_dist(metric, mle_dist_params, metric_type)
    
    # Sample from the PyTorch distribution object and convert the resulting samples to a list
    simulated_data_points = mle_dist.sample((test_batch_size,)).tolist()
    
    # If verbose output is enabled, print the name of the cognitive test and the simulated data
    if verbose:
        print(f"metric: {metric}, simulated data: {simulated_data_points}")

    # Return the list of simulated data for the cognitive test
    return simulated_data_points


