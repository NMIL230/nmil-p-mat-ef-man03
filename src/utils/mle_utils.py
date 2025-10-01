"""
    This file has the functions and logic for performing MLE predictions on the data for each metric
"""

import sys
from requests import session
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))

sys.path.append(parent_dir)

import torch
from torch import nn
from utils.data_distribution_utils import *
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

COMPUTE_DEVICE = "cpu"
# fitting binomial distribution
def get_mle_binomial(data, counts, max_progress_fails=100, tracker=None): 
    dist_class = torch.distributions.binomial.Binomial
    f1 = data.sum()/counts
    probs =torch.clamp(f1, min=1e-2, max=1 - 1e-2) # never exceed 0.99 or below 0.01
    best_f = probs.item()
    dist = dist_class(counts, probs)
    best_total_log_prob = -dist.log_prob(data).sum().item()
    best_probs = best_f
    
   
    return best_probs, best_f, best_total_log_prob


# fitting beta distribution
def get_mle_beta(data, max_progress_fails=100, tracker=None):
    progress_fails = 0
    lowest_loss = torch.inf
    dist_class = torch.distributions.beta.Beta
    f1 = torch.nn.Parameter(torch.randn(1))
    f2 = torch.nn.Parameter(torch.randn(1))
    optimizer = torch.optim.Adam([f1, f2], lr=0.01)
    while progress_fails < max_progress_fails:
        optimizer.zero_grad()
        alpha, beta = torch.nn.functional.softplus(f1), torch.nn.functional.softplus(f2)
        dist = dist_class(alpha, beta)
        loss = -dist.log_prob(data).sum()
        loss.backward()
        optimizer.step()

        if loss.item() < lowest_loss:  # if progress
            lowest_loss = loss.item()
            progress_fails = 0
            best_alpha = alpha
            best_beta = beta
            best_f = torch.tensor([f1, f2])
            best_total_log_prob = loss * -1
        else:
            progress_fails += 1
    return best_alpha, best_beta, best_f, best_total_log_prob


def initialize_sigmoid_params(data_per_length, count_per_length, lengths, device):
    """
    Computes data-driven initialization for sigmoid parameters (alpha, beta).
    Ignores lengths with zero total counts.
    
    Args:
        data_per_length: List of tensors representing the number of successes per length.
        count_per_length: List of tensors representing the number of trials per length.
        lengths: List of corresponding lengths.
        device: Torch device (CPU/GPU).
        
    Returns:
        Tuple (alpha_init, beta_init) as torch tensors.
    """
    # import pdb; pdb.set_trace()
    # Filter valid lengths where counts > 0
    valid_indices = [i for i, cnt in enumerate(count_per_length) if cnt > 0]

    if not valid_indices:
        return torch.tensor(2.0, device=device), torch.tensor(1.0, device=device)  # Default values if no valid data

    # Convert to tensors
    valid_lengths = torch.tensor([lengths[i] for i in valid_indices], device=device, dtype=torch.float32)
    valid_counts = torch.tensor([count_per_length[i] for i in valid_indices], device=device, dtype=torch.float32)

    # Compute empirical success rates per length
    success_rates = torch.tensor(
        [data_per_length[i].item() / count_per_length[i] for i in valid_indices], 
        device=device
    )

    # Weighted mean length
    mean_length = (valid_lengths * success_rates * valid_counts).sum() / (success_rates * valid_counts).sum()

    # if nan, set to 0
    if torch.isnan(mean_length):
        mean_length = torch.tensor(0.0, device=device)
    
    # # Weighted variance & standard deviation
    # variance_length = ((valid_lengths - mean_length) ** 2 * success_rates * valid_counts).sum() / (success_rates * valid_counts).sum()
    # std_length = torch.sqrt(variance_length + 1e-6)  # Small epsilon to avoid division by zero

    # # Initialize alpha & beta
    # alpha_init = torch.logit(torch.clamp(mean_length / (1 + mean_length), min=1e-6, max=1-1e-6)) if mean_length > 0 else torch.tensor(1.0, device=device)
    # beta_init = 1 / (std_length + 1e-6) if std_length > 0 else torch.tensor(1.0, device=device)

    return mean_length, 2.0  # alpha_init, beta_init
# fitting 2 parameter sigmoid to span taks
def get_mle_span_tasks_2params(
    data_per_length, count_per_length, lengths, max_progress_fails=100, tracker=None,
    initialize_from_data = True
):
    device = torch.device(COMPUTE_DEVICE)
    progress_fails = 0
    lowest_loss = torch.inf
    dist_class = torch.distributions.binomial.Binomial
    # fs = torch.nn.Parameter(torch.randn(2, device=device))
    if initialize_from_data:
        alpha_init, beta_init = initialize_sigmoid_params(data_per_length, count_per_length, lengths, device)
        fs = torch.nn.Parameter(torch.tensor([alpha_init, beta_init], device=device
        ))
    else:
        fs = torch.nn.Parameter(torch.tensor([5.0,3.0], device=device))
    print(f"Initial sigmoid params: {fs}, data_per_length: {data_per_length}, count_per_length: {count_per_length}")
    losses = []
    optimizer = torch.optim.Adam([fs], lr=1e-2)
    
    count_per_length = [torch.tensor(cnt).to(device) for cnt in count_per_length] 
    data_per_length = [data.to(device) for data in data_per_length]
    g, l = torch.tensor(0.02,device=device), torch.tensor(0.02,device=device)  # set values (v6)
    tol = 1e-3
    iters = 0

    while progress_fails < max_progress_fails:
        optimizer.zero_grad()
        sigmoid = get_differentiable_sigmoid(fs[0], fs[1], g, l)
        loss = torch.tensor(0.0, device=device)
        for ix, length in enumerate(lengths):
            data = data_per_length[ix]
            prob = sigmoid(torch.tensor(length).to(device))
            dist = dist_class( count_per_length[ix], prob)
            loss = loss + -dist.log_prob(data).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([fs], 0.1)
        optimizer.step()
        losses.append(loss.item())
        if (loss.item() < lowest_loss) and abs(loss.item() - lowest_loss)>tol:  # if progress
            lowest_loss = loss.item()
            progress_fails = 0
            best_f = fs
            best_total_log_prob = loss * -1
        else:
            progress_fails += 1

        iters+=1
        # print(f"{loss.item()}:{lowest_loss}-{progress_fails}/{max_progress_fails}")
    print(f"Sigmoid: Used {iters} iterations on {device}")
    return best_f, best_total_log_prob


# computes MLE parameters for each metric in a session
def get_mle_params_per_metric(
    update_w_data_dict, mpf=200, verbose2=False, metrics=RELEVANT_METRICS, session=None
):
    # mpf = max progress fails
    mle_params_out = {}
    
    binarySpan_dict ={} # will aggregate data across different lengths for binary span tasks
    
    sspan_metric = ""
    cspan_metric = ""
    metrics_dict, _ = create_metrics_dict(
        use_relevant_metrics_only=False
    )  # get metric dicts with all metrics
    total_prob = 0
    for metric in metrics:
        best_total_log_prob = 0
        metric_type = metrics_dict[metric]["type"]
        data = update_w_data_dict[metric]
        data = torch.tensor(data)

        if metric_type == "binarySpan":
            if len(data) > 0:
                summary_metric = metrics_dict[metric]["summary_metric_label"]
                if not (summary_metric in binarySpan_dict.keys()):
                    binarySpan_dict[summary_metric]={"lengths":[],"data_per_length":[],"count_per_length":[]}
                
                binarySpan_dict[summary_metric]["lengths"].append(metrics_dict[metric]["length"])
                binarySpan_dict[summary_metric]["data_per_length"].append(data.sum())
                binarySpan_dict[summary_metric]["count_per_length"].append(len(data))
                
        elif metric_type == "timing":

            if len(data) >= 2:
                true_loc, true_scale = torch.log(data).mean(), torch.log(data).std(
                    unbiased=True
                )
                mle_params_out[metric] = [true_loc.item(), true_scale.item()]
            elif len(data) == 1:
                true_loc, true_scale = torch.log(data).mean(), torch.tensor(0.01)
                mle_params_out[metric] = [true_loc.item(), true_scale.item()]

        elif metric_type == "beta":
            if len(data) > 0:
                mle_alpha, mle_beta, _, best_prob = get_mle_beta(
                    data, max_progress_fails=mpf
                )
                mle_params_out[metric] = [mle_alpha, mle_beta]
                # print(metric,"beta",best_prob)

        elif metric_type == "binary":  # acc (all binomial dists)
            if len(data) > 0:
                mle_probs, _, best_prob = get_mle_binomial(
                    data.sum(),
                    counts=len(data),
                    max_progress_fails=mpf,
                )
                mle_params_out[metric] = [mle_probs]
                #print(mle_probs,mle_params_out[metric])
        else:
            raise RuntimeError("Invalid metric type")
        total_prob += best_total_log_prob

    for binary_span_metric in binarySpan_dict.keys ():
        bs_metric_data = binarySpan_dict[binary_span_metric]
        print(binary_span_metric)
        if len(bs_metric_data["data_per_length"]) > 0:
            failures = 1
            maxpf = mpf
            while failures < 10:
                try:
                    fs, best_total_log_prob = get_mle_span_tasks_2params(
                        bs_metric_data["data_per_length"],
                        bs_metric_data["count_per_length"],
                        bs_metric_data["lengths"],
                        max_progress_fails=maxpf,
                    )
                    total_prob += best_total_log_prob
                    break
                except:
                    # pdb.set_trace()
                    print(f"Failure number {failures}/10")
                    failures += 1
                    maxpf = int(maxpf / 2)

            if failures > 9:
                print("Failed 10 times to get SSpan mle fs")
            
            psiTheta, psiSigma,_,_ = convert_sigmoid_params(fs[0], fs[1]) # converts to theta and sigma from a and b output from the optimizer
            mle_params_out[binary_span_metric] = [psiTheta, psiSigma]

     # print("Log Probability is ",total_prob )
    return mle_params_out


# computes and return
def generate_mle_best_params(debug=False):

    params_list = {}
    _,_,participant_ids = prepare_data(remove_outliers=False) # all data
    print("N participants:" , len(participant_ids))
    
    max_index = len(participant_ids)
    
    if debug:
        max_index = 10 # only do the first 10
        
    for session in range(0, max_index, 1):
        print(f"Session {participant_ids[session]}")
        data_dict, metrics, _ = prepare_data(heldout_obs_ids =[participant_ids[session]], get_heldout_instead = True)
        
        
        update_w_data, _ = extract_update_w_data_dict(data_dict=data_dict, metrics=metrics)

        mle_params_out = get_mle_params_per_metric(
            update_w_data, mpf=100, metrics=metrics, session=session
        )

        params_list[participant_ids[session]] = mle_params_out
    return params_list


def compute_total_log_probabilities(debug=False):
    mle_best_params = torch.load(
        f"../data/{DATASET}/all_data-best_mle_params_mpf100.pt"
    )
    metrics_dict, _ = create_metrics_dict(
        use_relevant_metrics_only=False
    )  # get matric dicts with all metrics
    _,_,participant_ids = prepare_data() # all data
    total_log_probs = {}
    max_index = len(participant_ids)
    
    if debug:
        max_index = 10 # only do the first 4
        
    for session in range(0, max_index , 1):
        print(f"Session {participant_ids[session]}")
        data_dict, metrics, _ = prepare_data(heldout_obs_ids =[participant_ids[session]], get_heldout_instead = True)
        best_mle_params_obs = mle_best_params[participant_ids[session]]
        total_log_prob_obs = {}
        
        for metric in RELEVANT_METRICS:
            metric_type = metrics_dict[metric][
                "type"
            ]  # get the type of the specific type- it could be timing, binary
            data, tmp_data, counts, mask = data_dict[
                metric
            ]  # tmp data shape: T x 1 x N
            if counts > 0:  # there is data
                counts = torch.tensor(counts).reshape(1)
                
                mle_dist_params = best_mle_params_obs[metrics_dict[metric]["summary_metric_label"]]
                # mle_dist_params = best_mle_params_obs[metric]
                mle_dist = mle_params_to_dist(
                    metric,
                    mle_dist_params,
                    metric_type,
                    counts=counts.item(),
                    metrics_dict=metrics_dict,
                    use_differentiable_sigmoid=True # use differentiable sigmoid since these log_probs are used for optimization
                )
                mle_log_prob_metric = mle_dist.log_prob(data[mask]).mean()
                
                total_log_prob_obs[metric] = mle_log_prob_metric.item()

                print(f"session {participant_ids[session]}  {metric} log_prob =, {mle_log_prob_metric}")

        total_log_probs[participant_ids[session]] = total_log_prob_obs

    mean_log_probabilities = dict(
        np.mean(pd.DataFrame(total_log_probs).T, axis=0)
    )  # compute the mean log_probability of each metric
    std_log_probabilities = dict(
        np.std(pd.DataFrame(total_log_probs).T, axis=0)
    )  # compute the std log_probability of each metric
    
    
    #in case any metric is missing, set it's mean to 0 and std =1
    for metric in RELEVANT_METRICS:
        if metric not in mean_log_probabilities.keys(): # metric is missing
            mean_log_probabilities[metric]=0
            std_log_probabilities[metric]=1
            print(f"{metric} is missing,  set to default mean and std (0/1)")
            
    torch.save(mean_log_probabilities, f"../data/{DATASET}/mean_log_probabilities.pt")
    torch.save(std_log_probabilities, f"../data/{DATASET}/std_log_probabilities.pt")

    return total_log_probs

def extract_mle_parameters(mle_params_file_path=os.path.join(parent_dir, f'data/{DATASET}/all_data-best_mle_params_mpf100.pt')):
    """
    Extracts MLE parameters from a given file path and organizes them into two DataFrames:
    one containing the first moments (e.g., mean) of the parameters for each session and metric,
    and another containing all parameters for each session and metric.

    Args:
        mle_params_file_path (str): Path to the file containing MLE parameters. Default is set to a specific path.

    Returns:
        tuple: A tuple containing two pandas DataFrames:
            - mle_data_set_first_moments: DataFrame with the first moments of the parameters for each session and metric.
            - mle_data_set_all_moments: DataFrame with all parameters for each session and metric.
    """
    # Load necessary metrics
    key_metrics = SUMMARIZED_METRICS
    
    # Load MLE data from file
    mle_data = torch.load(mle_params_file_path)
    
    # Initialize lists to store session data
    all_sessions_list_first_moments = []
    all_sessions_list_all_params = []
    
    # Loop through each session in MLE data
    for session, session_data in mle_data.items():
        session_dict_first_moments = {"session": session}
        session_dict_all_params = {"session": session}
        
        # Extract first moment for each metric of the session
        for metric in key_metrics:
            if session_data.get(metric) is None:
                session_dict_first_moments[metric] = ""
            else:
                params = torch.tensor(session_data.get(metric))
                session_dict_first_moments[metric] = params[0].item()
                
                # Store all parameters for each metric
                for k in range(params.shape[0]):
                    session_dict_all_params[f"{metric}_param{k+1}"] = params[k].item()
        
        # Append session data to lists
        all_sessions_list_first_moments.append(session_dict_first_moments)
        all_sessions_list_all_params.append(session_dict_all_params)

    # Create DataFrames from the session data lists
    mle_data_set_first_moments = pd.DataFrame(all_sessions_list_first_moments)
    mle_data_set_all_moments = pd.DataFrame(all_sessions_list_all_params)

    return mle_data_set_first_moments, mle_data_set_all_moments
 
