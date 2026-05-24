import ast
import random
from pathlib import Path

import torch

from nmil_dlvm.utils.active_learning_utils import (
    calculate_kld_given_metric,
    compute_correct_mle_ratio,
    compute_naive_mle_ratio,
    move_distribution_to_cuda,
)
from nmil_dlvm.utils.data_distribution_utils import (
    COMPUTE_DEVICE,
    activation_dict,
    dist_dict,
    mle_params_to_dist,
)


def load_synthetic_data(args, logger, relevant_metrics):
    """Load a synthetic per-session oracle in the DALE synthetic format."""
    synthetic_data_file = args.synthetic_data_file
    logger.info(f"Loading synthetic data from: {synthetic_data_file}")
    all_synthetic_data = torch.load(synthetic_data_file)

    key = args.test_session_id
    if key not in all_synthetic_data:
        key = f"{key}_sim{args.sim_index}"
        if key not in all_synthetic_data:
            raise KeyError(
                f"Neither '{args.test_session_id}' nor '{key}' found in synthetic data. "
                f"Available keys: {list(all_synthetic_data.keys())[:5]}"
            )

    session_data = all_synthetic_data[key]
    if isinstance(session_data, str):
        session_data = ast.literal_eval(session_data)

    synthetic_data_dict = {}
    for metric in relevant_metrics:
        if metric in session_data:
            synthetic_data_dict[metric] = session_data[metric].copy()
        else:
            synthetic_data_dict[metric] = []

    mle_params_file = args.mle_params_file
    logger.info(f"Loading MLE parameters from: {mle_params_file}")
    all_best_mle_params = torch.load(mle_params_file)
    best_mle_params_obs = all_best_mle_params[args.test_session_id]
    logger.info(f"Loaded synthetic oracle for session {args.test_session_id}")
    return best_mle_params_obs, synthetic_data_dict


def generate_primer_sequence(args, summarized_metrics, relevant_complex_span, relevant_simple_span):
    """Generate the fixed primer sequence used by the synthetic DALE workflow."""
    sequence = []
    repetitions = args.primer_sequence_task_repetitions

    for _ in range(repetitions):
        for task in summarized_metrics:
            if "CorsiComplex" in task:
                selected_task = random.choice(relevant_complex_span)
            elif "SimpleSpan" in task:
                selected_task = random.choice(relevant_simple_span)
            else:
                selected_task = task
            sequence.append(selected_task)

    return sequence


def compute_probs_data_and_kld_for_syn_data(
    curr_posterior_belief_latent_dist,
    model,
    synthetic_data_dict,
    all_metrics,
    best_mle_params_obs,
    n_samples=500,
    metrics_dict=None,
):
    """Compute model/MLE log-probability and KLD for synthetic DALE observations."""
    latent_points = curr_posterior_belief_latent_dist.rsample((n_samples,))
    model_outputs = model(latent_points.to(COMPUTE_DEVICE))

    total_log_prob_data = 0.0
    mle_log_prob = 0.0
    total_kld = 0.0

    for metric in all_metrics:
        if metric not in synthetic_data_dict or len(synthetic_data_dict[metric]) == 0:
            continue

        metric_type = metrics_dict[metric]["type"]
        fidxs = metrics_dict[metric]["f_idxs"]
        data_list = synthetic_data_dict[metric]

        if metric_type in {"binary", "binarySpan"}:
            data_sum = sum(data_list)
            counts = len(data_list)
            data = torch.tensor([data_sum], dtype=torch.float).to(COMPUTE_DEVICE)
            counts_tensor = torch.tensor([counts], dtype=torch.float).to(COMPUTE_DEVICE)
        else:
            data = torch.tensor(data_list, dtype=torch.float).to(COMPUTE_DEVICE)
            counts_tensor = torch.tensor([len(data_list)], dtype=torch.float).to(COMPUTE_DEVICE)

        dist_params = activation_dict[metric_type](
            model_outputs[:, :, fidxs],
            counts_tensor,
            metrics_dict[metric]["length"],
        )
        dist = dist_dict[metric_type](*dist_params)
        prob_data = dist.log_prob(data).sum() / n_samples

        if "Complex" in metric:
            mle_dist_params = best_mle_params_obs["CorsiComplex"]
        elif "Simple" in metric:
            mle_dist_params = best_mle_params_obs["SimpleSpan"]
        else:
            mle_dist_params = best_mle_params_obs[metric]

        if metric_type in {"binary", "binarySpan"}:
            mle_dist = mle_params_to_dist(
                metric,
                mle_dist_params,
                metric_type,
                counts=counts_tensor.item(),
            )
        else:
            mle_dist = mle_params_to_dist(
                metric,
                mle_dist_params,
                metric_type,
                counts=1,
            )

        mle_prob = mle_dist.log_prob(data).sum()
        mle_prob = max(mle_prob, prob_data)

        total_log_prob_data += prob_data.item()
        mle_log_prob += mle_prob.item()
        total_kld += calculate_kld_given_metric(move_distribution_to_cuda(mle_dist), dist, metric)

    return total_log_prob_data, mle_log_prob, total_kld


def evaluate_model_fit_performance_for_syn_data(
    curr_posterior_belief_latent_dist,
    model,
    synthetic_data_dict,
    all_metrics,
    best_mle_params_obs,
    n_samples=500,
    metrics_dict=None,
    mle_ratio_type="naive",
):
    """Evaluate DALE posterior fit against synthetic oracle observations."""
    total_log_prob_data, mle_log_prob_data, total_kld = compute_probs_data_and_kld_for_syn_data(
        curr_posterior_belief_latent_dist,
        model,
        synthetic_data_dict,
        all_metrics,
        best_mle_params_obs,
        n_samples=n_samples,
        metrics_dict=metrics_dict,
    )

    performance = {
        "mle_ratio_naive": compute_naive_mle_ratio(total_log_prob_data, mle_log_prob_data),
        "mle_ratio_correct": compute_correct_mle_ratio(total_log_prob_data, mle_log_prob_data),
        "model_log_prob_data": total_log_prob_data,
        "mle_log_prob_data": mle_log_prob_data,
        "total_kld": total_kld,
        "meu_z": curr_posterior_belief_latent_dist.mean.tolist(),
        "sigma_z": curr_posterior_belief_latent_dist.stddev.tolist(),
    }
    return performance


def default_synthetic_data_path(repo_root: Path, dataset: str):
    return repo_root / "analysis" / "dlvm_imle_comparison" / "synthetic_data" / dataset / "all_synthetic_data_N240.pt"

