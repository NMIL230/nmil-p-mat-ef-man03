from difflib import diff_bytes
import sys

sys.path.append("../")
from utils.generate_artificial_clustered_data import generate_2D_clustered_data
from utils.compute_expected_IG_per_test import (
    get_IG_per_test as get_KL_per_cognitive_test,
)
import torch
from utils.exact_GPLVM import exact_gplvm
from gpytorch.priors import NormalPrior
import numpy as np
import torch
import argparse
import time
import random
import os


def _maybe_import_wandb():
    """Attempt to import wandb, returning the module or None if unavailable."""
    try:
        import wandb  # type: ignore
    except ModuleNotFoundError:
        return None

    os.environ.setdefault("WANDB_SILENT", "true")
    return wandb


def _log_tracker(tracker, payload):
    """Safely log metrics when a tracker is available."""
    if tracker is not None:
        tracker.log(payload)


softplus = torch.nn.Softplus()

# For a more complete tutorial of this experiment, see DALE_example.ipynb

METRICS = [1, 2, 3, 4, 5, 6]

metrics_to_dist_dict = {}
metrics_to_dist_dict[1] = torch.distributions.log_normal.LogNormal
metrics_to_dist_dict[2] = torch.distributions.log_normal.LogNormal
metrics_to_dist_dict[3] = torch.distributions.normal.Normal
metrics_to_dist_dict[4] = torch.distributions.normal.Normal
metrics_to_dist_dict[5] = torch.distributions.beta.Beta
metrics_to_dist_dict[6] = torch.distributions.beta.Beta


def compute_mle_ratio(
    curr_posterior_beleif_latent_dist,
    model,
    data_dict,
    all_metrics,
    best_mle_params_obs,
    id_str="",
    n_samples=1,
    verbose=False,
):

    latent_points = curr_posterior_beleif_latent_dist.rsample((n_samples,))
    # mean_z = curr_posterior_beleif_latent_dist.loc  # for mean 2 ...
    latent_points = latent_points.reshape(-1, latent_points.shape[-1])

    svgp_dist = model(latent_points.cuda())
    svgp_samples_per_sample = []
    for _ in range(n_samples):
        svgp_samples_per_sample.append(svgp_dist.rsample().unsqueeze(-3))
    f = (
        torch.cat(svgp_samples_per_sample)
        .reshape(n_samples, n_samples, -1, model.out_sz)
        .reshape(n_samples * n_samples, -1, model.out_sz)
    )  # --> Sdist x Ssvgp x N x O --> Sdist*Ssvgp x N x O
    f_mean = f.mean(axis=0).unsqueeze(0)

    total_log_prob_data = 0
    mle_log_prob_data_check = 0
    mean_tot_log_prob_data = 0

    return_dict = {}
    for metric in all_metrics:
        return_dict[id_str + "raw_log_prob_" + str(metric)] = 0

    for metric in all_metrics:
        fidxs = [2 * metric - 2, 2 * metric - 1]
        data = torch.tensor(data_dict[metric]).float()  # tmp data shape: T x 1 x N
        dist_params = softplus(f[:, :, fidxs])  # .tolist()
        dist_params = [dist_params[:, :, 0], dist_params[:, :, 1]]
        dist = metrics_to_dist_dict[metric](*dist_params)

        # Mean version 1: Take mean of all f params gotten by sampling and passing through model
        mean_params = softplus(f_mean[:, :, fidxs])  # .tolist()
        mean_params = [mean_params[:, :, 0], mean_params[:, :, 1]]
        mean_dist = metrics_to_dist_dict[metric](*mean_params)
        mean_tot_log_prob_data += mean_dist.log_prob(data.cuda()).sum()

        log_prob_metric = dist.log_prob(data.cuda()).sum() / (n_samples * n_samples)
        total_log_prob_data += log_prob_metric
        return_dict[id_str + "raw_log_prob_" + str(metric)] = log_prob_metric

        # also compute mle ratio to mle itself for direct comparison (SANITY CHECK)
        mle_dist_params = best_mle_params_obs[fidxs].tolist()
        mle_dist = metrics_to_dist_dict[metric](*mle_dist_params)
        mle_log_prob_data_check += mle_dist.log_prob(data).sum()  # /n_samples

    mle_rat = total_log_prob_data.detach().cpu() / mle_log_prob_data_check
    if mle_log_prob_data_check < 0:  # if it's negative, want smaller negative number
        mle_rat = 1 / mle_rat
    mle_rat = max(mle_rat, 0)  # if only total log prob is negative
    if verbose:
        print("mle ratio:", mle_rat)

    for metric in all_metrics:
        lp_metric = return_dict[id_str + "raw_log_prob_" + str(metric)]
        return_dict[id_str + "prcnt_total_log_prob_from_" + str(metric)] = (
            lp_metric / total_log_prob_data
        )

    return_dict[id_str + "total_log_prob_data"] = total_log_prob_data
    return_dict[id_str + "mle_ratio"] = mle_rat

    return_dict[id_str + "mean_tot_log_prob_data"] = mean_tot_log_prob_data
    return_dict[id_str + "mean_mle_ratio"] = max(
        mean_tot_log_prob_data.detach().cpu() / mle_log_prob_data_check, 0
    )
    return_dict[id_str + "mle_log_prob_data_check"] = mle_log_prob_data_check

    return return_dict


def update_latent_dist_from_data(
    update_w_data,
    max_epochs=600,
    lr=0.01,
    model=None,
    max_n_progress_fails=300,
    prior_x=None,
    meu_z=None,
    sigma_z=None,
    tracker=None,
    n_samples=10,
    iteration=0,
    grad_clip=0.2,
    min_allowed_log_prob=-1000,
    random_baseline=False,
    oracle_baseline=False,
    debug=False,
    train_w_kld1=False,
    train_w_kld2=False,
):

    if prior_x is None:
        latent_dim = meu_z.shape[1]
        prior_x = NormalPrior(torch.zeros(1, latent_dim), torch.ones(1, latent_dim))

    optimizer = torch.optim.Adam([meu_z, sigma_z], lr=lr)
    progress_fails = 0
    lowest_loss = np.inf
    if oracle_baseline:
        max_epochs = max(max_epochs, 5_000)
        max_n_progress_fails = max(max_n_progress_fails, 2_000)
    for e in range(max_epochs):
        model.gp = (
            model.gp.train()
        )  # clears cache to prevent backward through graph error
        model.gp = model.gp.eval()

        optimizer.zero_grad()
        latent_dist = torch.distributions.Normal(
            meu_z, torch.nn.functional.softplus(sigma_z)
        )
        latent_points = latent_dist.rsample((n_samples,))  # n_samples x 1 x Latent_dim
        latent_points = latent_points.reshape(
            -1, latent_points.shape[-1]
        )  # n_samples x Latent_dim

        svgp_dist = model(latent_points.cuda())
        svgp_samples_per_sample = []
        for _ in range(n_samples):
            samp = svgp_dist.rsample()
            svgp_samples_per_sample.append(samp.unsqueeze(-3))
        f = torch.cat(svgp_samples_per_sample)
        f = f.reshape(n_samples, n_samples, -1, model.out_sz).reshape(
            n_samples * n_samples, -1, model.out_sz
        )  # --> Sdist x Ssvgp x N x O --> Sdist*Ssvgp x N x O

        total_log_prob_data = 0
        norm_const = 0
        for iy, metric in enumerate(update_w_data.keys()):
            data = update_w_data[metric]
            if len(data) > 0:
                data = torch.tensor(data).float()
                fidxs = [2 * metric - 2, 2 * metric - 1]
                norm_const += len(data)
                dist_params = softplus(f[:, :, fidxs])
                dist_params = [dist_params[:, :, 0], dist_params[:, :, 1]]
                dist = metrics_to_dist_dict[metric](*dist_params)
                probs = dist.log_prob(data.cuda())
                probs = torch.clamp(probs, min=min_allowed_log_prob)
                log_prob_task = probs.sum() / (n_samples * n_samples)
                total_log_prob_data += log_prob_task

        norm_const = max(1, norm_const)
        total_log_prob_data = total_log_prob_data / (norm_const)

        KLD1, KLD2 = 0.0, 0.0
        if train_w_kld1:
            KLD1 = torch.distributions.kl_divergence(latent_dist, prior_x).mean()
        if train_w_kld2:
            KLD2 = (
                model.gp.variational_strategy.kl_divergence()
                .div(n_samples * n_samples)
                .sum()
            )
        loss = -total_log_prob_data + KLD1 + KLD2
        loss.backward()

        torch.nn.utils.clip_grad_norm_([meu_z, sigma_z], grad_clip)
        optimizer.step()

        if random_baseline:
            prefix = f"RAND_train{iteration}_"
        elif oracle_baseline:
            prefix = f"ORACLE_train{iteration}_"
        else:
            prefix = f"train{iteration}_"
        _log_tracker(
            tracker,
            {
                prefix + "norm_const": norm_const,
                prefix + "log_prob_data": total_log_prob_data,
                prefix + "loss": loss.item(),
                prefix + "KLD1": KLD1,
                prefix + "KLD2": KLD2,
                prefix + "epoch_num": e + 1,
                prefix + "progresss_fails": progress_fails,
                prefix + "lowest_loss": lowest_loss,
            },
        )

        if loss.item() < lowest_loss:  # if progress
            lowest_loss = loss.item()
            progress_fails = 0
            best_meu_z = meu_z
            best_sigma_z = sigma_z
        else:
            progress_fails += 1
        if progress_fails > max_n_progress_fails:
            break

    return (
        torch.distributions.Normal(
            best_meu_z, torch.nn.functional.softplus(best_sigma_z)
        ),
        best_meu_z,
        best_sigma_z,
    )


def active_learning(
    N=1000,
    n_clusters=3,
    held_out_session_idx=0,
    seed=0,
    latent_dim=2,
    lr=0.01,  # 0.005,
    T=200,
    Mv=20,
    My=20,
    verbose=False,
    test_budget=-1,
    version=2,
    max_epcohs=10_000,  # 2_000
    max_n_progress_fails=2_000,  # 500
    test_batch_size=1,
    track_w_wandb=False,
    relearn_latent_dim_from_scratch=False,
    grad_clip=0.3,
    min_allowed_log_prob=-500,
    add_random_baseline=True,
    oracle_is_raw_data=True,
    n_train_samples=10,
    n_samples_comp_mle_ratio=10,
    train_w_kld1=False,
    train_w_kld2=False,
    stdev_artificial_data=1.0,
    N_MLE_POINTS=100_000,
):

    if test_budget == -1:
        test_budget = 180

    # get artificial data means, will generate normal stds with std = 1 (or other hyperparamteer )
    n_artificial_distribution_parameters = len(METRICS) * 2
    # torch.manual_seed(seed)

    X, Y, cluster_centers = generate_2D_clustered_data(
        N=N, n_clusters=n_clusters, seed=seed
    )
    X_heldout = torch.tensor(X[held_out_session_idx]).float().cuda()
    Y_heldout = Y[held_out_session_idx]  # cluster label
    X_not_heldout = (
        torch.cat(
            [
                torch.tensor(X[0:held_out_session_idx]),
                torch.tensor(X[held_out_session_idx + 1 :]),
            ]
        )
        .float()
        .cuda()
    )
    Y_not_heldout = (
        torch.cat(
            [
                torch.tensor(Y[0:held_out_session_idx]),
                torch.tensor(Y[held_out_session_idx + 1 :]),
            ]
        )
        .float()
        .cuda()
    )
    Y_not_heldout = Y_not_heldout.unsqueeze(-1)
    if verbose:
        print(
            "X_not_heldout", X_not_heldout.shape, "y not heldout", Y_not_heldout.shape
        )

    model = exact_gplvm(
        X_not_heldout,
        torch.randn(X_not_heldout.shape[0], n_artificial_distribution_parameters),
        out_sz=n_artificial_distribution_parameters,
    )
    model = model.eval()

    for param in model.parameters():
        param.requires_grad = False

    # true_ho_dist_params = model(X_heldout.unsqueeze(0) ).sample()  # torch.Size([1, 8])
    true_ho_dist_params = model(X_heldout.unsqueeze(0)).loc
    true_ho_dist_params = softplus(
        true_ho_dist_params
    )  # softplus everything by default now

    tracker = None
    wandb_lib = None
    if track_w_wandb:
        wandb_lib = _maybe_import_wandb()
        if wandb_lib is None:
            raise RuntimeError(
                "wandb is not installed but tracking was requested via --track_w_wandb"
            )

        tracker = wandb_lib.init(
            entity="nmaus",
            project="EF-MATH-GPLVM-active-learning-artificial-data",
            config={
                "lr": lr,
                "T": T,
                "Mv": Mv,
                "My": My,
                "N": N,
                "n_clusters": n_clusters,
                "seed": seed,
                "n_artificial_distribution_parameters": n_artificial_distribution_parameters,
                "held_out_session_num": held_out_session_idx,
                "latent_dim": latent_dim,
                "test_budget": test_budget,
                "max_epcohs": max_epcohs,
                "max_n_progress_fails": max_n_progress_fails,
                "test_batch_size": test_batch_size,
                "relearn_latent_dim_from_scratch": relearn_latent_dim_from_scratch,
                "add_random_baseline": add_random_baseline,
                "grad_clip": grad_clip,
                "oracle_is_raw_data": oracle_is_raw_data,
                "min_allowed_log_prob": min_allowed_log_prob,
                "version": version,
                "train_w_kld1": train_w_kld1,
                "train_w_kld2": train_w_kld2,
                "n_train_samples": n_train_samples,
                "relevant only": True,
                "ARTIFICIAL-DATA": True,
                "N_MLE_POINTS": N_MLE_POINTS,
            },
        )

        if getattr(wandb_lib, "run", None) is not None:
            print("running", wandb_lib.run.name)

    _log_tracker(tracker, {"z1_heldout": X_heldout[0], "z2_heldout": X_heldout[1]})

    prior_x = NormalPrior(torch.zeros(1, latent_dim), torch.ones(1, latent_dim))
    meu_z = torch.nn.Parameter(torch.zeros(1, latent_dim))
    sigma_z = torch.nn.Parameter(torch.ones(1, latent_dim))

    _log_tracker(
        tracker,
        {
            "num_tests_run": 0,
            "z1": meu_z.squeeze()[0].item(),
            "z2": meu_z.squeeze()[1].item(),
            "sigma1": sigma_z.squeeze()[0].item(),
            "sigma2": sigma_z.squeeze()[1].item(),
        },
    )
    print("INITIAL MEU AND SIGMA:")
    print(
        "z1",
        meu_z.squeeze()[0].item(),
        "z2",
        meu_z.squeeze()[1].item(),
        "sigma1",
        sigma_z.squeeze()[0].item(),
        "sigma2",
        sigma_z.squeeze()[1].item(),
    )

    if add_random_baseline:
        RAND_meu_z = torch.nn.Parameter(torch.zeros(1, latent_dim))
        RAND_sigma_z = torch.nn.Parameter(torch.ones(1, latent_dim))

    curr_posterior_beleif_latent_dist, RAND_posterior_beleif_latent_dist = (
        prior_x,
        prior_x,
    )
    ordered_tests_chosen_to_run, RAND_ordered_tests_chosen_to_run = [], []

    update_w_data = {}  # keys are metrics, values are lists of simulated data gaathered
    RAND_update_w_data = {}  # ranodm baseline
    ORACLE_update_w_data = {}
    for metric in METRICS:
        update_w_data[metric] = []
        RAND_update_w_data[metric] = []
        ORACLE_update_w_data[metric] = []

    # FIRST, DO COMPARITIVE ANALYSIS W/ 'ORACLE' == access to all data
    for metric in METRICS:
        ORACLE_update_w_data[metric] += get_data_list_from_mle_data_generator(
            metric, true_ho_dist_params, N_MLE_POINTS, verbose=verbose
        )

    all_mle_ratios, expected_IGs_of_each_chosen_test = [], []
    RAND_all_mle_ratios, RAND_expected_IGs_of_each_chosen_test = [], []

    ORACLE_meu_z = torch.nn.Parameter(torch.zeros(1, latent_dim))
    ORACLE_sigma_z = torch.nn.Parameter(
        torch.ones(1, latent_dim) * 0.545
    )  # softpluts will make 0.545 ~ 1.0

    (
        ORACLE_posterior_beleif_latent_dist,
        ORACLE_meu_z,
        ORACLE_sigma_z,
    ) = update_latent_dist_from_data(
        ORACLE_update_w_data,
        max_epcohs,
        lr,
        model,
        max_n_progress_fails,
        prior_x,
        ORACLE_meu_z,
        ORACLE_sigma_z,
        tracker=tracker,
        n_samples=n_train_samples,
        iteration=0,
        grad_clip=grad_clip,
        min_allowed_log_prob=min_allowed_log_prob,
        random_baseline=False,
        oracle_baseline=True,
        train_w_kld1=train_w_kld1,
        train_w_kld2=train_w_kld2,
    )

    _log_tracker(
        tracker,
        {
            "ORACLE_meu_z1": ORACLE_meu_z.squeeze()[0],
            "ORACLE_meu_z2": ORACLE_meu_z.squeeze()[1],
        },
    )

    oracle_return_dict = compute_mle_ratio(
        ORACLE_posterior_beleif_latent_dist,
        model,
        ORACLE_update_w_data,
        METRICS,
        true_ho_dist_params,
        id_str="ORACLE_",
        n_samples=n_samples_comp_mle_ratio,
    )

    _log_tracker(tracker, oracle_return_dict)

    all_active_meu_z = [meu_z]
    all_rand_meu_z = [RAND_meu_z]
    for i in range(test_budget):
        # GET MLE RATIO FOR RANOM VERSION TOO!
        if add_random_baseline:
            rand_return_dict = compute_mle_ratio(
                RAND_posterior_beleif_latent_dist,
                model,
                ORACLE_update_w_data,
                METRICS,
                true_ho_dist_params,
                id_str="RAND_",
                n_samples=n_samples_comp_mle_ratio,
            )
            rand_return_dict["num_tests_run"] = i
            _log_tracker(tracker, rand_return_dict)

        return_dict = compute_mle_ratio(
            curr_posterior_beleif_latent_dist,
            model,
            ORACLE_update_w_data,
            METRICS,
            true_ho_dist_params,
            id_str="",
            n_samples=n_samples_comp_mle_ratio,
        )
        return_dict["num_tests_run"] = i
        _log_tracker(tracker, return_dict)

        if i == 0:  # log prior mle ratio --> find outliers
            _log_tracker(tracker, {"prior_mle_ratio": return_dict["mle_ratio"]})

        all_mle_ratios.append(
            float(return_dict["mle_ratio"])
        )  # type cast in case its a tensor
        if add_random_baseline:
            RAND_all_mle_ratios.append(
                float(rand_return_dict["RAND_mle_ratio"])
            )  # type cast in case its a tensor

        # Log table of chosen test periodically
        if (tracker is not None) and (((i + 1) % 10 == 0) or (i >= (test_budget - 1))):
            log_chosen_tests_in_table(
                tracker,
                wandb_lib,
                ordered_tests_chosen_to_run,
                expected_IGs_of_each_chosen_test,
                all_mle_ratios,
                all_active_meu_z,
            )
            if add_random_baseline:
                log_chosen_tests_in_table(
                    tracker,
                    wandb_lib,
                    RAND_ordered_tests_chosen_to_run,
                    RAND_expected_IGs_of_each_chosen_test,
                    RAND_all_mle_ratios,
                    all_rand_meu_z,
                    rand_baseline=True,
                )

        start = time.time()
        KL_per_test = get_KL_per_cognitive_test(
            curr_posterior_beleif_latent_dist, model, METRICS, metrics_to_dist_dict
        )
        _log_tracker(tracker, {"get_kl_per_test_time": time.time() - start})

        if add_random_baseline:
            RAND_KL_per_test = get_KL_per_cognitive_test(
                RAND_posterior_beleif_latent_dist, model, METRICS, metrics_to_dist_dict
            )

        if tracker is not None:
            for kl, test in zip(KL_per_test, METRICS):
                _log_tracker(tracker, {str(test) + "_KL": kl})

        # Run test with highest expected IG
        test_to_run_next = METRICS[KL_per_test.argmax().item()]
        expected_IG_of_next_test = KL_per_test.max().item()
        update_w_data[test_to_run_next] += get_data_list_from_mle_data_generator(
            test_to_run_next, true_ho_dist_params, test_batch_size, verbose=verbose
        )

        # log stuff:
        idx_for_next_test = METRICS.index(test_to_run_next)
        expected_IG_of_next_test = KL_per_test[idx_for_next_test]
        print(
            f"Chosen Test Number {i+1}/{test_budget}: {test_to_run_next}, Expected IG: {expected_IG_of_next_test}"
        )
        ordered_tests_chosen_to_run.append(test_to_run_next)
        _log_tracker(tracker, {f"expected_IG_chosen_test_next": expected_IG_of_next_test})
        expected_IGs_of_each_chosen_test.append(expected_IG_of_next_test)

        assert not relearn_latent_dim_from_scratch

        if relearn_latent_dim_from_scratch:
            meu_z = torch.nn.Parameter(torch.zeros(1, latent_dim))
            sigma_z = torch.nn.Parameter(
                torch.ones(1, latent_dim) * 0.545
            )  # softpluts will make 0.545 ~ 1.0

        start = time.time()

        print("")
        print("BEFORE UPDATE:")
        print("meuz", meu_z, "sigma z", sigma_z)
        print("")
        (
            curr_posterior_beleif_latent_dist,
            meu_z,
            sigma_z,
        ) = update_latent_dist_from_data(
            update_w_data,
            max_epcohs,
            lr,
            model,
            max_n_progress_fails,
            prior_x,
            meu_z,
            sigma_z,
            tracker=tracker,
            n_samples=n_train_samples,
            iteration=i,
            grad_clip=grad_clip,
            min_allowed_log_prob=min_allowed_log_prob,
            train_w_kld1=train_w_kld1,
            train_w_kld2=train_w_kld2,
        )

        all_active_meu_z.append(meu_z)
        _log_tracker(
            tracker,
            {
                "num_tests_run": i + 1,
                "z1": meu_z.squeeze()[0].item(),
                "z2": meu_z.squeeze()[1].item(),
                "sigma1": sigma_z.squeeze()[0].item(),
                "sigma2": sigma_z.squeeze()[1].item(),
            },
        )
        print("AFTER UPDATE:")
        print("meuz", meu_z, "sigma z", sigma_z)

        _log_tracker(tracker, {"update_posterior_beleif_time": time.time() - start})

        if add_random_baseline:
            random_idx = random.randint(0, len(METRICS) - 1)
            RAND_test_to_run_next = METRICS[random_idx]
            RAND_update_w_data[
                RAND_test_to_run_next
            ] += get_data_list_from_mle_data_generator(
                RAND_test_to_run_next,
                true_ho_dist_params,
                test_batch_size,
                verbose=verbose,
            )

            # log stuff
            idx_for_rand_test_kl = METRICS.index(RAND_test_to_run_next)
            RAND_expected_IG_of_next_test = RAND_KL_per_test[idx_for_rand_test_kl]
            RAND_ordered_tests_chosen_to_run.append(RAND_test_to_run_next)
            _log_tracker(
                tracker,
                {
                    f"RAND_expected_IG_chosen_test_next": RAND_expected_IG_of_next_test
                },
            )
            RAND_expected_IGs_of_each_chosen_test.append(RAND_expected_IG_of_next_test)

            if relearn_latent_dim_from_scratch:
                RAND_meu_z = torch.nn.Parameter(torch.zeros(1, latent_dim))
                RAND_sigma_z = torch.nn.Parameter(
                    torch.ones(1, latent_dim) * 0.545
                )  # softpluts will make 0.545 ~ 1.0

            (
                RAND_posterior_beleif_latent_dist,
                RAND_meu_z,
                RAND_sigma_z,
            ) = update_latent_dist_from_data(
                RAND_update_w_data,
                max_epcohs,
                lr,
                model,
                max_n_progress_fails,
                prior_x,
                RAND_meu_z,
                RAND_sigma_z,
                tracker=tracker,
                n_samples=n_train_samples,
                iteration=i,
                grad_clip=grad_clip,
                min_allowed_log_prob=min_allowed_log_prob,
                random_baseline=True,
                train_w_kld1=train_w_kld1,
                train_w_kld2=train_w_kld2,
            )

            all_rand_meu_z.append(RAND_meu_z)

    # log_meu_zs_table(all_active_meu_z, all_rand_meu_z)

    if tracker is not None:
        tracker.finish()

    return ordered_tests_chosen_to_run, all_mle_ratios


def get_data_list_from_mle_data_generator(
    metric, best_mle_params_obs, test_batch_size, verbose=False
):
    fidxs = [2 * metric - 2, 2 * metric - 1]

    mle_dist_params = best_mle_params_obs[fidxs]
    mle_dist = metrics_to_dist_dict[metric](*mle_dist_params)

    simulated_data_points = []

    n_avg_over = 200
    for i in range(test_batch_size):
        simulated_data_samples = mle_dist.sample((n_avg_over,))  # .tolist()
        simulated_data_points.append(simulated_data_samples.mean())

    if verbose:
        prob_sim_data = mle_dist.log_prob(
            torch.tensor(simulated_data_points).float().cuda()
        ).sum() / len(simulated_data_points)
        print(f"metric: {metric}, avg probs simulated data: {prob_sim_data}")
    return simulated_data_points


def log_chosen_tests_in_table(
    tracker,
    wandb_module,
    ordered_tests_chosen_to_run,
    expected_IGs_of_each_chosen_test,
    all_mle_ratios,
    meu_z_list,
    rand_baseline=False,
):
    if tracker is None or wandb_module is None:
        return
    cols = [
        "Chosen Test",
        "Expected IG",
        "MLE Ratio After Adding Test Data",
        "Meu Z1",
        "Meu Z2",
    ]
    data_list = []
    init_z = meu_z_list[0].squeeze()
    init_z1, init_z2 = init_z[0], init_z[1]
    data_list.append(["None", 0.0, all_mle_ratios[0], init_z1, init_z2])
    for ix, test in enumerate(ordered_tests_chosen_to_run):
        exIG = expected_IGs_of_each_chosen_test[ix]
        mlerat = all_mle_ratios[ix + 1]
        meu_z = meu_z_list[ix + 1].squeeze()
        z1, z2 = meu_z[0], meu_z[1]
        if type(exIG) != float:
            exIG = exIG.item()
        if type(mlerat) != float:
            mlerat = mlerat.item()

        data_list.append([str(test), exIG, mlerat, z1, z2])

    chosen_tests_table = wandb_module.Table(columns=cols, data=data_list)

    if rand_baseline:
        _log_tracker(tracker, {f"RAND_chosen_tests_table": chosen_tests_table})
    else:
        _log_tracker(tracker, {f"chosen_tests_table": chosen_tests_table})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--held_out_session_num", type=int, default=-1)
    parser.add_argument("--start_session", type=int, default=0)
    parser.add_argument("--stop_session", type=int, default=20)
    parser.add_argument("--latent_dim", type=int, default=13)
    parser.add_argument("--test_budget", type=int, default=-1)
    parser.add_argument("--test_bsz", type=int, default=1)
    parser.add_argument(
        "--relearn_latent_dim_from_scratch", type=bool, default=False
    )  # dist*
    parser.add_argument("--grad_clip", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--max_n_progress_fails", type=int, default=200)  # 100
    parser.add_argument("--max_epcohs", type=int, default=500)  # 300
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--Mv", type=int, default=20)
    parser.add_argument("--My", type=int, default=20)
    parser.add_argument("--n_train_samples", type=int, default=10)
    parser.add_argument("--n_samples_comp_mle_ratio", type=int, default=10)
    parser.add_argument("--track_w_wandb", type=bool, default=False)
    parser.add_argument("--train_w_kld1", type=bool, default=False)
    parser.add_argument("--train_w_kld2", type=bool, default=False)
    parser.add_argument(
        "--use_mle_generator_for_data", type=bool, default=True
    )  # False --> use actual data

    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--n_clusters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--stdev_artificial_data", type=float, default=1.0)
    parser.add_argument("--N_MLE_POINTS", type=int, default=10_000)
    parser.add_argument("--min_allowed_log_prob", type=float, default=-1000)

    args = parser.parse_args()

    if args.held_out_session_num == -1:
        heldout_session_nums = np.arange(args.start_session, args.stop_session + 1)
    else:
        heldout_session_nums = [args.held_out_session_num]

    for heldout_session_num in heldout_session_nums:
        _, _ = active_learning(
            N=args.N,
            n_clusters=args.n_clusters,
            held_out_session_idx=heldout_session_num,
            seed=args.seed,
            latent_dim=2,
            lr=args.lr,
            T=args.T,
            Mv=args.Mv,
            My=args.My,
            verbose=args.verbose,
            test_budget=args.test_budget,
            max_epcohs=args.max_epcohs,  # 2_000
            max_n_progress_fails=args.max_n_progress_fails,  # 500
            test_batch_size=args.test_batch_size,
            track_w_wandb=True,
            relearn_latent_dim_from_scratch=args.relearn_latent_dim_from_scratch,
            grad_clip=args.grad_clip,
            min_allowed_log_prob=args.min_allowed_log_prob,
            add_random_baseline=True,
            oracle_is_raw_data=True,
            n_train_samples=10,
            n_samples_comp_mle_ratio=args.n_samples_comp_mle_ratio,
            train_w_kld1=False,
            train_w_kld2=False,
            stdev_artificial_data=args.stdev_artificial_data,
            N_MLE_POINTS=args.N_MLE_POINTS,
        )


if __name__ == "__main__":
    main()
