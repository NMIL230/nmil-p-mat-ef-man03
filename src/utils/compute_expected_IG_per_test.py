import torch

softplus = torch.nn.Softplus()


def metrics_to_fidxs_artificial_data(metric):
    # Assumes 2 parameters per distribution and numeric metrics [1,2,3, ... ]
    return [2 * metric - 2, 2 * metric - 1]


def unifrom_softplus(metric, raw_params):
    # NOTE:
    # We make the simple assumption that we should apply softplus activation to all parameters uniformly
    # But any activation could be used for each parameter,
    # and differrent activations can be used for different metrics/params
    # ie distributions wherer parameters must be in [0,1] may require sigmoid activation, etc.
    # (hence whe we assume that any apply_activation function also takes in the metric id)
    return softplus(raw_params)


def get_IG_per_test(
    current_posterior_beleif_latent_dist,
    model,
    test_metrics,
    metrics_to_dist_dict,
    metrics_to_fidxs=metrics_to_fidxs_artificial_data,
    apply_activation=unifrom_softplus,
    T=10,
    Mv=10,
    My=200,
):

    IG_per_test = []
    for metric in test_metrics:
        # Draw T of z samples from latent dist
        zs = current_posterior_beleif_latent_dist.rsample((T,)).squeeze()

        # Passs zs through model to get multivariate normal
        svgp_dist = model(zs)

        # Draw Mv samples from multivariate normaal (svgp dist)
        svgp_samples_per_sample = []
        for _ in range(Mv):
            samps = svgp_dist.sample().unsqueeze(-3)
            svgp_samples_per_sample.append(samps)
        fs_per_Z = torch.cat(svgp_samples_per_sample)

        # Get paramerizers for dist. (using fidxs = indexes that parameterize dist for particular metric)
        dist_params = apply_activation(metric, fs_per_Z[:, :, metrics_to_fidxs(metric)])
        dist_params = [dist_params[:, :, 0], dist_params[:, :, 1]]
        # get dists parameterized by each set of params
        all_dists = metrics_to_dist_dict[metric](*dist_params)

        # Sampe My samples from each dist
        y_samples = all_dists.sample((My,))

        # Compute probability of each y sample given each theta (T,Mv dists)
        #   (For each of the   My,Mv,T y_samps, we have a probability given each of the (Mv,T) distributions )
        probs_all_dists = all_dists.log_prob(y_samples.unsqueeze(-1).unsqueeze(-1))

        # Now average over all dists from the same z (ie average over Mv dimension to get one prob per zi for each y)
        probs_all_dists = torch.logsumexp(
            probs_all_dists - torch.log(torch.tensor(Mv)), -2
        )

        # Gather the probability of each y_sample given the zi it was sampled from (probs given home dists from 'home' z)
        probs_home_dists = torch.cat(
            [probs_all_dists[:, :, ix, ix].unsqueeze(-1) for ix in range(T)], dim=-1
        )

        # Compute the average probability of each y_sample over all z1, z2, ..., zT
        avg_probs_all_dists = torch.logsumexp(
            probs_all_dists - torch.log(torch.tensor(T)), -1
        )

        # Compute Expected IG from obtaining a datapoint for this metric
        diff = probs_home_dists - avg_probs_all_dists
        # First average over Mv,My
        avg_IG = (1 / (My * Mv)) * (diff.sum(dim=0).sum(dim=0))
        # Then average over T
        avg_IG = (1 / T) * avg_IG.sum()

        IG_per_test.append(avg_IG)

    return torch.tensor(IG_per_test).float()
