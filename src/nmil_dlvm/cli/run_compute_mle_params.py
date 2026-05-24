import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Script to generate MLE best params and compute log probabilities."
    )
    parser.add_argument(
        "--generate_new",
        action="store_true",
        default=True,
        help="Generate new files for MLE best params.",
    )
    parser.add_argument(
        "--compute_log_prob_only",
        action="store_true",
        default=False,
        help="Compute new log probabilities.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run in debug mode.",
    )
    args = parser.parse_args()

    import torch

    from nmil_dlvm.paths import data_dir
    from nmil_dlvm.utils.data_distribution_utils import DATASET
    from nmil_dlvm.utils.mle_utils import (
        compute_total_log_probabilities,
        generate_mle_best_params,
    )

    mle_params_path = data_dir(DATASET) / "all_data-best_mle_params_mpf100.pt"
    total_log_prob_path = data_dir(DATASET) / "all_data-best_mle_total_log_prob_mpf100.pt"

    if not args.compute_log_prob_only:
        result = generate_mle_best_params(debug=args.debug)

        if args.generate_new:
            current_params = {}
        else:
            current_params = torch.load(mle_params_path, weights_only=False)

        for session in result.keys():
            current_params[session] = result[session]
        torch.save(current_params, mle_params_path)
    else:
        print("Computing log probabilities only.")

    log_probs = compute_total_log_probabilities(debug=args.debug)

    if args.generate_new:
        current_prob = {}
    else:
        current_prob = torch.load(total_log_prob_path, weights_only=False)

    for session in log_probs.keys():
        current_prob[session] = log_probs[session]
    torch.save(current_prob, total_log_prob_path)


if __name__ == "__main__":
    main()
