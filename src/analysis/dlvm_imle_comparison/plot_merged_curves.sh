#!/bin/bash

ground_truth_dir="../create_synthetic_individuals/synthetic_sessions_ground_truth/simulated_data"
params_dir="./fitted_parameters/COLL10_SIM/"

# figure 4
python3 plot_merged_curves.py --ground_truth_dir "$ground_truth_dir" --params_dir "$params_dir" --metric "kld" --normalize_errors --plot_std_dev --eval_dataset_type "validation_simulated" --show_single_dlvm_plot --show_single_imle_plot

# python3 plot_merged_curves.py --ground_truth_dir "$ground_truth_dir" --params_dir "$params_dir" --metric "kld" --normalize_errors --plot_std_dev --eval_dataset_type "validation_simulated" --show_single_imle_plot
# python3 plot_merged_curves.py --ground_truth_dir "$ground_truth_dir" --params_dir "$params_dir" --metric "rmse" --plot_std_dev --normalize_errors --eval_dataset_type "validation_simulated"


# d2_ground_truth_pt_file="$ground_truth_dir/D2_synthetic_ground_truth_parameters_mongoose-dive-7464.pt"
# d3_ground_truth_pt_file="$ground_truth_dir/D3_synthetic_ground_truth_parameters_beaver-slide-5310.pt"
# d1_ground_truth_pt_file="$ground_truth_dir/D1_synthetic_ground_truth_parameters_wolverine-zoom-7298.pt"

# params_dir="./fitted_parameters/COLL10/"
# d2_params_dir="$params_dir/D2_synthetic_ground_truth_parameters_mongoose-dive-7464"
# d3_params_dir="$params_dir/D3_synthetic_ground_truth_parameters_beaver-slide-5310"
# d1_params_dir="$params_dir/D1_synthetic_ground_truth_parameters_wolverine-zoom-7298"

# d2_synthetic_data_dir="../generate_synthetic_item_observations/synthetic_data/COLL10/D2_synthetic_ground_truth_parameters_mongoose-dive-7464/"
# d3_synthetic_data_dir="../generate_synthetic_item_observations/synthetic_data/COLL10/D3_synthetic_ground_truth_parameters_beaver-slide-5310/"
# d1_synthetic_data_dir="../generate_synthetic_item_observations/synthetic_data/COLL10/D1_synthetic_ground_truth_parameters_wolverine-zoom-7298/"


# python3 plot_marginal_median.py --ground_truth_pt_file "$d2_ground_truth_pt_file" --params_dir "$d2_params_dir" --metric "kld" --normalize_errors --synthetic_data_dir "$d2_synthetic_data_dir"
# python3 plot_marginal_median.py --ground_truth_pt_file "$d3_synthetic_data_dir" --params_dir "$d3_params_dir" --metric "rmse" --normalize_errors
# python3 plot_marginal_median.py --ground_truth_pt_file "$d1_synthetic_data_dir" --params_dir "$d1_params_dir" --metric "rmse" --normalize_errors

# now do the same for the training data
ground_truth_dir="../../data/COLL10_SIM/"
# python3 plot_merged_curves.py --ground_truth_dir "$ground_truth_dir" --params_dir "$params_dir" --metric "kld" --normalize_errors --plot_std_dev --eval_dataset_type "training_set" --show_single_imle_plot
# python3 plot_merged_curves.py --ground_truth_dir "$ground_truth_dir" --params_dir "$params_dir" --metric "rmse" --plot_std_dev --normalize_errors --eval_dataset_type "training_set" --show_single_imle_plot

# plot marginal median for the training set (Figure 3)
synthetic_data_dir="../generate_synthetic_item_observations/synthetic_data/COLL10_SIM/all_data-best_mle_params_mpf100/"
ground_truth_pt_file="$ground_truth_dir/all_data-best_mle_params_mpf100.pt"
d2_params_dir="$params_dir/D2_all_data-best_mle_params_mpf100"
python3 plot_marginal_median.py --ground_truth_pt_file "$ground_truth_pt_file" --params_dir "$d2_params_dir" --metric "kld" --normalize_errors --synthetic_data_dir "$synthetic_data_dir" --eval_dataset_type "training_set"
# python3 plot_marginal_median.py --ground_truth_pt_file "$ground_truth_pt_file" --params_dir "$d2_params_dir" --metric "rmse" --normalize_errors --synthetic_data_dir "$synthetic_data_dir" --eval_dataset_type "training_set"
