#!/bin/bash

# N_values=(50 200)
N_values=(1 2 3 5 )
# Uncomment the line below to use a smaller set for testing
# N_values=(1 2 3 5)

# Limit the number of parallel jobs to 3
max_jobs=1
# mle_params_file="mle_params_4_evaluation_purposes.pt" # for COLL10_SIM
mle_params_file="all_data-best_mle_params_mpf100.pt" # for COLL10

CURRENT_DIR=$(pwd)

base_synthetic_data_dir="$CURRENT_DIR/../generate_synthetic_item_observations/synthetic_data/COLL10_SIM/"

d2_synthetic_data_dir="$base_synthetic_data_dir/D2_synthetic_ground_truth_parameters_mongoose-dive-7464"
d3_synthetic_data_dir="$base_synthetic_data_dir/D3_synthetic_ground_truth_parameters_beaver-slide-5310"
d1_synthetic_data_dir="$base_synthetic_data_dir/D1_synthetic_ground_truth_parameters_wolverine-zoom-7298"

train_data_synthetic_data_dir="$base_synthetic_data_dir/all_data-best_mle_params_mpf100"


experiment_type="training" # or "validation"

latent_dims=( 1 3)
# Loop through each N value and run the command in parallel
for N in "${N_values[@]}"; do
    echo "Fitting models for N = $N"
    (
        # Uncomment if you need to generate simulation data
        # python ../generate_synthetic_item_observations/generate_simulation_data.py --N "$N" --num_synthetic_runs 1 --ground_truth_param_file "$mle_params_file" --use_n_per_task
        
        # Fix: Consistent argument naming (latent-dim vs latent_dim)
        for latent_dim in "${latent_dims[@]}"; do
            eval_dataset_type="validation_simulated"
            if [ "$latent_dim" -eq 2 ]; then
                synthetic_data_dir="$d2_synthetic_data_dir"
                model_id="mongoose-dive-7464"
            elif [ "$latent_dim" -eq 3 ]; then
                synthetic_data_dir="$d3_synthetic_data_dir"
                model_id="beaver-slide-5310"
            elif [ "$latent_dim" -eq 1 ]; then
                synthetic_data_dir="$d1_synthetic_data_dir"
                model_id="wolverine-zoom-7298"
            fi
            if [ "$experiment_type" == "training" ]; then
                # change the synthetic data dir to the training data dir
                synthetic_data_dir="$train_data_synthetic_data_dir"
                eval_dataset_type="training_set"
            fi
            
            python3 fit_dlvm_and_imle_models_to_data.py --fit-dlvm --latent-dim "$latent_dim" --model-id "$model_id" --max_N "$N" --synthetic_data_dir "$synthetic_data_dir" --eval_dataset_type "$eval_dataset_type" --fit-imle
        
        done
        
        # python3 fit_models_on_sim_data.py --fit-imle --max_N "$N"
        # # python3 plot_compare_search.py --ground_truth_param_file "$mle_params_file" --metric kld --normalize_errors
        # # python3 plot_compare_search.py --ground_truth_param_file "$mle_params_file" --metric rmse --normalize_errors
        # # python3 plot_compare_search.py --ground_truth_param_file "$mle_params_file" --metric kld
        # # python3 plot_compare_search.py --ground_truth_param_file "$mle_params_file" --metric rmse
        # # Uncomment if you need to compare IMLE parameters
        # # python compare_imle_params.py --N "$N" --ground_truth_pt_file "all_data-best_mle_params_mpf100.pt"  
    ) &
    
    ((current_jobs++))
    if ((current_jobs >= max_jobs)); then
        wait -n # Wait for any job to finish
        ((current_jobs--))
    fi
done
wait



 
