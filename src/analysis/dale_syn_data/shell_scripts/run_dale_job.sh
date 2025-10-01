#!/bin/bash
# Script 2: Run individual DALE experiment based on configuration
# Echo the host that the job is running on for debugging purposes
echo "Job is running on host: $LSB_HOSTNAME"

echo "$1"

# Read the file name as the part before "." in the argument
config_file=$(echo "$1" | cut -d'.' -f1)
# Read the index as the part after "." in the argument
job_index=$(echo "$1" | cut -d'.' -f2)

submit_mode="${2:-bsub}"

# Add 1 to the job index
job_index=$((job_index+1))
printf "Selected Row ID : %d\n" $job_index

# Read the index row from train_configs/dale_experiments.csv
selected_config=$(sed -n "${job_index}p" train_configs/${config_file}.csv)

# Check if the configuration line is empty
if [ -z "$selected_config" ]; then
    echo "Error: Configuration file is empty."
    exit 1
fi

# Debugging: Print the selected configuration
echo "Selected Configuration: $selected_config"

# Parse the selected configuration into individual parameters
IFS=',' read -r experiment_name dataset N run_mode model_path test_budget num_restarts latent_dim num_eval_sessions session_seed use_synthetic_data random_baseline enable_primer_sequence primer_sequence_task_repetitions mle_params_file synthetic_data_file <<< "$selected_config"

# Log the parsed configuration
echo "=== Parsed Configuration ==="
echo "Experiment Name: $experiment_name (type: STRING)"
echo "Dataset: $dataset (type: STRING)"
echo "N: $N (type: INT)"
echo "Run Mode: $run_mode (type: STRING)"
echo "Model Path: $model_path (type: STRING)"
echo "Test Budget: $test_budget (type: INT)"
echo "Num Restarts: $num_restarts (type: INT)"
echo "Latent Dim: $latent_dim (type: INT)"
echo "Num Eval Sessions: $num_eval_sessions (type: STRING/INT)"
echo "Session Seed: $session_seed (type: INT)"
echo "Use Synthetic Data: $use_synthetic_data (type: BOOL)"
echo "Random Baseline: $random_baseline (type: BOOL)"
echo "Enable Primer Sequence: $enable_primer_sequence (type: BOOL)"
echo "Primer Sequence Task Repetitions: $primer_sequence_task_repetitions (type: INT)"
echo "MLE Params File: $mle_params_file (type: STRING)"
echo "Synthetic Data File: $synthetic_data_file (type: STRING)"

if [ "$submit_mode" == "bsub" ]; then
    # Run the DALE script on cluster
    # TODO: Update this path to your project directory on the cluster
    cd /storage1/fs1/dbarbour/Active/MarkLu/nmil-mat-ef-dlvm-nn/analysis/dale_syn_data/
    pwd
    echo "Running DALE experiment for $experiment_name"
    source ~/.bashrc 
    # conda init 

    conda env list
    # conda env create -f /storage1/fs1/dbarbour/Active/MarkLu/nmil-mat-ef-dlvm-nn/src/environment.yml
   
    # TODO: Update this to your conda environment name
    conda activate nmil-dlvm-nn
else
    echo "Running locally ??? TODO"
    # TODO: Adjust this path based on your local directory structure
    cd /storage1/fs1/dbarbour/Active/MarkLu/nmil-mat-ef-dlvm-nn/analysis/dale_syn_data/
fi

# Check if synthetic data exists (using the path from CSV)
if [ ! -f "$synthetic_data_file" ]; then
    echo "ERROR: Synthetic data file not found: $synthetic_data_file"
    echo "Please generate synthetic data first using generate_simulation_data.py"
    exit 1
fi

# Check if MLE params file exists
if [ ! -f "$mle_params_file" ]; then
    echo "ERROR: MLE params file not found: $mle_params_file"
    echo "Please ensure MLE params file exists"
    exit 1
fi

# Generate session IDs based on latent dimension
generate_session_ids() {
    local latent_dim=$1
    local all_sessions=""
    
    if [ "$latent_dim" == "1" ]; then
        # Generate LD1-001 to LD1-088
        for i in $(seq -f "%03g" 1 88); do
            all_sessions="$all_sessions LD1-$i"
        done
    elif [ "$latent_dim" == "2" ]; then
        # Generate LD2-001 to LD2-088
        for i in $(seq -f "%03g" 1 88); do
            all_sessions="$all_sessions LD2-$i"
        done
    elif [ "$latent_dim" == "3" ]; then
        # Generate LD3-001 to LD3-088
        for i in $(seq -f "%03g" 1 88); do
            all_sessions="$all_sessions LD3-$i"
        done
    else
        echo "Unknown latent dimension: $latent_dim"
        exit 1
    fi
    
    echo "$all_sessions"
}

# Select evaluation sessions
select_eval_sessions() {
    local num_eval_sessions="$1"
    local all_sessions="$2"
    local eval_sessions=""

    # Accept three forms:
    # 1) "all" => use all sessions
    # 2) integer N => first N sessions
    # 3) explicit list of session IDs (e.g., "LD2-037 LD2-041 LD2-042")
    if [ "$num_eval_sessions" == "all" ]; then
        eval_sessions="$all_sessions"
    elif [[ "$num_eval_sessions" =~ ^[0-9]+$ ]]; then
        eval_sessions=$(echo $all_sessions | cut -d' ' -f1-$num_eval_sessions)
    else
        # treat as explicit session IDs list
        eval_sessions="$num_eval_sessions"
    fi

    echo "$eval_sessions"
}

# Generate appropriate session IDs based on latent dimension
all_sessions=$(generate_session_ids "$latent_dim")
eval_sessions=$(select_eval_sessions "$num_eval_sessions" "$all_sessions")

echo "=== Session Information ==="
echo "Latent Dimension: $latent_dim"
echo "All Sessions: $all_sessions"
echo "Eval Sessions: $eval_sessions"

# Build the Python command
python_cmd="python run_DALE_syn_data.py"
python_cmd="$python_cmd --synthetic_data_file $synthetic_data_file"
python_cmd="$python_cmd --mle_params_file $mle_params_file"
python_cmd="$python_cmd --run_mode $run_mode"
python_cmd="$python_cmd --test_budget $test_budget"
python_cmd="$python_cmd --use_synthetic_data $use_synthetic_data"
python_cmd="$python_cmd --num_restarts $num_restarts"
python_cmd="$python_cmd --latent_dim $latent_dim"
python_cmd="$python_cmd --model_path $model_path"
python_cmd="$python_cmd --primer_sequence_task_repetitions $primer_sequence_task_repetitions"

# Use experiment_name as run_name to set results subfolder
python_cmd="$python_cmd --run_name $experiment_name"

# Add random baseline flag if true
if [ "$random_baseline" == "True" ]; then
    python_cmd="$python_cmd --random_baseline True"
fi

if [ "$enable_primer_sequence" == "True" ]; then
    python_cmd="$python_cmd --enable_primer_sequence True"
fi

# Add eval test session IDs
if [ -n "$eval_sessions" ]; then
    python_cmd="$python_cmd --eval_test_session_ids $eval_sessions"
fi

# TODO: Add held_out_sessions handling if needed
# For now, assuming no held-out sessions as in your original script
# If you need to add held-out sessions, parse them from config and add:
# python_cmd="$python_cmd --trained_model_held_out_ids $held_out_sessions"

echo "=== Running Python Command ==="
echo "$python_cmd"

# Execute the Python command
eval $python_cmd
