#!/bin/bash
# Script 1: Generate DALE experiment configurations
# This script creates a CSV file with all experiment configurations

# Define paths as variables for easy reuse
# TODO: Update these paths according to your setup
REPO_ROOT="../.."
SAVED_MODELS_DIR="$REPO_ROOT/saved_models"
DATA_DIR="$REPO_ROOT/data"
DATASET_NAME="COLL10_SIM"
SYNTHETIC_DATA_DIR="$REPO_ROOT/analysis/dlvm_imle_comparison/synthetic_data"

# Model paths
MODEL_3D_PATH="$SAVED_MODELS_DIR/$DATASET_NAME/heldout_obsmulti/variationalNN_relevant_only_latentdim3_beaver-slide-5310.pt"
MODEL_2D_PATH="$SAVED_MODELS_DIR/$DATASET_NAME/heldout_obsmulti/variationalNN_relevant_only_latentdim2_mongoose-dive-7464.pt"
MODEL_1D_PATH="$SAVED_MODELS_DIR/$DATASET_NAME/heldout_obsmulti/variationalNN_relevant_only_latentdim1_wolverine-zoom-7298.pt"

# MLE params file paths
MLE_PARAMS_3D_PATH="$DATA_DIR/$DATASET_NAME/D3_synthetic_ground_truth_parameters_beaver-slide-5310.pt"
MLE_PARAMS_2D_PATH="$DATA_DIR/$DATASET_NAME/D2_synthetic_ground_truth_parameters_mongoose-dive-7464.pt"
MLE_PARAMS_1D_PATH="$DATA_DIR/$DATASET_NAME/D1_synthetic_ground_truth_parameters_wolverine-zoom-7298.pt"

# Synthetic data file paths
SYNTHETIC_DATA_2D_PATH="$SYNTHETIC_DATA_DIR/$DATASET_NAME/D2_all_synthetic_data_N240.pt"
SYNTHETIC_DATA_3D_PATH="$SYNTHETIC_DATA_DIR/$DATASET_NAME/D3_all_synthetic_data_N240.pt"
SYNTHETIC_DATA_1D_PATH="$SYNTHETIC_DATA_DIR/$DATASET_NAME/D1_all_synthetic_data_N240.pt"

# Function to generate session IDs based on latent dimension
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

# Function to select evaluation sessions
select_eval_sessions() {
    local num_eval_sessions=$1
    local all_sessions=$2
    local eval_sessions=""
    
    if [ "$num_eval_sessions" == "all" ]; then
        eval_sessions="$all_sessions"
    else
        # Take the first N sessions
        eval_sessions=$(echo $all_sessions | cut -d' ' -f1-$num_eval_sessions)
    fi
    
    echo "$eval_sessions"
}

# Define experiment parameters as arrays (final configured set)
# 保留 c 组定义，但暂不启用；改为启用 d 组 1D 实验
experiment_names=(
    # emergency re-run groups: only incomplete/missing sessions
    "exp_e1_1d_dale_ps0"
    "exp_e2_1d_dale_ps2"
    "exp_e3_1d_dale_ps4"
    "exp_e4_1d_random"
)
datasets=("COLL10_SIM")
N_values=(240)
run_modes=("run") # "run" or "debug"
test_budgets=(240)
# num_restarts per-experiment is overridden in case blocks
num_restarts=(10)
num_eval_sessions=(all) # Can be numbers or "all"
session_seeds=(42)
use_synthetic_data_values=("True")
primer_sequence_task_repetitions=(4) # This will be overridden by experiment-specific values

# Job configuration
job_name="dale_experiments"

# Create config directory
mkdir -p train_configs
config_csv_file="train_configs/$job_name.csv"

# Remove the config file if it exists
if [ -f "$config_csv_file" ]; then
    rm $config_csv_file
fi

# Write CSV header
echo "experiment_name,dataset,N,run_mode,model_path,test_budget,num_restarts,latent_dim,num_eval_sessions,session_seed,use_synthetic_data,random_baseline,enable_primer_sequence,primer_sequence_task_repetitions,mle_params_file,synthetic_data_file" > $config_csv_file

counter=0
# Generate all combinations
# TODO: Adjust the nested loops based on which parameters you want to vary
for experiment_name in ${experiment_names[@]}; do
    # Set experiment-specific parameters based on experiment name
    case $experiment_name in
        # --- Emergency groups (custom session sets) ---
        "exp_e1_1d_dale_ps0")
            # Incomplete/missing for exp_d1_1d_dale_ps0
            latent_dims=(1)
            model_paths=("$MODEL_1D_PATH")
            mle_params_files=("$MLE_PARAMS_1D_PATH")
            synthetic_data_files=("$SYNTHETIC_DATA_1D_PATH")
            random_baseline_values=("False")
            enable_primer_sequence_values=("False")
            primer_repetitions_values=(4)
            # 22 sessions: broken {067,078} + missing {068..077,079..088}
            CUSTOM_SESSIONS="LD1-067 LD1-068 LD1-069 LD1-070 LD1-071 LD1-072 LD1-073 LD1-074 LD1-075 LD1-076 LD1-077 LD1-078 LD1-079 LD1-080 LD1-081 LD1-082 LD1-083 LD1-084 LD1-085 LD1-086 LD1-087 LD1-088"
            split_count=4
            num_restarts=(3)
            ;;
        "exp_e2_1d_dale_ps2")
            # Incomplete/missing for exp_d2_1d_dale_ps2
            latent_dims=(1)
            model_paths=("$MODEL_1D_PATH")
            mle_params_files=("$MLE_PARAMS_1D_PATH")
            synthetic_data_files=("$SYNTHETIC_DATA_1D_PATH")
            random_baseline_values=("False")
            enable_primer_sequence_values=("True")
            primer_repetitions_values=(2)
            # 17 sessions: broken {069,081} + missing {070,071,072,073,074,075,076,077,082..088}
            CUSTOM_SESSIONS="LD1-069 LD1-070 LD1-071 LD1-072 LD1-073 LD1-074 LD1-075 LD1-076 LD1-077 LD1-081 LD1-082 LD1-083 LD1-084 LD1-085 LD1-086 LD1-087 LD1-088"
            split_count=4
            num_restarts=(3)
            ;;
        "exp_e3_1d_dale_ps4")
            # Incomplete/missing for exp_d3_1d_dale_ps4
            latent_dims=(1)
            model_paths=("$MODEL_1D_PATH")
            mle_params_files=("$MLE_PARAMS_1D_PATH")
            synthetic_data_files=("$SYNTHETIC_DATA_1D_PATH")
            random_baseline_values=("False")
            enable_primer_sequence_values=("True")
            primer_repetitions_values=(4)
            # 65 sessions: broken {004,015,025,037,047,058,071,082} + missing list from report
            CUSTOM_SESSIONS="LD1-004 LD1-005 LD1-006 LD1-007 LD1-008 LD1-009 LD1-010 LD1-011 LD1-015 LD1-016 LD1-017 LD1-018 LD1-019 LD1-020 LD1-021 LD1-022 LD1-025 LD1-026 LD1-027 LD1-028 LD1-029 LD1-030 LD1-031 LD1-032 LD1-033 LD1-037 LD1-038 LD1-039 LD1-040 LD1-041 LD1-042 LD1-043 LD1-044 LD1-047 LD1-048 LD1-049 LD1-050 LD1-051 LD1-052 LD1-053 LD1-054 LD1-055 LD1-058 LD1-059 LD1-060 LD1-061 LD1-062 LD1-063 LD1-064 LD1-065 LD1-066 LD1-071 LD1-072 LD1-073 LD1-074 LD1-075 LD1-076 LD1-077 LD1-082 LD1-083 LD1-084 LD1-085 LD1-086 LD1-087 LD1-088"
            split_count=8
            num_restarts=(3)
            ;;
        "exp_e4_1d_random")
            # Incomplete/missing for exp_d4_1d_random
            latent_dims=(1)
            model_paths=("$MODEL_1D_PATH")
            mle_params_files=("$MLE_PARAMS_1D_PATH")
            synthetic_data_files=("$SYNTHETIC_DATA_1D_PATH")
            random_baseline_values=("True")
            enable_primer_sequence_values=("False")
            primer_repetitions_values=(4)
            # 48 sessions: broken {004,015,026,036,053,064,069} + missing list from report
            CUSTOM_SESSIONS="LD1-004 LD1-005 LD1-006 LD1-007 LD1-008 LD1-009 LD1-010 LD1-011 LD1-015 LD1-016 LD1-017 LD1-018 LD1-019 LD1-020 LD1-021 LD1-022 LD1-026 LD1-027 LD1-028 LD1-029 LD1-030 LD1-031 LD1-032 LD1-033 LD1-036 LD1-037 LD1-038 LD1-039 LD1-040 LD1-041 LD1-042 LD1-043 LD1-044 LD1-053 LD1-054 LD1-055 LD1-064 LD1-065 LD1-066 LD1-069 LD1-070 LD1-071 LD1-072 LD1-073 LD1-074 LD1-075 LD1-076 LD1-077"
            split_count=8
            num_restarts=(3)
            ;;
        "exp_d1_1d_dale_ps0")
            latent_dims=(1)
            model_paths=("$MODEL_1D_PATH")
            mle_params_files=("$MLE_PARAMS_1D_PATH")
            synthetic_data_files=("$SYNTHETIC_DATA_1D_PATH")
            random_baseline_values=("False")
            enable_primer_sequence_values=("False")
            primer_repetitions_values=(4) # not used when primer disabled
            CUSTOM_SESSIONS=""
            split_count=8
            num_restarts=(3)
            ;;
        "exp_d2_1d_dale_ps2")
            latent_dims=(1)
            model_paths=("$MODEL_1D_PATH")
            mle_params_files=("$MLE_PARAMS_1D_PATH")
            synthetic_data_files=("$SYNTHETIC_DATA_1D_PATH")
            random_baseline_values=("False")
            enable_primer_sequence_values=("True")
            primer_repetitions_values=(2)
            CUSTOM_SESSIONS=""
            split_count=8
            num_restarts=(3)
            ;;
        "exp_d3_1d_dale_ps4")
            latent_dims=(1)
            model_paths=("$MODEL_1D_PATH")
            mle_params_files=("$MLE_PARAMS_1D_PATH")
            synthetic_data_files=("$SYNTHETIC_DATA_1D_PATH")
            random_baseline_values=("False")
            enable_primer_sequence_values=("True")
            primer_repetitions_values=(4)
            CUSTOM_SESSIONS=""
            split_count=8
            num_restarts=(3)
            ;;
        "exp_d4_1d_random")
            latent_dims=(1)
            model_paths=("$MODEL_1D_PATH")
            mle_params_files=("$MLE_PARAMS_1D_PATH")
            synthetic_data_files=("$SYNTHETIC_DATA_1D_PATH")
            random_baseline_values=("True")
            enable_primer_sequence_values=("False")
            primer_repetitions_values=(4) # not used when primer disabled
            CUSTOM_SESSIONS=""
            split_count=8
            num_restarts=(3)
            ;;

        "exp_c1_3d_dale_ps0")
            latent_dims=(3)
            model_paths=("$MODEL_3D_PATH")
            mle_params_files=("$MLE_PARAMS_3D_PATH")
            synthetic_data_files=("$SYNTHETIC_DATA_3D_PATH")
            random_baseline_values=("False")
            enable_primer_sequence_values=("False")
            primer_repetitions_values=(4) # not used when primer disabled
            split_count=8
            num_restarts=(100)
            ;;
        "exp_c2_3d_dale_ps2")
            latent_dims=(3)
            model_paths=("$MODEL_3D_PATH")
            mle_params_files=("$MLE_PARAMS_3D_PATH")
            synthetic_data_files=("$SYNTHETIC_DATA_3D_PATH")
            random_baseline_values=("False")
            enable_primer_sequence_values=("True")
            primer_repetitions_values=(2)
            split_count=8
            num_restarts=(100)
            ;;
        "exp_c3_3d_dale_ps4")
            latent_dims=(3)
            model_paths=("$MODEL_3D_PATH")
            mle_params_files=("$MLE_PARAMS_3D_PATH")
            synthetic_data_files=("$SYNTHETIC_DATA_3D_PATH")
            random_baseline_values=("False")
            enable_primer_sequence_values=("True")
            primer_repetitions_values=(4)
            split_count=8
            num_restarts=(100)
            ;;
        "exp_c4_3d_random")
            latent_dims=(3)
            model_paths=("$MODEL_3D_PATH")
            mle_params_files=("$MLE_PARAMS_3D_PATH")
            synthetic_data_files=("$SYNTHETIC_DATA_3D_PATH")
            random_baseline_values=("True")
            enable_primer_sequence_values=("False")
            primer_repetitions_values=(4) # not used when primer disabled
            split_count=8
            num_restarts=(100)
            ;;
        "exp_c5_2d_dale_ps0")
            latent_dims=(2)
            model_paths=("$MODEL_2D_PATH")
            mle_params_files=("$MLE_PARAMS_2D_PATH")
            synthetic_data_files=("$SYNTHETIC_DATA_2D_PATH")
            random_baseline_values=("False")
            enable_primer_sequence_values=("False")
            primer_repetitions_values=(4) # not used when primer disabled
            split_count=8
            num_restarts=(10)
            ;;
        "exp_c6_2d_dale_ps2")
            latent_dims=(2)
            model_paths=("$MODEL_2D_PATH")
            mle_params_files=("$MLE_PARAMS_2D_PATH")
            synthetic_data_files=("$SYNTHETIC_DATA_2D_PATH")
            random_baseline_values=("False")
            enable_primer_sequence_values=("True")
            primer_repetitions_values=(2)
            split_count=8
            num_restarts=(10)
            ;;
        "exp_c7_2d_dale_ps4")
            latent_dims=(2)
            model_paths=("$MODEL_2D_PATH")
            mle_params_files=("$MLE_PARAMS_2D_PATH")
            synthetic_data_files=("$SYNTHETIC_DATA_2D_PATH")
            random_baseline_values=("False")
            enable_primer_sequence_values=("True")
            primer_repetitions_values=(4)
            split_count=8
            num_restarts=(10)
            ;;
        "exp_c8_2d_random")
            latent_dims=(2)
            model_paths=("$MODEL_2D_PATH")
            mle_params_files=("$MLE_PARAMS_2D_PATH")
            synthetic_data_files=("$SYNTHETIC_DATA_2D_PATH")
            random_baseline_values=("True")
            enable_primer_sequence_values=("False")
            primer_repetitions_values=(4) # not used when primer disabled
            split_count=8
            num_restarts=(10)
            ;;
        *)
            echo "Unknown experiment name: $experiment_name"
            exit 1
            ;;
    esac

    for dataset in ${datasets[@]}; do
        for N in ${N_values[@]}; do
            for run_mode in ${run_modes[@]}; do
                for model_path in ${model_paths[@]}; do
                    for mle_params_file in ${mle_params_files[@]}; do
                        for synthetic_data_file in ${synthetic_data_files[@]}; do
                            for test_budget in ${test_budgets[@]}; do
                                for num_restart in ${num_restarts[@]}; do
                                    for latent_dim in ${latent_dims[@]}; do
                                        for num_eval_session in ${num_eval_sessions[@]}; do
                                            for session_seed in ${session_seeds[@]}; do
                                                for use_synthetic_data in ${use_synthetic_data_values[@]}; do
                                                    for random_baseline in ${random_baseline_values[@]}; do
                                                        for enable_primer_sequence in ${enable_primer_sequence_values[@]}; do
                                                            for primer_repetitions in ${primer_repetitions_values[@]}; do
                                                                # 生成session列表：如指定了CUSTOM_SESSIONS则用之，否则生成全部88个
                                                                if [ -n "$CUSTOM_SESSIONS" ]; then
                                                                    all_sessions="$CUSTOM_SESSIONS"
                                                                else
                                                                    all_sessions=$(generate_session_ids "$latent_dim")
                                                                fi

                                                                if [ -z "$split_count" ]; then split_count=1; fi
                                                                if [ "$split_count" -gt 1 ]; then
                                                                    # 将session按split_count等分（支持自定义数量，含奇数）
                                                                    if [ -n "$CUSTOM_SESSIONS" ]; then
                                                                        total=$(wc -w <<< "$CUSTOM_SESSIONS")
                                                                    else
                                                                        total=88
                                                                    fi
                                                                    base=$(( total / split_count ))
                                                                    rem=$(( total % split_count ))
                                                                    start=1
                                                                    for (( part=1; part<=split_count; part++ )); do
                                                                        size=$base
                                                                        if [ $part -le $rem ]; then
                                                                            size=$((base + 1))
                                                                        fi
                                                                        end=$(( start + size - 1 ))
                                                                        eval_sessions=$(echo $all_sessions | cut -d' ' -f${start}-${end})
                                                                        # 将本子任务写入CSV，run_name保持相同（experiment_name相同）
                                                                        echo "$experiment_name,$dataset,$N,$run_mode,$model_path,$test_budget,$num_restart,$latent_dim,$eval_sessions,$session_seed,$use_synthetic_data,$random_baseline,$enable_primer_sequence,$primer_repetitions,$mle_params_file,$synthetic_data_file" >> $config_csv_file
                                                                        counter=$((counter+1))
                                                                        start=$(( end + 1 ))
                                                                    done
                                                                else
                                                                    # 不拆分：按既有策略选择N个或全部
                                                                    eval_sessions=$(select_eval_sessions "$num_eval_session" "$all_sessions")
                                                                    echo "$experiment_name,$dataset,$N,$run_mode,$model_path,$test_budget,$num_restart,$latent_dim,$eval_sessions,$session_seed,$use_synthetic_data,$random_baseline,$enable_primer_sequence,$primer_repetitions,$mle_params_file,$synthetic_data_file" >> $config_csv_file
                                                                    counter=$((counter+1))
                                                                fi
                                                            done
                                                        done
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "Total number of configurations: $counter"
total_jobs=$counter

# Get submit mode from command line argument, default to bsub if not provided
submit_mode="${1:-bsub}"  # Use bsub if no argument provided

batch_size=1000
if [ "$submit_mode" == "bsub" ]; then
    # TODO: Update this log directory path to your cluster's log directory
    LOG_DIR="/storage1/fs1/dbarbour/Active/MarkLu/nmil-mat-ef-dlvm-nn/analysis/dale_syn_data/shell_scripts/logs"
    mkdir -p $LOG_DIR
fi

# Submit job arrays in batches
for ((start=0; start<total_jobs; start+=batch_size)); do
    end=$((start + batch_size - 1))
    if [ $end -ge $total_jobs ]; then
        end=$((total_jobs - 1))
    fi
    
    # Create the job array for this batch
    job_array="$job_name[$((start+1))-$((end+1))]"
    echo "Submitting job array: $job_array"

    if [ "$submit_mode" == "bsub" ]; then
        # TODO: Update these cluster-specific settings
        blacklisted_hosts=("compute1-exec-1ba")
        select_clause="gpuhost"
        for host in $blacklisted_hosts; do
            select_clause="${select_clause} && hname!='${host}'"
        done
        echo "Select clause: $select_clause"

        # Submit the job array
        export LSF_DOCKER_RUN_LOGLEVEL=DEBUG
        export PATH=/opt/conda/bin:$PATH
        # TODO: Update these volume paths to match your cluster setup
        export LSF_DOCKER_VOLUMES="/home/lu.zeyu:/home/lu.zeyu /scratch1/fs1/dbarbour:/scratch1/fs1/dbarbour /storage1/fs1/dbarbour/Active:/storage1/fs1/dbarbour/Active" 
        export LSF_DOCKER_SHM_SIZE=16g 
        
        bsub -n 1 \
            -J "$job_array" \
            -G compute-dbarbour \
            -q general \
            -M 32GB \
            -m "general" \
            -u mark.lu@wustl.edu \
            -o "${LOG_DIR}/dale_exp.%J.%I.txt" \
            -e "${LOG_DIR}/dale_exp.%J.%I.err" \
            -R "select[$select_clause] rusage[mem=32GB] span[hosts=1]" \
            -g /lu.zeyu/limit100 \
            -gpu 'num=1:j_exclusive=no' \
            -a 'docker(rapidsai/rapidsai:21.10-cuda11.0-runtime-ubuntu20.04-py3.8)' \
            "/storage1/fs1/dbarbour/Active/MarkLu/nmil-mat-ef-dlvm-nn/analysis/dale_syn_data/shell_scripts/run_dale_job.sh" "$job_name.\$LSB_JOBINDEX" "$submit_mode"

        echo "Batch from job $start to $end has been submitted as a job array."
    else
        # Run locally
        for ((i=start+1; i<=end+1; i++)); do
            echo "Running job $i locally"
            ./run_dale_job.sh "$job_name.$i" "local"
        done
    fi
done
