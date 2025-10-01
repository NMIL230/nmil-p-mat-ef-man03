import sys
sys.path.append("../")
sys.path.append("../../")
import pandas as pd
import numpy as np
import random as random
from utils.set_seed import set_seed
import argparse
import os

BLACKLIST_SET =[]
DEFAULT_HELDOUT_SET = []
METRICS_DICT = None
RELEVANT_METRICS = None
SUMMARIZED_METRICS = None
ALL_METRICS_MOMENTS_LABEL_DICT = None
ALL_METRICS = None
SUMMARIZED_METRICS_MAIN_MOMENTS_LABEL_DICT = None
SUMMARIZED_METRICS_METRIC_TYPES = None

def create_config_file(DATASET):
    data_csv = pd.read_csv(f"../data/{DATASET}/all_data_time_not_normed.csv")
    data_csv = data_csv.rename(columns = {"task_label":"summary_metric_label"})

    ALL_PARTICIPANT_IDS = list(data_csv["user_session"].unique())
    sample_size = int(np.floor(0.1 * len(ALL_PARTICIPANT_IDS)))
    random_sample = random.sample(ALL_PARTICIPANT_IDS, sample_size)
    DEFAULT_HELDOUT_SET =random_sample # use the random as the default HELDOUT_SET
    BLACKLIST_SET = []

    LOCAL_METRICS_DICT = {  
    }

    data_csv.groupby(["metric","summary_metric_label","data_type"])
    # Group by 'metric' and extract unique 'data_type' and the common 'summary_metric_label' for each 'metric'
    grouped = data_csv.groupby('metric')

    for metric, group in grouped:
        unique_data_types = group['data_type'].unique().tolist()
        unique_metric_label = group['summary_metric_label'].unique().tolist()
        if len(unique_data_types) > 1:
            raise ValueError(f"Multiple data types found for metric {metric}: {unique_data_types}")
            
        if len(unique_metric_label) > 1:
            raise ValueError(f"Multiple data types found for metric {metric}: {unique_metric_label}")
            
        summary_metric_label = unique_metric_label[0] 
        
        data_type = unique_data_types[0]
        

        LOCAL_METRICS_DICT[metric] = {
            'type': data_type,
            'summary_metric_label': summary_metric_label if data_type =="binarySpan" else metric,
            'vis_label': summary_metric_label
        }

    RELEVANT_METRICS =list(LOCAL_METRICS_DICT.keys()) # modify this if you wany to ignore some tasks and not just take the defaults
    ALL_METRICS = RELEVANT_METRICS

    # SUMMARIZED_METRICS =["RunningSpan","CorsiBlock","LetterNumberSpan_is_correct", "NearTransfer","Countermanding_reaction_time",
    #                      "MatrixReasoning_is_correct","NBack","D2Neo_hit_accuracy"]
    SUMMARIZED_METRICS = set()
    SUMMARIZED_METRICS_MAIN_MOMENTS_LABEL_DICT = {}
    SUMMARIZED_METRICS_METRIC_TYPES = {}
    for metric in RELEVANT_METRICS:
        if LOCAL_METRICS_DICT[metric]["type"] =="binarySpan":
            SUMMARIZED_METRICS.add(LOCAL_METRICS_DICT[metric]["summary_metric_label"])
            SUMMARIZED_METRICS_MAIN_MOMENTS_LABEL_DICT[LOCAL_METRICS_DICT[metric]["summary_metric_label"]] = LOCAL_METRICS_DICT[metric]["summary_metric_label"]
            SUMMARIZED_METRICS_METRIC_TYPES[LOCAL_METRICS_DICT[metric]["summary_metric_label"]] = LOCAL_METRICS_DICT[metric]["type"]
        else:
            SUMMARIZED_METRICS.add(metric)
            SUMMARIZED_METRICS_MAIN_MOMENTS_LABEL_DICT[metric] = LOCAL_METRICS_DICT[metric]["vis_label"]
            SUMMARIZED_METRICS_METRIC_TYPES[metric] = LOCAL_METRICS_DICT[metric]["type"]

    SUMMARIZED_METRICS = list(SUMMARIZED_METRICS)

    ALL_METRICS_MOMENTS_LABEL_DICT = {}

    for metric in SUMMARIZED_METRICS_METRIC_TYPES.keys():
        metric_type = SUMMARIZED_METRICS_METRIC_TYPES[metric]
        if metric_type == "binary":
            ALL_METRICS_MOMENTS_LABEL_DICT[f"{metric}_param1"] = SUMMARIZED_METRICS_MAIN_MOMENTS_LABEL_DICT[metric]
        elif metric_type == "binarySpan":
            ALL_METRICS_MOMENTS_LABEL_DICT[f"{metric}_param1"] = f"{metric} psiTheta" 
            ALL_METRICS_MOMENTS_LABEL_DICT[f"{metric}_param2"] = f"{metric} psiSigma" 
        elif metric_type == "timing":
            ALL_METRICS_MOMENTS_LABEL_DICT[f"{metric}_param1"] = f"{SUMMARIZED_METRICS_MAIN_MOMENTS_LABEL_DICT[metric]} mean" 
            ALL_METRICS_MOMENTS_LABEL_DICT[f"{metric}_param2"] = f"{SUMMARIZED_METRICS_MAIN_MOMENTS_LABEL_DICT[metric]} std" 
        else:
            print ("Unknown metric type : ",metric_type)
    
    variables_dict = {
        "BLACKLIST_SET": BLACKLIST_SET,
        "DEFAULT_HELDOUT_SET": DEFAULT_HELDOUT_SET,
        "METRICS_DICT": LOCAL_METRICS_DICT,
        "RELEVANT_METRICS": RELEVANT_METRICS,
        "SUMMARIZED_METRICS": SUMMARIZED_METRICS,
        "ALL_METRICS_MOMENTS_LABEL_DICT": ALL_METRICS_MOMENTS_LABEL_DICT,
        "ALL_METRICS": ALL_METRICS,
        "SUMMARIZED_METRICS_MAIN_MOMENTS_LABEL_DICT": SUMMARIZED_METRICS_MAIN_MOMENTS_LABEL_DICT,
        "SUMMARIZED_METRICS_METRIC_TYPES": SUMMARIZED_METRICS_METRIC_TYPES
    }
    write_py_file(variables_dict,DATASET)
        
def write_py_file(variables_dict,filename):
    
    BLACKLIST_SET = variables_dict.get("BLACKLIST_SET")
    DEFAULT_HELDOUT_SET = variables_dict.get("DEFAULT_HELDOUT_SET")
    METRICS_DICT = variables_dict.get("METRICS_DICT")
    RELEVANT_METRICS = variables_dict.get("RELEVANT_METRICS")
    SUMMARIZED_METRICS = variables_dict.get("SUMMARIZED_METRICS")
    ALL_METRICS_MOMENTS_LABEL_DICT = variables_dict.get("ALL_METRICS_MOMENTS_LABEL_DICT")
    ALL_METRICS = variables_dict.get("ALL_METRICS")
    SUMMARIZED_METRICS_MAIN_MOMENTS_LABEL_DICT = variables_dict.get("SUMMARIZED_METRICS_MAIN_MOMENTS_LABEL_DICT")
    SUMMARIZED_METRICS_METRIC_TYPES = variables_dict.get("SUMMARIZED_METRICS_METRIC_TYPES")
    
    # Path for the new Python file
    output_file_path = f'config_files/{filename}.py'

    # Writing the variables and their data to the new Python file
    with open(output_file_path, 'w') as file:
        file.write("# Auto-generated file containing variables and their data\n")
        
        file.write("# EDIT THIS IF YOU NEED TO CHANGE HOW THE DATA SHOULD BE MODELED OR VISUALIZED\n\n")

        file.write("BLACKLIST_SET = ")
        file.write(repr(BLACKLIST_SET))
        file.write("\n\n")

        file.write("# Randomly selected test set - 10% of the IDs in the .csv file\n")
        file.write("DEFAULT_HELDOUT_SET = ")
        file.write(repr(DEFAULT_HELDOUT_SET))
        file.write("\n\n")

        file.write("# Define all the metric variables to be modeled with their respective assumed distributions (types)\n")
        file.write("METRICS_DICT = ")
        file.write(repr(METRICS_DICT))
        file.write("\n\n")
        
        file.write("# All metrics including those ignored in RELEVANT_METRICS\n")
        file.write("ALL_METRICS = ")
        file.write(repr(ALL_METRICS))
        file.write("\n\n")

        file.write("# Subset of the metrics of interest\n")
        file.write("RELEVANT_METRICS = ")
        file.write(repr(RELEVANT_METRICS))
        file.write("\n\n")

        file.write("# Subset of the metrics of interest\n")
        file.write("SUMMARIZED_METRICS = ")
        file.write(repr(SUMMARIZED_METRICS))
        file.write("\n\n")
        
        file.write("# Labels for visualization purposes\n")
        file.write("ALL_METRICS_MOMENTS_LABEL_DICT = ")
        file.write(repr(ALL_METRICS_MOMENTS_LABEL_DICT))
        file.write("\n\n")
        
        file.write("# Reverse mapping of the summarized metrics to types\n")
        file.write("SUMMARIZED_METRICS_METRIC_TYPES = ")
        file.write(repr(SUMMARIZED_METRICS_METRIC_TYPES))
        file.write("\n\n")

        file.write("SUMMARIZED_METRICS_MAIN_MOMENTS_LABEL_DICT = ")
        file.write(repr(SUMMARIZED_METRICS_MAIN_MOMENTS_LABEL_DICT))
        file.write("\n\n")
        
        # Ensure all data is written to disk
        file.flush()
        os.fsync(file.fileno())

    print(f"Config Variables have been written to {output_file_path}.")
    

def main():
    parser = argparse.ArgumentParser(description="Create a configuration file for a dataset.")
    parser.add_argument('--dataset', required=True, help='The name of the dataset')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite the existing configuration file if it exists')
    
    args = parser.parse_args()
    
    dataset_file = f"{args.dataset}.py"
    
    if not os.path.exists(dataset_file) or args.overwrite:
        create_config_file(args.dataset)
    else:
        print(f"{dataset_file} already exists. Use --overwrite to overwrite the file.")

if __name__ == "__main__":
    main()