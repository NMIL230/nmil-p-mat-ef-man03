
import os
import sys
from pathlib import Path

from matplotlib import pyplot as plt
# add absolute path to utils folder
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from nmil_dlvm.paths import ANALYSIS_ARTIFACTS_ROOT, data_dir, ensure_dir
from nmil_dlvm.utils.active_learning_utils import load_trained_model
from nmil_dlvm.utils.data_distribution_utils import ALL_METRICS_MOMENTS_LABEL_DICT, CURR_METRICS_DICT, DATASET
from nmil_dlvm.utils.grid_search_utils import compute_predictions_fom_latent_points, generate_grid
from semopy import Model, Optimizer, stats
from semopy.inspector import inspect

import numpy as np
import torch
import pandas as pd

import argparse
import pandas as pd

from nmil_dlvm.utils.mle_utils import extract_mle_parameters
import logging
from scipy.stats import shapiro
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer

SEM_ANALYSIS_ROOT = ensure_dir(ANALYSIS_ARTIFACTS_ROOT / "sem_analysis")
LOGS_DIR = ensure_dir(SEM_ANALYSIS_ROOT / "logs")
GENERATED_DATA_DIR = ensure_dir(SEM_ANALYSIS_ROOT / "generated_data")
PLOTS_DIR = ensure_dir(SEM_ANALYSIS_ROOT / "plots")

# Ensure the logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)


logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.FileHandler(LOGS_DIR / "sem_model_comparison.log")
handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)

logger.info("Logging initialized with file handler")

def load_mle_params(mle_params_file_path):
    """Helper function to load MLE parameters."""
    return extract_mle_parameters(mle_params_file_path)

def load_data(csv_file):
    """Helper function to load data from CSV."""
    df = pd.read_csv(csv_file)
    return df

def build_and_log_sem_results(df, sem_model):
    """Helper function to build the SEM model and log the results."""
    results, fit_measures,latent_values= build_sem_model(df, sem_model)
   
    return results, fit_measures,latent_values

def log_sem_results(model_name, results, fit_measures):
    """Helper function to log SEM model results."""
    logging.info(f"{model_name} RESULTS")
    logging.info(f"Results:\n{results}")
    logging.info(f"Fit Measures:\n{fit_measures}")

def compare_fits(fit_measures_1, fit_measures_2):
    """Helper function to compare fit measures between two models."""
    logging.info("COMPARISON OF FITS:")
    # for metric in fit_measures_1.keys():
    #     fit_1 = fit_measures_1.get(metric, 'N/A')
    #     fit_2 = fit_measures_2.get(metric, 'N/A')
    #     logging.info(f"{metric}: Generative Model = {fit_1}, Raw MLE Model = {fit_2}")



def run_exploratory_factor_analysis(df, n_factors):
    """
    Perform Exploratory Factor Analysis (EFA) on the given dataframe.

    Parameters:
    df (pd.DataFrame): The dataframe containing the data for EFA.
    n_factors (int): The number of factors to extract.

    Returns:
    pd.DataFrame: Factor loadings of the variables on the extracted factors.
    """
    # drop non-numeric columns and rows with missing values
    df = df.select_dtypes(include=[np.number]).dropna()

    # Step 1: Standardize the data
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df)
    
    # Step 2: Initialize and fit the FactorAnalyzer
    fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
    fa.fit(scaled_df)
    
    # Step 3: Get the factor loadings
    loadings = pd.DataFrame(fa.loadings_, index=df.columns)
    
    # Optionally, you can print the eigenvalues to understand the variance explained
    eigenvalues, _ = fa.get_eigenvalues()
    print("Eigenvalues:", eigenvalues)
    
    return loadings

def run_and_plot_pca(df, file_name, title="PCA Plot"):
    """
    Runs PCA on the data and plots the explained variance ratio.

    Args:
    - df: Pandas DataFrame containing the data.
    - file_name: String representing the name of the file to save the plot.
    - title: Title of the plot.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # Select only numeric columns
    df_numeric = df.select_dtypes(include=[np.number])

    # drop rows with missing values
    df_numeric = df_numeric.dropna()


    # Perform PCA
    pca = PCA()
    pca.fit(df_numeric)

    # Plot the explained variance ratio for the first 6 components
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, 7), pca.explained_variance_ratio_[:6], alpha=0.7, align='center')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title(title)
    plt.xticks(range(1, 7))
    plt.grid()
    plt.savefig(file_name)
    plt.close()

def plot_correlation_matrix_with_values(df, file_name, title="Correlation Matrix with Values"):
    """
    Plots the correlation matrix with correlation values in the respective cells and saves the plot.

    Args:
    - df: Pandas DataFrame containing the data.
    - file_name: String representing the name of the file to save the plot.
    - title: Title of the plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Select only numeric columns
    df_numeric = df.select_dtypes(include=[np.number])

    # Compute the correlation matrix
    corr = df_numeric.corr()

    # Change the column names to shorter names (assuming you have ALL_METRICS_MOMENTS_LABEL_DICT defined)
    col_names = [ALL_METRICS_MOMENTS_LABEL_DICT.get(col, col) for col in df_numeric.columns]
    corr.columns = col_names
    corr.index = col_names

    # Sort the correlation matrix by column names in ascending order
    corr = corr.reindex(sorted(corr.columns), axis=0)
    corr = corr.reindex(sorted(corr.columns), axis=1)

    # Ensure x-labels are at the bottom
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.top'] = False
    plt.rcParams['xtick.labeltop'] = False

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    cax = ax.matshow(corr, cmap='coolwarm')

    # Set axis labels with shorter names
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)

    # Annotate each cell with the correlation value
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', color='black', fontsize=6)

    # Add a color bar
    fig.colorbar(cax)

    # Adjust layout to avoid overlap of x-axis labels and plot
    plt.subplots_adjust(left=0.2, right=0.8, top=0.85, bottom=0.2)

    # Add a title above the plot
    plt.suptitle(title, fontsize=14)

    # Save the plot
    plt.savefig(file_name, bbox_inches='tight')  # Use bbox_inches='tight' to ensure nothing is cut off

    # Close the plot
    plt.close()


def main(args):
    # Initialize the csv file path
    model_id = args.model_id
    latent_dim = args.latent_dim
    csv_file = GENERATED_DATA_DIR / f"ld{latent_dim}_{model_id}_latent_space_preds.csv"
    mle_params_file_path = data_dir(DATASET) / "all_data-best_mle_params_mpf100.pt"

    logging.info("Model ID: ", model_id)
    logging.info("Latent Dimension: ", latent_dim)

    # Load the generative data
    df_generative = load_data(csv_file)
    
    # Load the MLE parameters from raw data
    mle_params_first_moments, mle_params_all_moments_df = load_mle_params(mle_params_file_path)

    # variables to drop due to multicollinearity
    # - SimpleSpan_param2, SimpleSpan_param1 correlate with CorsiComplex_param2, CorsiComplex_param1
    # - RunningSpan_correct_w_len_2_param1,RunningSpan_correct_w_len_3_param1 correlate with CorsiComplex_param2, CorsiComplex_param1

    drop_list = ["RunningSpan_correct_w_len_2_param1","RunningSpan_correct_w_len_3_param1", "SimpleSpan_param2", "SimpleSpan_param1"]

    # accuracy variables to logit transform
    accuracy_vars = ["D2_hit_accuracy_param1", "PasatPlus_correctly_answered_param1", "RunningSpan_correct_w_len_2_param1", "RunningSpan_correct_w_len_3_param1"]

    # Call the function to plot and save the correlation matrix
    # make directory to save the plots
    dataset_plots_dir = ensure_dir(PLOTS_DIR / DATASET)
    plot_correlation_matrix_with_values(df_generative, dataset_plots_dir / f"{model_id}_correlation_matrix.png", title=f"Parameters Latent Space Generative Model ({model_id}, Training dataset = {DATASET})")

    # mle_params_all_moments_df = mle_params_all_moments_df.drop(columns=drop_list)
    plot_correlation_matrix_with_values(mle_params_all_moments_df, dataset_plots_dir / "MLE_correlation_matrix.png", title = f"MLE Parameters from Raw Data (Dataset = {DATASET})")

    # Drop columns suffixed by "_param2" for PCA and EFA
    df_generative_pca_efa = df_generative.loc[:, ~df_generative.columns.str.endswith('_param2')]
    mle_params_all_moments_df_pca_efa = mle_params_all_moments_df.loc[:, ~mle_params_all_moments_df.columns.str.endswith('_param2')]

    run_and_plot_pca(df_generative_pca_efa, dataset_plots_dir / f"{model_id}_pca_plot.png", title=f"PCA Plot for Generative Model ({model_id}, Training dataset = {DATASET})")
    run_and_plot_pca(mle_params_all_moments_df_pca_efa, dataset_plots_dir / "MLE_pca_plot.png", title=f"PCA Plot for MLE Parameters (Dataset = {DATASET})")

    loadings_generative = run_exploratory_factor_analysis(df_generative_pca_efa, 3)
    loadings_mle = run_exploratory_factor_analysis(mle_params_all_moments_df_pca_efa, 3)

     # drop the columns
    df_generative = df_generative.drop(columns=drop_list)
    mle_params_all_moments_df = mle_params_all_moments_df.drop(columns=drop_list)

    # Save the factor loadings to CSV
    loadings_generative.to_csv(dataset_plots_dir / f"{model_id}_factor_loadings_generative.csv")
    loadings_mle.to_csv(dataset_plots_dir / "MLE_factor_loadings.csv")

    # check for normality of the data
    for col in df_generative.columns:
        if col in accuracy_vars:
            # logit transform the accuracy variables
            df_generative[col] = np.log(df_generative[col]/(1-df_generative[col]))
            mle_params_all_moments_df[col] = np.log(mle_params_all_moments_df[col]/(1-mle_params_all_moments_df[col]))
        else:
            stat, p_value = shapiro(df_generative[col])
            if p_value < 0.05:
                df_generative[col] = np.log(df_generative[col])
                mle_params_all_moments_df[col] = np.log(mle_params_all_moments_df[col])
                print(f"Transformed {col} to log scale")

    # do mean imputation for missing values
    df_generative = df_generative.fillna(df_generative.mean())

    # impute only numerical missing values with mean - ignore "session" column
    numerical_cols = mle_params_all_moments_df.select_dtypes(include=[np.number]).columns
    if 'session' in numerical_cols:
        numerical_cols = numerical_cols.drop('session')
    mle_params_all_moments_df[numerical_cols] = mle_params_all_moments_df[numerical_cols].fillna(mle_params_all_moments_df[numerical_cols].mean())

    
    # Define SEM models
    sem_model_main_all_params = """
    InhibitoryControl =~ Countermanding_reaction_time_param1 + Countermanding_reaction_time_param2 + Stroop_reaction_time_param1 + Stroop_reaction_time_param2 + D2_hit_accuracy_param1 + PasatPlus_correctly_answered_param1
    # WorkingMemory =~ CorsiComplex_param1 + CorsiComplex_param2    
    Cognitive_Flexibility =~ PasatPlus_correctly_answered_param1 + D2_hit_accuracy_param1 + CorsiComplex_param1 + CorsiComplex_param2
    # WorkingMemory ~ InhibitoryControl
    # Cognitive_Flexibility ~ WorkingMemory
    Cognitive_Flexibility ~ InhibitoryControl
    # executive_function =~ Cognitive_Flexibility + WorkingMemory + InhibitoryControl
    # executive_function ~ 1
    """

    sem_model_main_param_only = """
        # Measurement Models: Latent variables measured by observed variables
        InhibitoryControl =~ Countermanding_reaction_time_param1 + Stroop_reaction_time_param1 + D2_hit_accuracy_param1 + PasatPlus_correctly_answered_param1
        WorkingMemory =~ CorsiComplex_param1  
        #Cognitive_Flexibility =~ PasatPlus_correctly_answered_param1 + D2_hit_accuracy_param1 + CorsiComplex_param1
        
        # Structural Model: Relationships between latent variables
        WorkingMemory ~ InhibitoryControl
        # Cognitive_Flexibility ~ WorkingMemory
        #Cognitive_Flexibility ~ InhibitoryControl
        
        # # Second-order Latent Variable: executive_function
        # executive_function =~ Cognitive_Flexibility + WorkingMemory + InhibitoryControl
        # executive_function ~ 1
        """

    # drop some features to address multicollinearity
    # df_generative = df_generative.drop(columns=['Countermanding_reaction_time_param2', 'Stroop_reaction_time_param2', 'CorsiComplex_param2', 'SimpleSpan_param2', 'RunningSpan_correct_w_len_3_param1'])


    logging.info("=================== +++++ SEM ALL PARAMETERS ++++================================")
    logging.info("SEM Model: ", sem_model_main_all_params)
    # Build and log SEM results for generative data
    results, fit_measures_generative,_ = build_and_log_sem_results(df_generative, sem_model_main_all_params)
    log_sem_results("GENERATIVE MODEL",results, fit_measures_generative)

    # Build and log SEM results for raw MLE parameters
    results, fit_measures_raw,_ = build_and_log_sem_results(mle_params_all_moments_df, sem_model_main_all_params)
    log_sem_results("RAW DATA MLE PARAMS", results, fit_measures_raw)

    # Compare the fits between generative model and raw MLE model
    compare_fits(fit_measures_generative, fit_measures_raw)

    # drop any columns with suffix _param2
    df_generative = df_generative.loc[:,~df_generative.columns.str.endswith('_param2')]
    mle_params_all_moments_df = mle_params_all_moments_df.loc[:,~mle_params_all_moments_df.columns.str.endswith('_param2')]

    logging.info("=================== +++++ SEM MAIN PARAMETERS ++++================================")
    logging.info("SEM Model: ", sem_model_main_param_only)
    # Build and log SEM results for generative data
    results, fit_measures_generative,_ = build_and_log_sem_results(df_generative, sem_model_main_param_only)
    log_sem_results("GENERATIVE MODEL",results, fit_measures_generative)

    # Build and log SEM results for raw MLE parameters
    results, fit_measures_raw, _= build_and_log_sem_results(mle_params_all_moments_df, sem_model_main_param_only)
    log_sem_results("RAW DATA MLE PARAMS", results, fit_measures_raw)

    # Compare the fits between generative model and raw MLE model
    compare_fits(fit_measures_generative, fit_measures_raw)

def build_sem_model(df, sem_model):
    """
    Function to load data, perform basic checks, build the SEM model, and fit it.
    
    Args:
    - df: Pandas DataFrame containing the SEM data.
    - sem_model: String defining the SEM model in semopy syntax.
    
    Returns:
    - results: DataFrame with the summary of estimated parameters and factor loadings.
    - fit_measures: Dictionary of fit measures (CFI, RMSEA, etc.).
    - latent_values: DataFrame with the latent values assigned to each individual, along with their PIDs.
    """
    
    # ---- 1. Basic Data Checks ----
    
    # 1.1 Check for missing values
    missing_data = df.isnull().sum()
    if missing_data.any():
        logging.info("Warning: MISSING DATA found in the following columns:")
        logging.info(missing_data[missing_data > 0])
        logging.info("Rows with missing data will be dropped...")
        df = df.dropna()  # Drop rows with missing values
    else:
        logging.info("No missing data found")
    
    
    # Remove non-numeric columns (but keep PID)
    df_numeric = df.select_dtypes(include=[np.number])  # Keep numeric columns only
    
    # 1.3 Check for multicollinearity
    corr_matrix = df_numeric.corr()
    high_corr = np.where(np.abs(corr_matrix) > 0.6)
    high_corr_pairs = [(df_numeric.columns[i], df_numeric.columns[j]) for i, j in zip(*high_corr) if i != j]
    if high_corr_pairs:
        logging.info("Warning: Multicollinearity detected between the following variables:")
        logging.info(high_corr_pairs)
    else:
        logging.info("No multicollinearity detected")
    
    # 1.4 Check for univariate normality
    normality_results = {}
    for col in df_numeric.columns:
        stat, p_value = shapiro(df_numeric[col])
        normality_results[col] = p_value
        if p_value < 0.05:
            logging.info(f"Warning: {col} is not normally distributed (p = {p_value:.4f})")
            # df_numeric[col] = np.log(df_numeric[col])
            logging.info(f"Transformed {col} to log scale")
    
    # 1.5 Check sample size
    num_obs, num_vars = df_numeric.shape
    if num_obs < 200:
        logging.info(f"Warning: Sample size is small ({num_obs} observations). SEM models may require larger sample sizes.")
    else:
        logging.info(f"Sample size is adequate ({num_obs} observations)")
    
    # ---- 2. SEM Model Building and Fitting ----
    
    model = Model(sem_model)
    model.load_dataset(df_numeric)  # Load the numeric dataset without PID
    opt = Optimizer(model)
    opt.optimize()
    
    # 2.4 Summarize the results (loadings, path coefficients, etc.)
    results = inspect(opt)
    
    # 2.5 Calculate fit measures (CFI, RMSEA, etc.)
    fit_measures = stats.gather_statistics(opt)

    # Create a formatted string for fit measures
    fit_measures_output = {
        "Chi-Square": fit_measures.chi2[0],  # Extracting chi-square value
        "Degrees of Freedom": fit_measures.dof,
        "CFI": fit_measures.cfi,
        "TLI": fit_measures.tli,
        "RMSEA": fit_measures.rmsea,
        "AIC": fit_measures.aic,
        "BIC": fit_measures.bic,
        "GFI": fit_measures.gfi,
        "AGFI": fit_measures.agfi,
        "NFI": fit_measures.nfi,
    }

    # If p-value exists, log it, else log a placeholder
    p_value = fit_measures.chi2[1] if len(fit_measures.chi2) > 1 else "Not available"
    fit_measures_output["p-value"] = p_value

    
    # 2.6 Extract latent values (factor scores)
    # latent_values = opt.predict_factors(df_numeric)  # Get the factor scores from the optimizer
    # latent_values = opt.model.predict_factors(df_numeric)  # Get the latent values
    # latent_values_df = pd.DataFrame(latent_values)  # Create DataFrame

    latent_values_df = None
    
    
    return results, fit_measures_output, latent_values_df

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate latent space results.")
    parser.add_argument("--latent_dim", type=int, default=3, help="Dimension of the latent space")
    parser.add_argument("--model_id", type=str, default="fragrant-lion-356", help="Model ID")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to generate in each dimension")

    args = parser.parse_args()

    main(args)

    logging.shutdown()
