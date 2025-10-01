#!/usr/bin/env python3
"""
Script to create ground truth sessions by sampling latent points from a 2D DLVM model.

This script:
1. Loads a specified 2D model
2. Extracts meu_z parameters to determine latent space bounds
3. Uniformly samples 100 points within those bounds
4. Computes marginal fits for each point using the model
5. Creates visualizations including:
   - 2D scatter plot of latent points
   - Individual marginal fit plots for each point
   - Combined PDF of all marginal plots
6. Saves parameter dictionaries with systematic IDs (A001, A002, etc.)

Usage:
    python create_ground_truth_sessions.py --model_path path/to/model.pt --output_dir output/
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# Add parent directory to path to import utils
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(parent_dir)

from utils.data_distribution_utils import (
    load_trained_model, 
    DATASET,
    RANDOM_SEED
)
from utils.grid_search_utils import (
    generate_grid,
    get_predictions_dicts_from_latent_points
)
from visualization.create_marginal_fits import (
    visualize_marginal_fits_many_methods,
    combine_pdfs_in_folder
)
from visualization.create_plots import (
    visualize_latent_space_to_parameter_mapping
)
# import set_seed
from utils.set_seed import set_seed

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('create_ground_truth_sessions.log')
        ]
    )
    return logging.getLogger(__name__)


def sample_uniform_latent_points(model, num_points=100):
    """
    Sample uniform points in latent space using existing generate_grid function.
    
    Args:
        model: The loaded DLVM model
        num_points (int): Number of points to sample (default: 100)
    
    Returns:
        torch.Tensor: Tensor of shape (num_points, 2) containing sampled latent points
    """
    logger = logging.getLogger(__name__)
    
    # Use existing generate_grid function to get points within model bounds
    # For 2D latent space, we need sqrt(num_points) per dimension to get approximately num_points total
    if model.meu_z.shape[1] == 1: # 1D latent space
        grid_points_per_dim = num_points
    else:
        grid_points_per_dim = int(np.ceil(np.sqrt(num_points)))
    
    logger.info(f"Generating grid with {grid_points_per_dim} points per dimension")
    grid_points = generate_grid(model, num_points=grid_points_per_dim)
    
    # Ensure we have exactly 88 points by sampling randomly from the grid
    if len(grid_points) > 88:
        set_seed(2*RANDOM_SEED) # just a new random seed for this function
        # Randomly sample 88 points from the grid
        indices = torch.randperm(len(grid_points))[:88]
        sampled_points = grid_points[indices]
    else:
        # If we have fewer points than requested, use all available
        sampled_points = grid_points
        logger.warning(f"Only {len(grid_points)} points available, using all of them")
    sampled_points = sampled_points.squeeze(0) # shape (num_points, 2)
    logger.info(f"Sampled {len(sampled_points)} points from latent space, shape: {sampled_points.shape}")
    return sampled_points


def create_latent_space_scatter(latent_points, output_dir, model_id=None, latent_dim=None):
    """
    Create a scatter plot of the sampled latent points.
    
    Args:
        latent_points (torch.Tensor): Tensor of shape (num_points, latent_dim) containing latent points
        output_dir (str): Directory to save the plot
        model_id (str): Model ID for filename
        latent_dim (int): Latent dimension
    
    Returns:
        str: Path to the saved scatter plot, or None if skipped
    """
    logger = logging.getLogger(__name__)
    
    # Skip visualization for 3D+ dimensions
    if latent_dim >= 3:
        logger.info(f"Skipping latent space scatter plot for {latent_dim}D latent space (not supported)")
        return None
    
    # Convert to numpy for plotting
    points_np = latent_points.detach().cpu().numpy()
    
    if latent_dim == 1:
        # Create 1D scatter plot
        plt.figure(figsize=(12, 6))
        plt.scatter(points_np[:, 0], np.zeros_like(points_np[:, 0]), alpha=0.6, s=100)
        
        # Add point IDs as annotations
        for i, x in enumerate(points_np[:, 0]):
            point_id = f"LD{latent_dim}-{i+1:03d}"
            plt.annotate(point_id, (x, 0), xytext=(0, 10), textcoords='offset points', 
                        fontsize=8, alpha=0.7, ha='center')
        
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('')
        plt.title('Sampled Latent Points in 1D Latent Space')
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.1, 0.1)  # Small range around y=0
        
    elif latent_dim == 2:
        # Create 2D scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(points_np[:, 0], points_np[:, 1], alpha=0.6, s=50)
        
        # Add point IDs as annotations
        for i, (x, y) in enumerate(points_np):
            point_id = f"LD{latent_dim}-{i+1:03d}"
            plt.annotate(point_id, (x, y), xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.7)
        
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.title('Sampled Latent Points in 2D Latent Space')
        plt.grid(True, alpha=0.3)
    
    # Save the plot
    scatter_path = os.path.join(output_dir, f'D{latent_dim}_latent_space_scatter_{model_id}.pdf')
    plt.savefig(scatter_path, format='pdf', dpi=300, bbox_inches='tight')
    # save the plot as a png
    plt.savefig(scatter_path.replace('.pdf', '.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved latent space scatter plot to: {scatter_path}")
    return scatter_path


def create_latent_space_parameter_mapping(model, latent_points, output_dir, args):
    """
    Create latent space to parameter mapping visualization using the new function.
    
    Args:
        model: The loaded DLVM model
        latent_points (torch.Tensor): Tensor of shape (num_points, latent_dim) containing latent points
        output_dir (str): Directory to save the plot
        args (argparse.Namespace): Arguments passed to the script (uses figure_width for output size control)
    
    Returns:
        str: Path to the saved mapping plot, or None if skipped
    """
    logger = logging.getLogger(__name__)
    
    # Skip visualization for 1D and 3D+ dimensions
    if args.latent_dim == 1:
        logger.info("Skipping latent space parameter mapping for 1D latent space (not meaningful)")
        return None
    elif args.latent_dim >= 3:
        logger.info(f"Skipping latent space parameter mapping for {args.latent_dim}D latent space (not supported)")
        return None
    
    logger.info("Creating latent space to parameter mapping visualization...")
    
    # Create the visualization using our new function
    fig = visualize_latent_space_to_parameter_mapping(
        model=model,
        validation_meu_z=latent_points,
        colormap='Grays',
        use_activation=True,
        show_ids=False,
        plot_title=f"Latent Space to Distributional Parameter Mapping for DLVM-{args.latent_dim} model",
        figure_width=args.figure_width
    )
    
    # Save the plot
    mapping_path = os.path.join(output_dir, f'D{args.latent_dim}_latent_space_parameter_mapping_{args.model_id}.pdf')
    fig.savefig(mapping_path, format='pdf', dpi=300, bbox_inches='tight')
    # save as png too
    fig.savefig(mapping_path.replace('.pdf', '.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved latent space parameter mapping to: {mapping_path}")
    return mapping_path


def create_individual_marginal_plots(latent_points, parameter_dicts, output_dir, args, model_id=None, latent_dim=None):
    """
    Create individual marginal fit plots for each sampled point.
    
    Args:
        latent_points (torch.Tensor): Tensor of shape (num_points, 2) containing latent points
        parameter_dicts (list): List of parameter dictionaries for each point
        output_dir (str): Directory to save the plots
        args (argparse.Namespace): Arguments passed to the script
    Returns:
        list: List of paths to saved individual PDF files
    """
    logger = logging.getLogger(__name__)
    
    marginal_dir = os.path.join(output_dir, 'marginal')
    os.makedirs(marginal_dir, exist_ok=True)
    
    saved_paths = []
    
    for i, (latent_point, params) in enumerate(zip(latent_points, parameter_dicts)):
        point_id = f"LD{latent_dim}-{i+1:03d}"
        logger.info(f"Creating marginal plot for point {point_id}")
        
        # create empty data du
        data_dict = {}
        for metric in params.keys():
            data_dict[metric] = []
    
        # add the predicted parameters to the data dict
        
        # Create models_data for visualization
        models_data = [
            {
                'params': params,
                'raw_data': None,  # No raw data for simulated points
                'label': f'{point_id}',
                'color': '#1f77b4',  # Blue
                'alpha': 0.8
            }
        ]
        
        # Create the marginal fit visualization
        try:
            fig = visualize_marginal_fits_many_methods(
                models_data=models_data,
                show_raw_data=False,  # No raw data to show
                show_curves=True,
                line_thickness=2,
                show_grid=False,
                verbose=False,
                show_legend_per_task=False,
                plot_title=f"Synthetic point {point_id}(meu_z={[float(round(point, 3)) for point in latent_point.detach().cpu().numpy().tolist()]}) sampled from DLVM model '{args.model_id}' latent space",
                title_fontsize=12
            )
            
            # Save individual plot
            plot_path = os.path.join(marginal_dir, f"D{latent_dim}_marginal_{point_id}_{model_id}.pdf")
            plt.savefig(plot_path, format='pdf', dpi=300, bbox_inches='tight')
            plt.close()
            
            saved_paths.append(plot_path)
            logger.info(f"Saved marginal plot for {point_id} to: {plot_path}")
            
        except Exception as e:
            logger.error(f"Failed to create marginal plot for {point_id}: {e}")
            logger.error("Full error trace:", exc_info=True)
            continue
    
    logger.info(f"Created {len(saved_paths)} individual marginal plots")
    return saved_paths


def save_parameter_data_with_ids(latent_points, parameter_dicts, output_dir, 
                                 model_id=None, latent_dim=None, verbose=True):
    """
    Save parameter dictionaries with systematic IDs and latent variables.
    
    Args:
        latent_points (torch.Tensor): Tensor of shape (num_points, 2) containing latent points
        parameter_dicts (list): List of parameter dictionaries for each point
        output_dir (str): Directory to save the data
    
    Returns:
        tuple: Paths to (summary_file, ground_truth_parameters_file, latent_variables_file)
    """
    logger = logging.getLogger(__name__)
    
    simulated_data_dir = os.path.join(output_dir, 'simulated_data')
    os.makedirs(simulated_data_dir, exist_ok=True)

    id_label ={
            "1": "LD1",
            "2": "LD2",
            "3": "LD3",
            "4": "LD4",
            "5": "LD5"
    }
    
    # Save individual parameter and latent coordinate files
    for i, (latent_point, params) in enumerate(zip(latent_points, parameter_dicts)):
        point_id = f"{id_label[str(latent_dim)]}-{i+1:03d}"
        
        # Save parameter dictionary
        param_file = os.path.join(simulated_data_dir, f"{point_id}_params.pt")
        # torch.save(params, param_file)
        
        # Save latent coordinates
        coord_file = os.path.join(simulated_data_dir, f"{point_id}_latent_coords.pt")
        # torch.save(latent_point, coord_file)
        
        logger.info(f"Saved individual data for {point_id}")
    
    # Create summary file with all parameters and latent coordinates
    summary_data = {}
    ground_truth_parameters = {}
    latent_variables = {}
    
    for i, (latent_point, params) in enumerate(zip(latent_points, parameter_dicts)):
        point_id = f"LD{latent_dim}-{i+1:03d}"
        
        # Store in summary data
        summary_data[point_id] = {
            'latent_coordinates': latent_point.detach().cpu().numpy(),
            'parameters': params
        }
        
        # Store parameters separately
        ground_truth_parameters[point_id] = params
        
        # Store latent variables separately
        latent_variables[point_id] = latent_point.detach().cpu().numpy()
    
    # Save all files
    summary_file = os.path.join(simulated_data_dir, f'D{latent_dim}_all_simulated_data_{model_id}.pt')
    torch.save(summary_data, summary_file)
    
    ground_truth_parameters_file = os.path.join(simulated_data_dir, f'D{latent_dim}_synthetic_ground_truth_parameters_{model_id}.pt')
    torch.save(ground_truth_parameters, ground_truth_parameters_file)
    
    latent_variables_file = os.path.join(simulated_data_dir, f'D{latent_dim}_synthetic_latent_variables_{model_id}.pt')
    torch.save(latent_variables, latent_variables_file)
    
    if verbose:
        logger.info(f"Saved summary data to: {summary_file}")
        logger.info(f"Saved ground truth parameters to: {ground_truth_parameters_file}")
        logger.info(f"Saved latent variables to: {latent_variables_file}")
    
    return summary_file, ground_truth_parameters_file, latent_variables_file


def main():
    """Main function to orchestrate the ground truth session creation."""
    parser = argparse.ArgumentParser(description='Create ground truth sessions from 2D DLVM model')
    parser.add_argument('--model-path', type=str, default=None, help='Path to the DLVM model file')
    parser.add_argument("--model-id", type=str, default=None, help="Model ID to load") # avoids requiring model path
    parser.add_argument('--output_dir', type=str, default='synthetic_sessions_ground_truth',
                       help='Output directory for results')
    parser.add_argument('--num_points', type=int, default=100,
                       help='Number of points to sample (default: 100)')
    parser.add_argument('--latent_dim', type=int, default=2,
                       help='Latent dimension of the model (default: 2)')
    parser.add_argument('--figure-width', type=float, default=6.4,
                       help='Width in inches for the latent-to-parameter mapping figure (default: 6.4)')
    parser.add_argument("--generate_marginal_visualizations", action="store_true", default=False,
                        help="Generate marginal visualizations for each point")
    
    args = parser.parse_args()
    
    if args.latent_dim == 1 and args.model_id is None:
        args.model_id = "wolverine-zoom-7298"  # No default 1D model
    elif args.latent_dim == 2 and args.model_id is None:
        args.model_id = "mongoose-dive-7464"
    elif args.latent_dim == 3 and args.model_id is None:
        args.model_id = "beaver-slide-5310"
    elif args.latent_dim == 4 and args.model_id is None:
        args.model_id = None
    elif args.latent_dim == 5 and args.model_id is None:
        args.model_id = None
    
    if args.model_id is None:
        logger.error("Model ID is not specified. Use --model-id to specify the model file.")
        sys.exit(1)
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting ground truth session creation")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Number of points: {args.num_points}")
    logger.info(f"Latent dimension: {args.latent_dim}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'simulated_data'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'marginal'), exist_ok=True)
    
    # Initialize variables for conditional generation
    individual_paths = []
    combined_pdf_path = None
    
    try:
        # Step 1: Load the model
        logger.info("Loading trained model...")
        if args.model_id is not None: 
            args.model_path = os.path.join(script_dir, f"../../saved_models/{DATASET}/heldout_obsmulti/variationalNN_relevant_only_latentdim{args.latent_dim}_{args.model_id}.pt")
        
        if args.model_path is None or not os.path.exists(args.model_path):
            logger.error("DLVM model path is not specified or does not exist. Use --model-path to specify the model file.")
            sys.exit(1)
        model = load_trained_model(args.latent_dim, model_path=args.model_path)
        logger.info("Model loaded successfully")

        # check that model.meu_z dimension matches args.latent_dim
        if model.meu_z.shape[1] != args.latent_dim:
            logger.error(f"Model.meu_z dimension ({model.meu_z.shape[1]}) does not match args.latent_dim ({args.latent_dim})")
            sys.exit(1)
        
        # Step 2: Sample uniform points in latent space
        logger.info("Sampling uniform points in latent space...")
        latent_points = sample_uniform_latent_points(model, num_points=args.num_points)
        
        # Step 3: Compute marginal fits for each point
        logger.info("Computing marginal fits for sampled points...")
        parameter_dicts = get_predictions_dicts_from_latent_points(
            latent_points, model, model_type="NN", with_activation=True
        )
        
        # Step 4: Create latent space scatter plot
        logger.info("Creating latent space scatter plot...")
        scatter_path = create_latent_space_scatter(latent_points, args.output_dir, args.model_id, args.latent_dim)
        
        # Step 4.5: Create latent space to parameter mapping visualization
        logger.info("Creating latent space to parameter mapping visualization...")
        mapping_path = create_latent_space_parameter_mapping(model, latent_points, args.output_dir, args)
        
        
        if args.generate_marginal_visualizations:
            # Step 5: Create individual marginal plots
            logger.info("Creating individual marginal plots...")
            individual_paths = create_individual_marginal_plots(
                latent_points, parameter_dicts, args.output_dir, args, args.model_id, args.latent_dim
            )
            
            # Step 6: Create combined PDF using existing function
            logger.info("Creating combined PDF of all marginal plots...")
            marginal_dir = os.path.join(args.output_dir, f'marginal')
            combined_pdf_path = os.path.join(args.output_dir, f'D{args.latent_dim}_combined_marginals_{args.model_id}.pdf')
            combine_pdfs_in_folder(marginal_dir, combined_pdf_path)
            logger.info(f"Created combined PDF: {combined_pdf_path}")
            
            # Delete temporary individual marginal PDFs to save space
            logger.info("Cleaning up temporary individual marginal PDFs...")
            pdf_files = [f for f in os.listdir(marginal_dir) if f.endswith('.pdf') and not f.startswith('_')]
            deleted_count = 0
            for pdf_file in pdf_files:
                pdf_path = os.path.join(marginal_dir, pdf_file)
                try:
                    os.remove(pdf_path)
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {pdf_file}: {e}")
            logger.info(f"Deleted {deleted_count} temporary individual marginal PDFs")
        
        # Step 7: Save parameter data with IDs
        logger.info("Saving parameter data with IDs...")
        summary_file, ground_truth_parameters_file, latent_variables_file = save_parameter_data_with_ids(
            latent_points, parameter_dicts, args.output_dir, args.model_id, args.latent_dim, verbose=False
        )
        
        # Summary
        logger.info("=" * 50)
        logger.info("GROUND TRUTH SESSION CREATION COMPLETED")
        logger.info("=" * 50)
        logger.info(f"Created {len(latent_points)} simulated points")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Files created:")
        if scatter_path is not None:
            logger.info(f"  - Latent space scatter: {scatter_path}")
        else:
            logger.info(f"  - Latent space scatter: Skipped (not supported for {args.latent_dim}D)")
        if mapping_path is not None:
            logger.info(f"  - Latent space parameter mapping: {mapping_path}")
        else:
            logger.info(f"  - Latent space parameter mapping: Skipped (not supported for {args.latent_dim}D)")
        
        if args.generate_marginal_visualizations:
            logger.info(f"  - Individual marginal plots: {len(individual_paths)} files")
            logger.info(f"  - Combined marginal PDF: {combined_pdf_path}")
        else:
            logger.info(f"  - Individual marginal plots: Skipped (--generate_marginal_visualizations not used)")
            logger.info(f"  - Combined marginal PDF: Skipped (--generate_marginal_visualizations not used)")
        
        logger.info(f"  - Parameter data: {summary_file}")
        logger.info(f"  - Ground truth parameters: {ground_truth_parameters_file}")
        logger.info(f"  - Latent variables: {latent_variables_file}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Error during ground truth session creation: {e}")
        raise


if __name__ == "__main__":
    main()
