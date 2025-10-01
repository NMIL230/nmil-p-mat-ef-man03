#!/usr/bin/env python3
"""
Script to visualize latent space trajectories from a 2D DLVM model.

This script:
1. Takes a 2D DLVM model and a nx2 matrix of latent positions
2. Creates a trajectory visualization connecting points 1 to n in sequence
3. Shows the path through the latent space with arrows indicating direction
4. Optionally computes and displays parameter changes along the trajectory

Usage:
    python visualize_DALE_latent_trajectory.py --model_path path/to/model.pt --latent_positions path/to/positions.pt --output_dir output/
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import pandas as pd
from scipy.interpolate import griddata
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
    get_predictions_dicts_from_latent_points,
    compute_loss_landscape
)
# from visualization.create_marginal_fits import (
#     combine_pdfs_in_folder
# )
from utils.set_seed import set_seed

# Local implementation to avoid wandb import issues
def combine_pdfs_in_folder(folder_path, output_path):
    """Combine all PDF files in a folder into a single PDF."""
    try:
        from PyPDF2 import PdfMerger
        merger = PdfMerger()

        # Iterate over all files in the folder
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith('.pdf') and "combined" not in filename:
                # Open each PDF file and append it to the merger
                file_path = os.path.join(folder_path, filename)
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as pdf_file:
                        merger.append(pdf_file)
                else:
                    print(f"File {file_path} does not exist and will be skipped.")

        # Write the combined PDF to the output file
        with open(output_path, 'wb') as output_file:
            merger.write(output_file)
        merger.close()
        print(f"Combined PDFs saved to: {output_path}")
    except ImportError:
        print("PyPDF2 not available, skipping PDF combination")
    except Exception as e:
        print(f"Error combining PDFs: {e}")

def compute_position_rmse(latent_positions, ground_truth_position):
    """Compute RMSE between latent positions and ground truth position."""
    return torch.sqrt(torch.mean((latent_positions - ground_truth_position) ** 2))


def setup_logging(verbose=False):
    """Set up logging configuration."""
    if verbose:
        level = logging.DEBUG
        handlers = [
            logging.StreamHandler(),
            logging.FileHandler('visualize_DALE_latent_trajectory.log')
        ]
    else:
        level = logging.INFO
        handlers = [
            logging.FileHandler('visualize_DALE_latent_trajectory.log')
        ]
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)


def load_latent_positions(positions_path):
    """
    Load latent positions from file.
    
    Args:
        positions_path (str): Path to the CSV file containing latent positions
        
    Returns:
        torch.Tensor: Tensor of shape (n, 2) containing latent positions
    """
    logger = logging.getLogger(__name__)
    
    # Load CSV file and extract the meu_z entry
    df = pd.read_csv(positions_path)
    
    # Parse the string representations of lists in meu_z column
    positions_list = []
    for meu_z_str in df['meu_z']:
        # Remove brackets and split by comma, then convert to float
        # Handle both single and double brackets
        clean_str = meu_z_str.strip('[]')
        if clean_str.startswith('[') and clean_str.endswith(']'):
            clean_str = clean_str[1:-1]  # Remove outer brackets if present
        
        # Split by comma and convert to floats
        values = [float(x.strip()) for x in clean_str.split(',')]
        positions_list.append(values)
    
    # Convert to numpy array and then to torch tensor
    positions = np.array(positions_list, dtype=np.float32)
    positions = torch.from_numpy(positions)
    
    logger.info(f"Loaded {len(positions)} latent positions with shape: {positions.shape}")
    return positions



def plot_loss_landscape_heatmap(ax, loss_landscape, loss_meu_z, trajectory_id="trajectory", normalize=True, cmap='viridis'):
    """
    Plot a loss landscape heatmap as background on the given axis.
    
    Args:
        ax: Matplotlib axis to plot on
        loss_landscape (torch.Tensor): Tensor of loss values
        loss_meu_z (torch.Tensor): Tensor of latent positions corresponding to loss values
        trajectory_id (str): Identifier for logging purposes
        normalize (bool): Whether to normalize loss values per session (default: True)
        
    Returns:
        matplotlib.contour.QuadContourSet: The contour plot object for potential colorbar creation
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Adding loss landscape heatmap for session {trajectory_id}")
    
    # Convert to numpy for plotting
    loss_np = loss_landscape.detach().cpu().numpy()
    meu_z_np = loss_meu_z.squeeze(1).detach().cpu().numpy()  # Remove middle dimension
    
    # Normalize loss values per session if requested
    # if normalize:
    #     from sklearn.preprocessing import MinMaxScaler
    #     scaler = MinMaxScaler()
    #     loss_np = scaler.fit_transform(loss_np.reshape(-1, 1)).flatten()
    #     # set loss value above 0.1 to 0.1
    #     loss_np = np.where(loss_np > 0.1, 0.1, loss_np)
    #     # split the loss_np value into deciles and set the loss value to the decile value

    
    # Create grid for interpolation
    x_coords = meu_z_np[:, 0]
    y_coords = meu_z_np[:, 1]
    
    # Create a regular grid for interpolation
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # Create grid with higher resolution for smooth heatmap
    grid_resolution = 100
    xi = np.linspace(x_min, x_max, grid_resolution)
    yi = np.linspace(y_min, y_max, grid_resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Interpolate loss values onto regular grid
    loss_grid = griddata((x_coords, y_coords), loss_np, (xi_grid, yi_grid), method='cubic', fill_value=np.nan)
    # clip the loss_grid between 0 and 0.1
    loss_grid = np.clip(loss_grid, 0, 0.1)
    
    # Create grayscale heatmap (darker = lower loss/better)
    im = ax.contourf(xi_grid, yi_grid, loss_grid, levels=5, cmap=cmap, alpha=0.6, zorder=0)
    
    logger.info(f"Loss landscape heatmap added with {len(loss_np)} points")
    return im

def plot_trajectory_on_axis(ax, latent_positions, model, ground_truth_position=None, rmse=None, loss_at_final_latent_position=None, trajectory_id="trajectory", 
                           loss_landscape=None, loss_meu_z=None, cmap='viridis', show_values_in_legend=True, show_axis_labels=True, 
                           drop_odd_number_annotations=True, scale_factor=1.0, text_scale_factor=None, show_colorbar=False,
                           show_legend=True, max_length_to_plot=240):
    """
    Plot a trajectory on a given matplotlib axis. Reusable plotting function.
    
    Args:
        ax: Matplotlib axis to plot on
        latent_positions (torch.Tensor): Tensor of shape (n, 2) containing latent positions
        model: DLVM model for consistent scaling
        ground_truth_position (torch.Tensor): Ground truth meu_z position for RMSE calculation
        rmse (float): RMSE value to display in legend
        loss_at_final_latent_position (float): Loss value at the final latent position to display in legend 
        trajectory_id (str): Identifier for the trajectory
        loss_landscape (torch.Tensor): Optional tensor of loss values for heatmap background
        loss_meu_z (torch.Tensor): Optional tensor of latent positions corresponding to loss values
        show_values_in_legend (bool): Whether to show values in legend
        show_axis_labels (bool): Whether to show axis labels
        drop_odd_number_annotations (bool): Whether to drop odd number annotations
        scale_factor (float): Scaling factor for symbols and text (default: 1.0)
        text_scale_factor (float): Separate scaling factor for text (default: None, uses scale_factor)
        show_colorbar (bool): Whether to show a colorbar for the loss landscape heatmap
        show_legend (bool): Whether to show a legend for the plot
        max_length_to_plot (int): DEPRECATED - No longer used. Truncation is now done upstream before plotting.
    Returns:
        matplotlib.contour.QuadContourSet or None: The heatmap contour object for colorbar creation
    """
    logger = logging.getLogger(__name__)
    
    # Use text_scale_factor if provided, otherwise use scale_factor
    if text_scale_factor is None:
        text_scale_factor = scale_factor
    
    # Add loss landscape heatmap as background if provided
    heatmap_im = None
    if loss_landscape is not None and loss_meu_z is not None:
        heatmap_im = plot_loss_landscape_heatmap(ax, loss_landscape, loss_meu_z, trajectory_id, cmap=cmap)
    
    # Convert to numpy for plotting (truncation already done upstream)
    positions_np = latent_positions.detach().cpu().numpy()
    
    # Calculate global range from model's latent space for consistent scaling
    model_x_range = model.meu_z[:, 0].cpu().max().item() - model.meu_z[:, 0].cpu().min().item()
    model_y_range = model.meu_z[:, 1].cpu().max().item() - model.meu_z[:, 1].cpu().min().item()
    global_max_range = max(model_x_range, model_y_range)
    
    # Scale arrow parameters based on global model range for consistency
    arrow_scale = global_max_range * 0.02  # 2% of the global range
    head_width = arrow_scale * 0.5
    head_length = arrow_scale * 0.3
    

    
    # Plot trajectory line with better styling
    ax.plot(positions_np[:, 0], positions_np[:, 1], 'k-', alpha=0.4, linewidth=2.5*scale_factor, zorder=1)
    
    # Plot all points with better styling - scale sizes
    scatter = ax.scatter(positions_np[:-1, 0], positions_np[:-1, 1], s=150*scale_factor, alpha=0.6, zorder=2)
    if show_values_in_legend:
        label = f'Final Position (RMSE: {rmse:.3f}, Log Prob: {loss_at_final_latent_position:.3f})' if rmse is not None else 'Final Position'
    else:
        label = 'Final Position'
    # Plot final position with standout symbol - scale sizes
    final_pos = positions_np[-1]
    ax.scatter(final_pos[0], final_pos[1], s=200*scale_factor, marker='o', color='lime', 
              alpha=0.9, zorder=5, edgecolors='black', linewidth=3*scale_factor, label=label)
    
    # Plot ground truth position if provided - scale sizes
    if ground_truth_position is not None:
        # Handle both torch.Tensor and numpy.ndarray types
        if isinstance(ground_truth_position, torch.Tensor):
            ground_truth_np = ground_truth_position.detach().cpu().numpy()
        else:
            ground_truth_np = ground_truth_position
        
        label = "Ground Truth"
        ax.scatter(ground_truth_np[0], ground_truth_np[1], s=350*scale_factor, marker='*', color='red', 
                  alpha=0.9, zorder=4, edgecolors='black', linewidth=2*scale_factor, label=label)
        logger.info(f"Ground truth position: {ground_truth_np} plotted")
    
    # Add point numbers and arrows with better styling                                     
    for i in range(len(positions_np)):
        # Point number with better positioning and styling - scale text sizes
        if i == len(positions_np) - 1:
            # Final position number - more prominent but not bold
            # Calculate scaled values for bbox
            scaled_pad = 0.3 * text_scale_factor
            scaled_linewidth = 1 * text_scale_factor
            ax.annotate(f'{i+1}', (positions_np[i, 0], positions_np[i, 1]), 
                       xytext=(10*text_scale_factor, 10*text_scale_factor), textcoords='offset points', 
                       fontsize=7*text_scale_factor, fontweight='normal',
                       bbox=dict(boxstyle=f'round,pad={scaled_pad}', facecolor='white', 
                                 alpha=0.9, edgecolor='black', linewidth=scaled_linewidth))
        else:
            # only show numbers if its even and above 10
            # Only annotate if not dropping odd numbers, or if even and >10 when dropping odds
            if not drop_odd_number_annotations or (drop_odd_number_annotations and i % 2 == 0) or (i <10):
                # Calculate scaled values for bbox
                scaled_pad = 0.2 * text_scale_factor
                scaled_linewidth = 0.5 * text_scale_factor
                ax.annotate(f'{i+1}', (positions_np[i, 0], positions_np[i, 1]), 
                        xytext=(8*text_scale_factor, 8*text_scale_factor), textcoords='offset points',  fontweight='bold',
                        fontsize=10*text_scale_factor, alpha=0.7,
                        bbox=dict(boxstyle=f'round,pad={scaled_pad}', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=scaled_linewidth))
        
        # Arrow to next point (except for last point)
        if i < len(positions_np) - 1:
            dx = positions_np[i+1, 0] - positions_np[i, 0]
            dy = positions_np[i+1, 1] - positions_np[i, 1]
            ax.arrow(positions_np[i, 0], positions_np[i, 1], dx*0.95, dy*0.95,
                    head_width=head_width*scale_factor, head_length=head_length*scale_factor, alpha=0.7, zorder=3)
    
    # Set fixed axis limits based on global model range with consistent padding
    padding = global_max_range * 0.1  # 10% padding
    # ax.set_xlim(model.meu_z[:, 0].cpu().min().item() - padding, model.meu_z[:, 0].cpu().max().item() + padding)
    # ax.set_ylim(model.meu_z[:, 1].cpu().min().item() - padding, model.meu_z[:, 1].cpu().max().item() + padding)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    # Set ticks to even numbers only
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_yticks([-4, -2, 0, 2, 4])
    # xticks and yticks size
    ax.tick_params(axis='x', labelsize=12*text_scale_factor)
    ax.tick_params(axis='y', labelsize=12*text_scale_factor)
    
    # Ensure equal aspect ratio for consistent scaling
    ax.set_aspect('equal', adjustable='box') 
    
    # Professional axis labels without bold - scale text sizes
    if show_axis_labels:
        ax.set_xlabel('Latent Dimension 1', fontsize=10*text_scale_factor)
        ax.set_ylabel('Latent Dimension 2', fontsize=10*text_scale_factor)

        # set font size for the x-tick labels
        ax.tick_params(axis='x', labelsize=8*text_scale_factor)
        # set font size for the y-tick labels
        ax.tick_params(axis='y', labelsize=8*text_scale_factor)
    
    # Professional grid styling
    ax.grid(False)
    ax.set_facecolor('white')
    
    # Add legend if ground truth is plotted - scale text size
    if ground_truth_position is not None and show_legend:
        ax.legend(fontsize=10*text_scale_factor, framealpha=0.9)
    
    return heatmap_im

def create_trajectory_visualization(latent_positions, output_dir, model=None, show_parameters=False, 
                                  trajectory_id="trajectory", colormap='viridis', ground_truth_position=None, limit_to_train_meu_z_range = True,
                                loss_landscape=None, loss_meu_z=None, cmap='viridis', loss_at_final_latent_position=None, final_meu_z=None,
                                  max_length_to_plot=240):
    """
        Create a trajectory visualization in latent space with ground truth comparison.
        

        Args:
            latent_positions (torch.Tensor): Tensor of shape (n, 2) containing latent positions (pre-truncated upstream)
            output_dir (str): Directory to save the plot
            model: Optional DLVM model for parameter computation
            show_parameters (bool): Whether to show parameter changes along trajectory
            trajectory_id (str): Identifier for the trajectory
            colormap (str): Colormap for trajectory coloring
            ground_truth_position (torch.Tensor): Ground truth meu_z position for RMSE calculation
            limit_to_train_meu_z_range (bool): Whether to limit the trajectory to the train meu_z range
            loss_landscape (torch.Tensor): Optional tensor of loss values for heatmap background
            loss_meu_z (torch.Tensor): Optional tensor of latent positions corresponding to loss values
            max_length_to_plot (int): DEPRECATED - No longer used. Truncation is now done upstream.
        Returns:
            tuple: (str: Path to the saved trajectory plot, float: RMSE value)
    
    """
    
    logger = logging.getLogger(__name__)
    

    # Convert to numpy for plotting
    positions_np = latent_positions.detach().cpu().numpy()
    

    # Calculate RMSE if ground truth is provided
    rmse = None
    if ground_truth_position is not None:
        logger.info(f"Ground truth position found for session {trajectory_id}: {ground_truth_position}")
        final_predicted = latent_positions[-1]  # Last predicted position
        logger.info(f"Final predicted position for session {trajectory_id}: {final_predicted}")
        
        # Handle both torch.Tensor and numpy.ndarray types
        if isinstance(ground_truth_position, torch.Tensor):
            ground_truth_np = ground_truth_position.detach().cpu().numpy()
            ground_truth_tensor = ground_truth_position
            logger.info(f"Ground truth is torch.Tensor, converted to numpy: {ground_truth_np}")
        else:
            ground_truth_np = ground_truth_position
            ground_truth_tensor = torch.from_numpy(ground_truth_position)
            logger.info(f"Ground truth is numpy.ndarray: {ground_truth_np}")
        
        # Debug the RMSE calculation
        logger.info(f"Final predicted shape: {final_predicted.shape}, type: {type(final_predicted)}")
        logger.info(f"Ground truth tensor shape: {ground_truth_tensor.shape}, type: {type(ground_truth_tensor)}")
        logger.info(f"Final predicted values: {final_predicted}")
        logger.info(f"Ground truth values: {ground_truth_tensor}")
        
        # Calculate RMSE
        diff = final_predicted - ground_truth_tensor
        squared_diff = diff ** 2
        mean_squared = torch.mean(squared_diff)
        # import pdb; pdb.set_trace()
        rmse = compute_position_rmse(final_predicted, ground_truth_tensor)
        
        logger.info(f"Difference: {diff}")
        logger.info(f"Squared difference: {squared_diff}")
        logger.info(f"Mean squared: {mean_squared}")
        logger.info(f"RMSE for session {trajectory_id}: {rmse:.4f}")
    else:
        logger.warning(f"No ground truth position provided for session {trajectory_id}")
    
    # Create figure with larger size for better resolution
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1)
    
    # Use the reusable plotting function (data already truncated upstream)
    heatmap_im = plot_trajectory_on_axis(ax, latent_positions, model, ground_truth_position, rmse, loss_at_final_latent_position, trajectory_id,
                                       loss_landscape=loss_landscape, loss_meu_z=loss_meu_z, cmap=cmap, show_values_in_legend=True, show_axis_labels=True)
    
    # Add colorbar if loss landscape was plotted
    if heatmap_im is not None:
        cbar = plt.colorbar(heatmap_im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Negative Log Probability', fontsize=7)
        cbar.ax.tick_params(labelsize=12)
    
    # Add title with professional styling
    ax.set_title(f'Latent Space Trajectory for Individual {trajectory_id}', fontsize=8, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save the plot with high resolution
    trajectory_path = os.path.join(output_dir, f'session_{trajectory_id}_DALE_latent_trajectory.pdf')
    plt.savefig(trajectory_path, format='pdf', dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    logger.info(f"Saved trajectory visualization to: {trajectory_path}")
    return trajectory_path, rmse

def create_rmse_histogram(rmse_values, session_ids, output_dir, dale_run_id):
    """
    Create a paper-quality histogram of RMSE values across all sessions.
    
    Args:
        rmse_values (list): List of RMSE values for each session
        session_ids (list): List of session IDs corresponding to RMSE values
        output_dir (str): Directory to save the plot
        dale_run_id (str): ID of the DALE run for the title
        
    Returns:
        str: Path to the saved histogram plot
    """
    logger = logging.getLogger(__name__)
    
    # Filter out None values (sessions without ground truth)
    valid_rmses = [(rmse, session_id) for rmse, session_id in zip(rmse_values, session_ids) if rmse is not None]
    
    if not valid_rmses:
        logger.warning("No valid RMSE values found for histogram")
        return None
    
    rmse_data = [rmse for rmse, _ in valid_rmses]
    valid_session_ids = [session_id for _, session_id in valid_rmses]
    
    # Calculate quartiles and identify key session IDs
    q1 = np.percentile(rmse_data, 25)
    q2 = np.percentile(rmse_data, 50)  # median
    q3 = np.percentile(rmse_data, 75)
    
    # Find session IDs for key statistics
    min_rmse_idx = np.argmin(rmse_data)
    max_rmse_idx = np.argmax(rmse_data)
    
    # Find closest to quartiles
    q1_idx = np.argmin(np.abs(np.array(rmse_data) - q1))
    q2_idx = np.argmin(np.abs(np.array(rmse_data) - q2))
    q3_idx = np.argmin(np.abs(np.array(rmse_data) - q3))
    
    # Get session IDs for key statistics
    min_session_id = valid_session_ids[min_rmse_idx]
    max_session_id = valid_session_ids[max_rmse_idx]
    q1_session_id = valid_session_ids[q1_idx]
    q2_session_id = valid_session_ids[q2_idx]
    q3_session_id = valid_session_ids[q3_idx]
    
    # Log the key session IDs
    logger.info("=" * 60)
    logger.info("KEY SESSION IDs BASED ON RMSE PERFORMANCE")
    logger.info("=" * 60)
    logger.info(f"Lowest RMSE Session: {min_session_id} (RMSE = {rmse_data[min_rmse_idx]:.4f})")
    logger.info(f"Highest RMSE Session: {max_session_id} (RMSE = {rmse_data[max_rmse_idx]:.4f})")
    logger.info(f"Q1 (25th percentile) Session: {q1_session_id} (RMSE = {rmse_data[q1_idx]:.4f})")
    logger.info(f"Q2 (50th percentile/median) Session: {q2_session_id} (RMSE = {rmse_data[q2_idx]:.4f})")
    logger.info(f"Q3 (75th percentile) Session: {q3_session_id} (RMSE = {rmse_data[q3_idx]:.4f})")
    logger.info("=" * 60)
    
    # Create paper-quality histogram
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create histogram with better styling
    n, bins, patches = ax.hist(rmse_data, bins=20, alpha=0.7, color='steelblue', 
                              edgecolor='black', linewidth=1.2, density=False)
    
    # Add statistics text box
    mean_rmse = np.mean(rmse_data)
    std_rmse = np.std(rmse_data)
    median_rmse = np.median(rmse_data)
    min_rmse = np.min(rmse_data)
    max_rmse = np.max(rmse_data)
    
    stats_text = f'Mean RMSE = {mean_rmse:.4f}\nStd RMSE = {std_rmse:.4f}\nMedian RMSE = {median_rmse:.4f}\nMin RMSE = {min_rmse:.4f}\nMax RMSE = {max_rmse:.4f}\nN = {len(rmse_data)}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=6,
           verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
           facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Paper-quality formatting
    ax.set_xlabel('RMSE', fontsize=8, fontweight='bold')
    ax.set_ylabel('Number of Sessions', fontsize=8, fontweight='bold')
    ax.set_title(f'Distribution of RMSE Values Across Sessions\nDALE Run: {dale_run_id}', 
                fontsize=10, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=7)
    
    # Add vertical lines for key statistics
    ax.axvline(mean_rmse, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Mean: {mean_rmse:.4f}')
    ax.axvline(median_rmse, color='orange', linestyle='--', linewidth=2, alpha=0.8, label=f'Median: {median_rmse:.4f}')
    # ax.axvline(q1, color='green', linestyle=':', linewidth=2, alpha=0.8, label=f'Q1: {q1:.4f}')
    # ax.axvline(q3, color='purple', linestyle=':', linewidth=2, alpha=0.8, label=f'Q3: {q3:.4f}')
    ax.legend(fontsize=7, framealpha=0.9)
    
    # Tight layout
    plt.tight_layout()
    
    # Save the plot
    histogram_path = os.path.join(output_dir, f'{dale_run_id}_RMSE_distribution.pdf')
    plt.savefig(histogram_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved RMSE histogram to: {histogram_path}")
    logger.info(f"RMSE Statistics - Mean: {mean_rmse:.4f}, Std: {std_rmse:.4f}, Median: {median_rmse:.4f}")
    
    # Return both the histogram path and the key session information
    key_sessions = {
        'min_session': min_session_id,
        'max_session': max_session_id,
        'q1_session': q1_session_id,
        'q2_session': q2_session_id,
        'q3_session': q3_session_id,
        'min_rmse': rmse_data[min_rmse_idx],
        'max_rmse': rmse_data[max_rmse_idx],
        'q1_rmse': rmse_data[q1_idx],
        'q2_rmse': rmse_data[q2_idx],
        'q3_rmse': rmse_data[q3_idx]
    }
    
    return histogram_path, key_sessions

def create_rmse_summary_table(rmse_values, session_ids, output_dir, dale_run_id):
    """
    Create a comprehensive summary table of RMSE values for all sessions.
    
    Args:
        rmse_values (list): List of RMSE values for each session
        session_ids (list): List of session IDs corresponding to RMSE values
        output_dir (str): Directory to save the table
        dale_run_id (str): ID of the DALE run for the filename
        
    Returns:
        str: Path to the saved summary table
    """
    logger = logging.getLogger(__name__)
    
    # Filter out None values (sessions without ground truth)
    valid_rmses = [(rmse, session_id) for rmse, session_id in zip(rmse_values, session_ids) if rmse is not None]
    
    if not valid_rmses:
        logger.warning("No valid RMSE values found for summary table")
        return None
    
    # Sort by RMSE (best performance first)
    valid_rmses.sort(key=lambda x: x[0])
    
    # Create summary table
    summary_file = os.path.join(output_dir, f'{dale_run_id}_RMSE_summary_table.txt')
    with open(summary_file, 'w') as f:
        f.write("COMPREHENSIVE RMSE SUMMARY TABLE\n")
        f.write("=" * 80 + "\n")
        f.write("Sessions ranked by RMSE performance (best to worst)\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Rank':<6} {'Session ID':<20} {'RMSE':<12} {'Percentile':<12}\n")
        f.write("-" * 80 + "\n")
        
        for rank, (rmse, session_id) in enumerate(valid_rmses, 1):
            percentile = (rank - 1) / len(valid_rmses) * 100
            f.write(f"{rank:<6} {session_id:<20} {rmse:<12.4f} {percentile:<12.1f}%\n")
        
        f.write("-" * 80 + "\n")
        
        # Add summary statistics
        rmse_data = [rmse for rmse, _ in valid_rmses]
        f.write(f"\nSUMMARY STATISTICS:\n")
        f.write(f"Total Sessions: {len(valid_rmses)}\n")
        f.write(f"Mean RMSE: {np.mean(rmse_data):.4f}\n")
        f.write(f"Std RMSE: {np.std(rmse_data):.4f}\n")
        f.write(f"Median RMSE: {np.median(rmse_data):.4f}\n")
        f.write(f"Min RMSE: {np.min(rmse_data):.4f}\n")
        f.write(f"Max RMSE: {np.max(rmse_data):.4f}\n")
        f.write(f"Q1 (25th percentile): {np.percentile(rmse_data, 25):.4f}\n")
        f.write(f"Q3 (75th percentile): {np.percentile(rmse_data, 75):.4f}\n")
    
    logger.info(f"Saved RMSE summary table to: {summary_file}")
    return summary_file

def load_ground_truth_latent_positions(meu_z_parameter_path):
    """
    Load meu_z parameters from file.
    
    Args:
        meu_z_parameter_path (str): Path to the file containing meu_z parameters
        
    Returns:
        torch.Tensor: Tensor of shape (n, 2) containing meu_z parameters
    """
    logger = logging.getLogger(__name__)
    
    latent_positions = torch.load(meu_z_parameter_path, weights_only=False)
    return latent_positions

def create_percentile_trajectories_plot(rmse_values, log_prob_values, session_ids, all_latent_positions, ground_truth_positions, 
                                      output_dir, dale_run_id, model, percentiles=[10, 50, 90], all_session_data=None, cmap='viridis'):
    """
    Create a 1x3 subplot showing trajectories for specific percentile sessions.
    Reuses the existing create_trajectory_visualization function.
    
    Args:
        rmse_values (list): List of RMSE values for each session
        log_prob_values (list): List of log probability values for each session
        session_ids (list): List of session IDs corresponding to RMSE values
        all_latent_positions (list): List of latent positions for each session
        ground_truth_positions (dict): Dictionary mapping session_id to ground truth position
        output_dir (str): Directory to save the plot
        dale_run_id (str): ID of the DALE run for the title
        model: DLVM model for consistent scaling
        percentiles (list): List of percentiles to plot (default: [10, 50, 90])
        all_session_data (dict): Dictionary containing all session data including loss landscapes
        cmap (str): Colormap for the plots
        
    Returns:
        str: Path to the saved percentile trajectories plot
    """
    logger = logging.getLogger(__name__)
    
    # Filter out None values (sessions without ground truth)
    valid_data = [(rmse, log_prob, session_id, latent_pos) for rmse, log_prob, session_id, latent_pos in 
                  zip(rmse_values, log_prob_values, session_ids, all_latent_positions) 
                  if rmse is not None and log_prob is not None]
    
    if not valid_data:
        logger.warning("No valid RMSE values found for percentile plot")
        return None
    
    # Sort by RMSE (best performance first)
    valid_data.sort(key=lambda x: x[0])
    rmse_data = [rmse for rmse, _, _, _ in valid_data]
    log_prob_data = [log_prob for _, log_prob, _, _ in valid_data]
    valid_session_ids = [session_id for _, _, session_id, _ in valid_data]
    valid_latent_positions = [latent_pos for _, _, _, latent_pos in valid_data]
    
    # Find sessions for each percentile
    percentile_sessions = {}
    for percentile in percentiles:
        if percentile == 0:
            idx = 0  # Best performing
        elif percentile == 100:
            idx = len(rmse_data) - 1  # Worst performing
        else:
            idx = int(len(rmse_data) * percentile / 100)
            idx = min(idx, len(rmse_data) - 1)  # Ensure we don't go out of bounds
        
        percentile_sessions[percentile] = {
            'session_id': valid_session_ids[idx],
            'rmse': rmse_data[idx],
            'log_prob': log_prob_data[idx],
            'latent_positions': valid_latent_positions[idx],
            'ground_truth': ground_truth_positions.get(valid_session_ids[idx])
        }
    
    # Create 1x3 subplot with larger figure size
    fig, axes = plt.subplots(1, len(percentiles), figsize=(8*len(percentiles), 8))
    if len(percentiles) == 1:
        axes = [axes]  # Make it iterable for single subplot
    
    # Reuse existing trajectory plotting function for each subplot
    for i, percentile in enumerate(percentiles):
        ax = axes[i]
        session_data = percentile_sessions[percentile]
        
        # Get loss landscape data for this session if available
        loss_landscape = None
        loss_meu_z = None
        if all_session_data and session_data['session_id'] in all_session_data:
            session_info = all_session_data[session_data['session_id']]
            loss_landscape = session_info.get('loss_landscape')
            loss_meu_z = session_info.get('loss_meu_z')
        
        # Use the reusable plotting function
        heatmap_im = plot_trajectory_on_axis(ax, session_data['latent_positions'], model, 
                                           session_data['ground_truth'], session_data['rmse'], session_data['log_prob'],
                                           f"{percentile}th_percentile_{session_data['session_id']}",
                                           loss_landscape=loss_landscape, loss_meu_z=loss_meu_z, cmap=cmap)
        
        # Add colorbar if loss landscape was plotted
        if heatmap_im is not None:
            cbar = plt.colorbar(heatmap_im, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('Negative Log Probability', fontsize=6)
            cbar.ax.tick_params(labelsize=10)
        
        # Update title to show percentile information with professional styling
        ax.set_title(f'{percentile}th Percentile\nIndividual: {session_data["session_id"]}\nRMSE: {session_data["rmse"]:.4f}', fontsize=8)

        # remove the grid
        ax.grid(False)
        # remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # remove the ticks
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        
    # Professional layout
    plt.tight_layout()
    
    # Save the plot with high resolution
    percentile_plot_path = os.path.join(output_dir, f'{dale_run_id}_percentile_trajectories.pdf')
    plt.savefig(percentile_plot_path, format='pdf', dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    logger.info(f"Saved percentile trajectories plot to: {percentile_plot_path}")
    
    # Log the selected sessions
    logger.info("=" * 60)
    logger.info("PERCENTILE TRAJECTORIES SELECTED")
    logger.info("=" * 60)
    for percentile in percentiles:
        session_data = percentile_sessions[percentile]
        logger.info(f"{percentile}th Percentile: Session {session_data['session_id']} (RMSE = {session_data['rmse']:.4f})")
    logger.info("=" * 60)
    
    return percentile_plot_path

def create_rmse_logprob_3x3_plot(rmse_values, log_prob_values, session_ids, all_latent_positions, ground_truth_positions, 
                                output_dir, dale_run_id, model, all_session_data=None, cmap='viridis', show_values_in_legend=False,
                                show_colorbar=False, show_axis_labels=False, fig_width=6.4, fig_height=None, 
                                base_fig_size=24.0, dpi=600, max_length_to_plot=240, show_reduced_xy_ticks=True):
    """
    Create a 3x3 subplot showing trajectories based on RMSE (x-axis) and Log Probability (y-axis) percentiles.
    Top-left corner shows worst performance (bad, bad).
    
    Args:
        rmse_values (list): List of RMSE values for each session
        log_prob_values (list): List of log probability values for each session
        session_ids (list): List of session IDs corresponding to the values
        all_latent_positions (list): List of latent positions for each session
        ground_truth_positions (dict): Dictionary mapping session_id to ground truth position
        output_dir (str): Directory to save the plot
        dale_run_id (str): ID of the DALE run for the title
        model: DLVM model for consistent scaling
        all_session_data (dict): Dictionary containing all session data including loss landscapes
        cmap (str): Colormap for the plots
        show_values_in_legend (bool): Whether to show values in legend
        show_colorbar (bool): Whether to show colorbar
        show_axis_labels (bool): Whether to show axis labels
        fig_width (float): Width of the figure in inches (default: 6.4)
        fig_height (float): Height of the figure in inches (default: None, uses fig_width for square)
        base_fig_size (float): Base figure size used for scaling calculations (default: 24.0)
        dpi (int): DPI for saving the plot (default: 600)
        max_length_to_plot (int): DEPRECATED - No longer used. Truncation is now done upstream (default: 240)
        show_reduced_xy_ticks (bool): If True, only show y-ticks on first column and x-ticks on last row (default: True)
        
    Returns:
        tuple: (str: Path to the saved 3x3 plot, list: Selected session IDs for the 3x3 plot)
    """
    logger = logging.getLogger(__name__)
    
    # Filter out None values (sessions without ground truth)
    valid_data = [(rmse, log_prob, session_id, latent_pos) for rmse, log_prob, session_id, latent_pos in 
                  zip(rmse_values, log_prob_values, session_ids, all_latent_positions) 
                  if rmse is not None and log_prob is not None]
    
    if not valid_data:
        logger.warning("No valid RMSE and log probability values found for 3x3 plot")
        return None, []
    
    # Sort by RMSE (best performance first - lowest RMSE is best)
    valid_data.sort(key=lambda x: x[0])
    
    # Calculate bucket sizes (10% of total)
    total_sessions = len(valid_data)
    bucket_size = max(1, total_sessions // 10)  # At least 1 session per bucket
    
    # Create RMSE buckets - INVERTED: 'low' gets worst RMSE (highest values), 'high' gets best RMSE (lowest values)
    rmse_buckets = {
        'low': valid_data[-bucket_size:],     # Worst RMSE (highest values) - now assigned to 'low'
        'middle': valid_data[total_sessions//2 - bucket_size//2:total_sessions//2 + bucket_size//2],  # Middle RMSE
        'high': valid_data[:bucket_size]      # Best RMSE (lowest values) - now assigned to 'high'
    }
    
    # Set figure height if not provided (default to square)
    if fig_height is None:
        fig_height = fig_width
    
    # Create 3x3 subplot with configurable dimensions
    fig, axes = plt.subplots(3, 3, figsize=(fig_width, fig_height))
    
    # Calculate scaling factors for different elements
    # Use more conservative scaling for text to maintain readability
    text_scale_factor = max(0.6, fig_width / base_fig_size)  # Minimum text scale of 0.6
    symbol_scale_factor = fig_width / base_fig_size  # Symbols can scale more aggressively
    
    # Create the 3x3 grid
    rmse_bucket_names = ['low', 'middle', 'high']
    log_prob_bucket_names = ['low', 'middle', 'high']
    
    # Store the first heatmap for global colorbar
    first_heatmap_im = None
    selected_session_ids = []  # Track the 9 selected session IDs

    
    for i, rmse_bucket in enumerate(rmse_bucket_names):
        for j, log_prob_bucket in enumerate(log_prob_bucket_names):
            ax = axes[j, i]  # Note: j is row (log prob), i is column (RMSE)
            
            # Get the RMSE bucket data
            bucket_data = rmse_buckets[rmse_bucket]
            
            # Sort this bucket by log probability (ascending - lower is better)
            bucket_data_sorted = sorted(bucket_data, key=lambda x: x[1])
            
            # Select the appropriate log prob performance within this bucket - INVERTED
            if log_prob_bucket == 'low':
                # Worst log prob (highest value) - now assigned to 'low'
                selected_session = bucket_data_sorted[-1]
            elif log_prob_bucket == 'middle':
                # Middle log prob
                middle_idx = len(bucket_data_sorted) // 2
                selected_session = bucket_data_sorted[middle_idx]
            else:  # log_prob_bucket == 'high'
                # Best log prob (lowest value) - now assigned to 'high'
                selected_session = bucket_data_sorted[0]
            
            # Extract session data
            session_rmse, session_log_prob, session_id, session_latent_positions = selected_session
            selected_session_ids.append(session_id)  # Add to list of selected sessions
            session_ground_truth = ground_truth_positions.get(session_id)
            
            # Get loss landscape data for this session if available
            loss_landscape = None
            loss_meu_z = None
            if all_session_data and session_id in all_session_data:
                session_info = all_session_data[session_id]
                loss_landscape = session_info.get('loss_landscape')
                loss_meu_z = session_info.get('loss_meu_z')
            
            # Use the reusable plotting function with scaling factors (data already truncated upstream)
            heatmap_im = plot_trajectory_on_axis(ax, session_latent_positions, model, 
                                               session_ground_truth, session_rmse, session_log_prob,
                                               f"RMSE_{rmse_bucket}_LogProb_{log_prob_bucket}_{session_id}",
                                               loss_landscape=loss_landscape, loss_meu_z=loss_meu_z, cmap=cmap, 
                                               show_values_in_legend=show_values_in_legend, drop_odd_number_annotations=True, 
                                               show_axis_labels=False, scale_factor=symbol_scale_factor, 
                                               text_scale_factor=text_scale_factor, 
                                               show_legend=True if rmse_bucket == 'high' and log_prob_bucket == 'high' else False)
            
            # Store the first heatmap for global colorbar (don't add individual colorbars)
            if heatmap_im is not None and first_heatmap_im is None:
                first_heatmap_im = heatmap_im
            
            # Update title to show bucket information - scale title text
            ax.set_title(f'ID: {session_id}\nRMSE: {session_rmse:.3f}, Log Prob: {session_log_prob:.3f}', 
                        fontsize=10*text_scale_factor)
            print("ID:", session_id, "RMSE:", session_rmse, "Log Prob:", session_log_prob)
            
            # Conditionally show axis tick labels based on position in grid
            if show_reduced_xy_ticks:
                # Only show y-axis labels for first column (i == 0)
                # Only show x-axis labels for last row (j == 2)
                show_y_labels = (i == 0)
                show_x_labels = (j == 2)
            else:
                # Show all labels if not reducing ticks
                show_y_labels = True
                show_x_labels = True
            
            if show_axis_labels:
                # set font size for the y-tick labels
                ax.tick_params(axis='y', labelsize=7*text_scale_factor)
                # set font size for the x-tick labels
                ax.tick_params(axis='x', labelsize=7*text_scale_factor)
            
            # remove splines 
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            # Configure ticks - conditionally show both tick marks and labels
            ax.tick_params(axis='both', which='both', bottom=show_x_labels, top=False, left=show_y_labels, 
                           right=False, labelbottom=show_x_labels, labeltop=False, 
                           labelleft=show_y_labels, labelright=False)
            # reduce spline width and make them less dominant
            ax.spines['bottom'].set_linewidth(0.5)
            ax.spines['left'].set_linewidth(0.5)
            # ax.spines['bottom'].set_edgecolor('gray')
            # ax.spines['left'].set_edgecolor('gray')
            ax.spines['bottom'].set_alpha(0.7)
            ax.spines['left'].set_alpha(0.7)
            ax.spines['top'].set_alpha(0.7)
            ax.spines['right'].set_alpha(0.7)

            # Make tick labels and ticks less dominant - match tick width to spline width
            ax.tick_params(axis='x', labelsize=10*text_scale_factor, width=0.5)
            ax.tick_params(axis='y', labelsize=10*text_scale_factor, width=0.5)
            
            # Set tick label alpha by directly modifying the label objects
            for label in ax.get_xticklabels():
                label.set_alpha(0.7)
            for label in ax.get_yticklabels():
                label.set_alpha(0.7)
    
    # Add global colorbar if any heatmap was plotted and show_colorbar is True
    if first_heatmap_im is not None and show_colorbar:
        # Create a horizontal global colorbar positioned at the top of the figure
        cbar = fig.colorbar(first_heatmap_im, ax=axes.ravel().tolist(), 
                           orientation='horizontal', shrink=0.6, pad=0.1,
                           ticks=[0.0, 0.02, 0.04, 0.06, 0.08, 0.1])
        
        # Set custom tick labels
        cbar.ax.set_xticklabels(['0.0', '0.02', '0.04', '0.06', '0.08', '≥0.1'])
        
        # cbar.set_label('Negative Log Probability', fontsize=10*text_scale_factor, labelpad=10)
        # position the colorbar labels at the top and text above the colorbar
        cbar.ax.tick_params(labeltop=False, labelbottom=True, labelsize=10*text_scale_factor)
        cbar.ax.text(0.5, 1.2, 'Negative Log Probability', ha='center', fontsize=12*text_scale_factor, transform=cbar.ax.transAxes)
        
        # Position the colorbar at the top, just after the heading
        cbar.ax.set_position([0.15, 0.97, 0.7, 0.02])  # [left, bottom, width, height]
    
    # Add overall title and labels with professional styling - scale text
    fig.suptitle(f'DALE Trajectories of 9 Representative Sessions after {max_length_to_plot} Observations', 
                 fontsize=18*text_scale_factor, y=1.04)  # Larger title
    
    # Add row and column labels with professional styling - scale text
    fig.text(0.5, 0.01, 'RMSE (High → Low)', ha='center', fontsize=14*text_scale_factor)
    fig.text(0.02, 0.3, 'Negative Log Probability (Low → High)', ha='center', rotation=90, fontsize=14*text_scale_factor)
    
    # Professional layout and spacing - better use of space
    plt.tight_layout()
    if show_colorbar:
        # With colorbar: less space for subplots
        plt.subplots_adjust(top=0.90, bottom=0.06, left=0.02, right=0.98, hspace=0.3, wspace=0.01)
    else:
        # Without colorbar: more space for subplots and better spacing i.e left and right padding is 0.08
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.01, right=0.99, hspace=0.0, wspace=0.0001)  # Reduced wspace from 0.05 to 0.02
    
    # Save the plot with high resolution
    plot_path = os.path.join(output_dir, f'{dale_run_id}_3x3_RMSE_LogProb_colorbar_{show_colorbar}_axis_labels_{show_axis_labels}.pdf')
    # also save a png file
    plot_path_png = os.path.join(output_dir, f'{dale_run_id}_3x3_RMSE_LogProb_colorbar_{show_colorbar}_axis_labels_{show_axis_labels}.png')
    plt.savefig(plot_path, format='pdf', dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig(plot_path_png, format='png', dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Saved 3x3 RMSE vs Log Probability plot to: {plot_path}")
    logger.info(f"Saved 3x3 RMSE vs Log Probability plot to: {plot_path}")
    
    # Log the selected sessions
    logger.info("=" * 80)
    logger.info("3x3 RMSE vs LOG PROBABILITY TRAJECTORIES SELECTED")
    logger.info("=" * 80)
    for i, rmse_bucket in enumerate(rmse_bucket_names):
        for j, log_prob_bucket in enumerate(log_prob_bucket_names):
            ax = axes[j, i]
            title = ax.get_title()
            logger.info(f"Position ({rmse_bucket} RMSE, {log_prob_bucket} Log Prob): {title}")
    logger.info("=" * 80)
    
    return plot_path, selected_session_ids

def create_rmse_logprob_scatter_plot(rmse_values, log_prob_values, session_ids, output_dir, dale_run_id, 
                                   highlight_sessions=None, cmap='viridis', figsize=(3.2, 2.4), dpi=600, add_stats_text=False):
    """
    Create a high-resolution scatter plot of RMSE vs Log Probability.
    
    Args:
        rmse_values (list): List of RMSE values for each session
        log_prob_values (list): List of log probability values for each session
        session_ids (list): List of session IDs corresponding to the values
        output_dir (str): Directory to save the plot
        dale_run_id (str): ID of the DALE run for the title
        highlight_sessions (list): Optional list of session IDs to highlight
        cmap (str): Colormap for the scatter plot
        figsize (tuple): Figure size (width, height)
        dpi (int): DPI for high resolution
        add_stats_text (bool): If True, add statistics text to the plot
    Returns:
        str: Path to the saved scatter plot
    """
    logger = logging.getLogger(__name__)
    
    # Filter out None values (sessions without ground truth)
    valid_data = [(rmse, log_prob, session_id) for rmse, log_prob, session_id in 
                  zip(rmse_values, log_prob_values, session_ids) 
                  if rmse is not None and log_prob is not None]
    
    if not valid_data:
        logger.warning("No valid RMSE and log probability values found for scatter plot")
        return None
    
    # Extract data
    rmse_data = [rmse for rmse, _, _ in valid_data]
    log_prob_data = [log_prob for _, log_prob, _ in valid_data]
    valid_session_ids = [session_id for _, _, session_id in valid_data]
    
    # Create high-resolution figure
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Create scatter plot with professional styling
    scatter = ax.scatter(rmse_data, log_prob_data, s=50, alpha=0.6, linewidth=1, marker='x')
    
    # Highlight specific sessions if provided
    if highlight_sessions:
        highlight_indices = [i for i, sid in enumerate(valid_session_ids) if sid in highlight_sessions]
        if highlight_indices:
            highlight_rmse = [rmse_data[i] for i in highlight_indices]
            highlight_log_prob = [log_prob_data[i] for i in highlight_indices]
            # Plot each highlighted point individually to add session ID annotation
            for i, (rmse, log_prob, idx) in enumerate(zip(highlight_rmse, highlight_log_prob, highlight_indices)):
                ax.scatter(rmse, log_prob, s=50, alpha=0.9,
                          c='red', linewidth=2, marker='x',
                          label=f'Representative Sessions (n={len(highlight_indices)})' if i==0 else "", zorder=5)
                ax.annotate(f'{valid_session_ids[idx]}', 
                          (rmse, log_prob),
                          xytext=(0, 5), textcoords='offset points',
                          fontsize=4, color='red', ha='center', va='bottom')
            logger.info(f"Highlighting {len(highlight_indices)} sessions in scatter plot")
    
    
    # Professional axis labels and styling
    ax.set_xlabel('Root Mean Squared Error (RMSE)', fontsize=8  )
    ax.set_ylabel('Negative Log Probability', fontsize=8)
    ax.set_title(f'RMSE and Log Probability of DALE Fits', 
                fontsize=10, pad=10)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=7)
    
    # Add statistics text box with correlation
    correlation = np.corrcoef(rmse_data, log_prob_data)[0, 1]
    if add_stats_text:
        stats_text = f'N = {len(valid_data)}\nMean RMSE: {np.mean(rmse_data):.4f}\nMean Log Prob: {np.mean(log_prob_data):.4f}\nr = {correlation:.3f}'

    
        # Combine stats and legend text
        combined_text = stats_text
        
        ax.text(0.98, 0.98, combined_text, transform=ax.transAxes,
            fontsize=6, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
        
    # plot the legend in the middle right
    # ax.legend(fontsize=6, framealpha=0.9, loc='center right')
    
    # Professional layout
    plt.tight_layout()
    
    # Save with high resolution
    scatter_plot_path = os.path.join(output_dir, f'{dale_run_id}_RMSE_vs_LogProb_scatter.pdf')
    plt.savefig(scatter_plot_path, format='pdf', dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    # also save a png file
    scatter_plot_path_png = os.path.join(output_dir, f'{dale_run_id}_RMSE_vs_LogProb_scatter.png')
    plt.savefig(scatter_plot_path_png, format='png', dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved RMSE vs Log Probability scatter plot to: {scatter_plot_path}")
    logger.info(f"Saved RMSE vs Log Probability scatter plot to: {scatter_plot_path}")
    logger.info(f"Correlation coefficient: {correlation:.3f}")
    
    return scatter_plot_path


def main():
    """Main function to orchestrate the trajectory visualization."""
    parser = argparse.ArgumentParser(description='Visualize latent space trajectories from 2D DLVM model')
    parser.add_argument('--model-path', type=str, default=None, help='Path to the DLVM model file')
    parser.add_argument("--model-id", type=str, default="mongoose-dive-7464", help="Model ID to load")
    parser.add_argument('--latent-positions', type=str, required=False, 
                       help='Path to file containing nx2 matrix of latent positions')
    parser.add_argument('--output-dir', type=str, default='trajectory_visualization',
                       help='Output directory for results')
    parser.add_argument('--trajectory-id', type=str, default='trajectory',
                       help='Identifier for the trajectory')
    parser.add_argument('--show-parameters', action='store_true',
                       help='Show parameter analysis along trajectory')
    parser.add_argument('--sample-points', type=int, nargs='+', default=None,
                       help='Specific point indices to show in marginal fits (0-indexed)')
    parser.add_argument("--performance-tracking-basepath", type=str, default="../../results/dale_runs/",
                       help='Path to base directory containing performance tracking data')
    parser.add_argument("--dale_run_id", type=str, default="yeti-thrive-1690",
                       help='ID of the DALE run to visualize')
    parser.add_argument("--session_id", type=str, default=None,
                       help='ID of the session to visualize. If not specified, all sessions in the dale_run_id folder will be processed.')
    parser.add_argument('--latent-dim', type=int, default=2,
                       help='Latent dimension of the model (default: 2)')
    parser.add_argument("--plot_extra_plots", action="store_true", default=False,
                       help="If extra plots should be plot")
    parser.add_argument("--verbose", action="store_true", default=False,
                       help="If verbose output should be printed")
    parser.add_argument("--max_length_to_plot", type=int, default=240,
                       help="Maximum length of the trajectory to plot")
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(verbose=args.verbose )
    logger.info("Starting trajectory visualization")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Latent positions: {args.latent_positions}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Trajectory ID: {args.trajectory_id}")
    
    # Create output directory named after dale_run_id with max_length_to_plot subfolder
    output_dir = os.path.join(args.output_dir, args.dale_run_id, f"DALE_max_{args.max_length_to_plot}_points")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")

    if args.latent_dim ==2:
        meu_z_parameter_path = os.path.join("../create_synthetic_individuals/synthetic_sessions_ground_truth/simulated_data/D2_synthetic_latent_variables_mongoose-dive-7464.pt")
        args.model_id = "mongoose-dive-7464"
        args.model_path = os.path.join(script_dir, f"../../saved_models/{DATASET}/heldout_obsmulti/variationalNN_relevant_only_latentdim{args.latent_dim}_{args.model_id}.pt")
        args.ground_truth_data_path = os.path.join("../generate_synthetic_item_observations/synthetic_data/COLL10_SIM/D2_synthetic_ground_truth_parameters_mongoose-dive-7464/all_synthetic_data_N240.pt")
    elif args.latent_dim == 3:
        meu_z_parameter_path = os.path.join("../create_synthetic_individuals/synthetic_sessions_ground_truth/simulated_data/D3_synthetic_ground_truth_parameters_beaver-slide-5310.pt")
        args.model_id = "beaver-slide-5310"
        args.model_path = os.path.join(script_dir, f"../../saved_models/{DATASET}/heldout_obsmulti/variationalNN_relevant_only_latentdim{args.latent_dim}_{args.model_id}.pt")
        args.ground_truth_data_path = os.path.join("../generate_synthetic_item_observations/synthetic_data/COLL10_SIM/D3_synthetic_ground_truth_parameters_beaver-slide-5310/all_synthetic_data_N240.pt")
    elif args.latent_dim == 4:
        meu_z_parameter_path = ""
        
    
    try:
        # Step 1: Load the model (if needed for parameter analysis)
        model = None
        logger.info("Loading trained model...")
        if args.model_id is not None: 
            args.model_path = os.path.join(script_dir, f"../../saved_models/{DATASET}/heldout_obsmulti/variationalNN_relevant_only_latentdim{args.latent_dim}_{args.model_id}.pt")
        
        # if no session_id is passed then identify all the session_ids in dale_run_id folder. each session_id is the folder name in the dale_run_id folder.
        if args.session_id is None:
            dale_run_path = os.path.join(args.performance_tracking_basepath, args.dale_run_id)
            session_ids = [f for f in os.listdir(dale_run_path) if os.path.isdir(os.path.join(dale_run_path, f))]
            logger.info(f"Found {len(session_ids)} session IDs: {session_ids}")
        else:
            session_ids = [args.session_id]
        
        if args.model_path is None or not os.path.exists(args.model_path):
            logger.error("DLVM model path is not specified or does not exist. Use --model-path to specify the model file.")
            sys.exit(1)
        model = load_trained_model(args.latent_dim, model_path=args.model_path)
        logger.info("Model loaded successfully")

        ground_truth_latent_positions = load_ground_truth_latent_positions(meu_z_parameter_path)
        new_ground_truth_latent_positions = {}
        logger.info("Ground truth latent positions loaded successfully")
        logger.info(f"Ground truth keys: {list(ground_truth_latent_positions.keys())}")

        # load ground truth model and data
        ground_truth_model = load_trained_model(args.latent_dim, model_path=args.model_path)
        ground_truth_data = torch.load(args.ground_truth_data_path, weights_only=False)
        logger.info("Ground truth model and data loaded successfully")

        # Process each session
        # Using dictionary structure to guarantee data alignment by session_id key
        # This eliminates any possibility of misalignment between RMSE values, 
        # latent positions, and session IDs that could occur with separate lists
        all_trajectory_paths = []
        session_data = {}  # Dictionary to store all session data by session_id
        ground_truth_positions = {}  # Store ground truth positions for percentile plot
        for session_id in session_ids:
            logger.info(f"Processing session: {session_id}")
            
            # Step 2: Load latent positions for this session
            logger.info(f"Loading latent positions for session {session_id}...")
            positions_path = os.path.join(args.performance_tracking_basepath, args.dale_run_id, session_id, f"analysis/performance_tracking_session_{session_id}.csv")
            
            if not os.path.exists(positions_path):
                logger.warning(f"Positions file not found for session {session_id}: {positions_path}")
                continue
                
            latent_positions = load_latent_positions(positions_path)
            
            # Truncate latent positions based on max_length_to_plot
            if args.max_length_to_plot is not None and len(latent_positions) > args.max_length_to_plot:
                logger.info(f"Truncating latent positions from {len(latent_positions)} to {args.max_length_to_plot}")
                latent_positions = latent_positions[:args.max_length_to_plot]
            
            # Get ground truth position for this session
            ground_truth_position = None
            if session_id in ground_truth_latent_positions:
                ground_truth_position = ground_truth_latent_positions[session_id]
                logger.info(f"Found ground truth for session {session_id}, shape: {ground_truth_position.shape}")
            else:
                logger.warning(f"No ground truth found for session {session_id}")
                logger.debug(f"Available ground truth keys: {list(ground_truth_latent_positions.keys())}")
            
            # Store ground truth position for percentile plot
            if ground_truth_position is not None:
                ground_truth_positions[session_id] = ground_truth_position
            

            # extract session data from ground truth data
            individual_session_data = ground_truth_data[f"{session_id}_sim1"]
            logger.info(f"Extracted session data for session {session_id}")

            # create a custom grid in the range of -5 to 5 in both dimensions with 200 points in each dimension
            x = np.linspace(-6, 6, 200)
            y = np.linspace(-6, 6, 200)
            xx, yy = np.meshgrid(x, y)
            custom_grid = torch.tensor(np.stack([xx.flatten(), yy.flatten()], axis=1), dtype=torch.float32)
            # also include the final latent position in the custom grid
            custom_grid = torch.cat([custom_grid, latent_positions], dim=0)

            # compute the log prob landscape for the session - do not randomize the order of the points to ensure the final latent position is at the end of the grid
            loss, meu_z = compute_loss_landscape(ground_truth_model, individual_session_data, 200, custom_grid, randomize_order=False)

            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            loss_np = scaler.fit_transform(loss.reshape(-1, 1)).flatten()
            # set loss value above 0.1 to 0.1
            loss_np = np.where(loss_np > 0.1, 0.1, loss_np)
            loss = torch.tensor(loss_np, dtype=torch.float32)

            # extract the loss value for the final latent position
            final_loss = loss[-1].item()  # Convert tensor to float
            final_meu_z = meu_z[-1]
            logger.info(f"Log_Prob at final latent position: {final_loss} at {final_meu_z}")
            
            # Filter to keep only top 10% of points with lowest loss values (best performance)
            # top_10_percent_idx = torch.topk(loss, k=int(0.1 * len(loss)), largest=False).indices
            # loss = loss[top_10_percent_idx]
            # meu_z = meu_z[top_10_percent_idx]
            
            
            logger.info(f"Computed log prob landscape for session {session_id}")
            trajectory_path = None
            if args.plot_extra_plots:
                # Step 3: Create trajectory visualization for this session
                logger.info(f"Creating trajectory visualization for session {session_id}...")
                trajectory_path, rmse = create_trajectory_visualization(
                    latent_positions, output_dir, model, args.show_parameters, session_id, 
                    ground_truth_position=ground_truth_position,
                    loss_landscape=loss, loss_meu_z=meu_z,
                    loss_at_final_latent_position=final_loss,
                    cmap = "gray"
                )
                all_trajectory_paths.append(trajectory_path)
            else:
                # just compute the rmse
                
                rmse = compute_position_rmse(final_meu_z, ground_truth_position)
                logger.info(f"Computed RMSE for session {session_id}: {rmse}")
                
            # Store all session data in dictionary - guarantees alignment by key
            session_data[session_id] = {
                'rmse': rmse,
                'latent_positions': latent_positions,
                'trajectory_path': trajectory_path,
                'ground_truth': ground_truth_position,
                'loss_landscape': loss,
                'loss_meu_z': meu_z,
                'log_prob': final_loss  # Store the log probability at final position (already converted to float)
            }
        
        # Log data collection summary
        logger.info(f"Successfully processed {len(session_data)} sessions")
        logger.info(f"Sessions with data: {sorted(list(session_data.keys()))}")
        valid_rmse_count = sum(1 for data in session_data.values() if data['rmse'] is not None)
        logger.info(f"Sessions with valid RMSE: {valid_rmse_count}")
        
        # Data integrity verification (guaranteed by dictionary structure)
        sessions_with_rmse = [sid for sid, data in session_data.items() if data['rmse'] is not None]
        sessions_with_ground_truth = [sid for sid, data in session_data.items() if data['ground_truth'] is not None]
        logger.info(f"Sessions with RMSE values: {sorted(sessions_with_rmse)}")
        logger.info(f"Sessions with ground truth: {sorted(sessions_with_ground_truth)}")
        logger.info("Dictionary structure guarantees perfect data alignment - no misalignment possible!")
        
        # Step 4: Combine all PDFs into a single file
        if len(all_trajectory_paths) > 1:
            logger.info("Combining all PDFs into a single file...")
            combined_pdf_path = os.path.join(output_dir, f"{args.dale_run_id}_all_sessions_trajectories.pdf")
            combine_pdfs_in_folder(output_dir, combined_pdf_path)
            logger.info(f"Created combined PDF: {combined_pdf_path}")
            
            # Delete individual PDFs after combining
            logger.info("Cleaning up individual PDFs...")
            deleted_count = 0
            for trajectory_path in all_trajectory_paths:
                try:
                    os.remove(trajectory_path)
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {trajectory_path}: {e}")
            logger.info(f"Deleted {deleted_count} individual PDFs")
            
        elif len(all_trajectory_paths) == 1:
            logger.info("Only one session processed, no need to combine PDFs")
        else:
            logger.warning("No PDFs were created")

        # Step 5: Create RMSE histogram
        key_sessions_info = None
        if session_data:
            # Extract aligned data from dictionary
            session_ids_with_data = list(session_data.keys())
            rmse_values_with_data = [session_data[sid]['rmse'] for sid in session_ids_with_data]
            if args.plot_extra_plots:
                histogram_path, key_sessions_info = create_rmse_histogram(rmse_values_with_data, session_ids_with_data, output_dir, args.dale_run_id)
            
                # Save key session information to a text file
                if key_sessions_info:
                    key_sessions_file = os.path.join(output_dir, f'{args.dale_run_id}_key_sessions_RMSE.txt')
                    with open(key_sessions_file, 'w') as f:
                        f.write("KEY SESSION IDs BASED ON RMSE PERFORMANCE\n")
                        f.write("=" * 60 + "\n")
                        f.write(f"Lowest RMSE Session: {key_sessions_info['min_session']} (RMSE = {key_sessions_info['min_rmse']:.4f})\n")
                        f.write(f"Highest RMSE Session: {key_sessions_info['max_session']} (RMSE = {key_sessions_info['max_rmse']:.4f})\n")
                        f.write(f"Q1 (25th percentile) Session: {key_sessions_info['q1_session']} (RMSE = {key_sessions_info['q1_rmse']:.4f})\n")
                        f.write(f"Q2 (50th percentile/median) Session: {key_sessions_info['q2_session']} (RMSE = {key_sessions_info['q2_rmse']:.4f})\n")
                        f.write(f"Q3 (75th percentile) Session: {key_sessions_info['q3_session']} (RMSE = {key_sessions_info['q3_rmse']:.4f})\n")
                        f.write("=" * 60 + "\n")
                    logger.info(f"Saved key session information to: {key_sessions_file}")
        else:
            logger.warning("No session data available to create histogram.")
        
        # Step 6: Create RMSE summary table
        if session_data and args.plot_extra_plots:
            create_rmse_summary_table(rmse_values_with_data, session_ids_with_data, output_dir, args.dale_run_id)
        else:
            logger.warning("No session data available to create summary table.")
        
        # Step 7: Create percentile trajectories plot
        if session_data and args.plot_extra_plots:
            # Extract aligned data from dictionary
            latent_positions_with_data = [session_data[sid]['latent_positions'] for sid in session_ids_with_data]
            log_prob_values_with_data = [session_data[sid]['log_prob'] for sid in session_ids_with_data]
            create_percentile_trajectories_plot(rmse_values_with_data, log_prob_values_with_data, session_ids_with_data, latent_positions_with_data, 
                                             ground_truth_positions, output_dir, args.dale_run_id, model, all_session_data=session_data,
                                             cmap = "gray")
        else:
            logger.warning("No session data available for percentile trajectories plot.")
        
        # Step 8: Create 3x3 RMSE vs Log Probability plot
        selected_3x3_session_ids = []  # Track selected session IDs from 3x3 plot
        if session_data:
            # Extract aligned data from dictionary
            latent_positions_with_data = [session_data[sid]['latent_positions'] for sid in session_ids_with_data]
            log_prob_values_with_data = [session_data[sid]['log_prob'] for sid in session_ids_with_data]
            # show with colorbar and axis labels
            plot_path, selected_3x3_session_data = create_rmse_logprob_3x3_plot(rmse_values_with_data, log_prob_values_with_data, session_ids_with_data, 
                                       latent_positions_with_data, ground_truth_positions, output_dir, args.dale_run_id, 
                                       model, all_session_data=session_data, cmap="gray", show_values_in_legend=False, 
                                       show_colorbar=True, show_axis_labels=True, fig_width=6.4, dpi=600,
                                       max_length_to_plot=args.max_length_to_plot,
                                       show_reduced_xy_ticks=True)
            if args.plot_extra_plots:
            # # show without colorbar and axis labels
                plot_path_no_labels, _ = create_rmse_logprob_3x3_plot(rmse_values_with_data, log_prob_values_with_data, session_ids_with_data, 
                                        latent_positions_with_data, ground_truth_positions, output_dir, args.dale_run_id, 
                                        model, all_session_data=session_data, cmap="gray", show_values_in_legend=False, 
                                        show_colorbar=False, show_axis_labels=False, fig_width=6.4, dpi=600,
                                        show_reduced_xy_ticks=True,
                                        max_length_to_plot=args.max_length_to_plot)
            else:
                logger.info("Skipping 3x3 RMSE vs Log Probability plot")
        else:
            logger.warning("No session data available for 3x3 RMSE vs Log Probability plot.")
        
        # Step 9: Create RMSE vs Log Probability scatter plot
        if session_data:
            # Extract aligned data from dictionary
            latent_positions_with_data = [session_data[sid]['latent_positions'] for sid in session_ids_with_data]
            log_prob_values_with_data = [session_data[sid]['log_prob'] for sid in session_ids_with_data]
            
            # Create scatter plot with highlighted sessions from 3x3 plot
            scatter_plot_path = create_rmse_logprob_scatter_plot(
                rmse_values_with_data, log_prob_values_with_data, session_ids_with_data, 
                output_dir, args.dale_run_id, highlight_sessions=selected_3x3_session_data, 
                cmap='viridis', figsize=(3.2, 3.2), dpi=600
            )
            
            if scatter_plot_path:
                logger.info(f"RMSE vs Log Probability scatter plot: {scatter_plot_path}")
        else:
            logger.warning("No session data available for RMSE vs Log Probability scatter plot.")
        
        # Summary
        logger.info("=" * 50)
        logger.info("TRAJECTORY VISUALIZATION COMPLETED")
        logger.info("=" * 50)
        logger.info(f"Found {len(session_ids)} total sessions, successfully processed {len(session_data)} sessions")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Files created:")
        if len(all_trajectory_paths) > 1:
            logger.info(f"  - Combined PDF: {os.path.join(output_dir, f'{args.dale_run_id}_all_sessions_trajectories.pdf')}")
            logger.info(f"  - Individual PDFs: {len(all_trajectory_paths)} (deleted after combining)")
        else:
            for trajectory_path in all_trajectory_paths:
                logger.info(f"  - Trajectory visualization: {trajectory_path}")
        
        # RMSE statistics
        valid_rmses = [data['rmse'] for data in session_data.values() if data['rmse'] is not None]
        if valid_rmses:
            logger.info(f"RMSE Statistics:")
            logger.info(f"  - Mean RMSE: {np.mean(valid_rmses):.4f}")
            logger.info(f"  - Std RMSE: {np.std(valid_rmses):.4f}")
            logger.info(f"  - Median RMSE: {np.median(valid_rmses):.4f}")
            logger.info(f"  - Min RMSE: {np.min(valid_rmses):.4f}")
            logger.info(f"  - Max RMSE: {np.max(valid_rmses):.4f}")
            logger.info(f"  - RMSE Histogram: {os.path.join(output_dir, f'{args.dale_run_id}_RMSE_distribution.pdf')}")
            if key_sessions_info:
                logger.info(f"  - Key Sessions File: {os.path.join(output_dir, f'{args.dale_run_id}_key_sessions_RMSE.txt')}")
            if session_data:
                logger.info(f"  - RMSE Summary Table: {os.path.join(output_dir, f'{args.dale_run_id}_RMSE_summary_table.txt')}")
                logger.info(f"  - Percentile Trajectories Plot: {os.path.join(output_dir, f'{args.dale_run_id}_percentile_trajectories.pdf')}")
        else:
            logger.info("No valid RMSE values computed")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Error during trajectory visualization: {e}")
        raise


if __name__ == "__main__":
    main()
