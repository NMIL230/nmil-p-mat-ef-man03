# Ground Truth Sessions Creation

This directory contains scripts to create ground truth sessions by sampling latent points from a 2D DLVM model and computing their marginal fits.

## Overview

The `create_ground_truth_sessions.py` script:

1. **Loads a specified 2D DLVM model**
2. **Extracts meu_z parameters** to determine latent space bounds
3. **Uniformly samples 100 points** within those bounds
4. **Computes marginal fits** for each point using the model
5. **Creates visualizations**:
   - 2D scatter plot of latent points
   - Individual marginal fit plots for each point
   - Combined PDF of all marginal plots
6. **Saves parameter dictionaries** with systematic IDs (A001, A002, etc.)

## Files

- `create_ground_truth_sessions.py` - Main script for creating ground truth sessions
- `test_ground_truth_sessions.py` - Test script to verify functionality
- `README_ground_truth_sessions.md` - This documentation file

## Usage

### Basic Usage

```bash
python create_ground_truth_sessions.py --model_path path/to/model.pt --output_dir output/
```

### Full Command Line Options

```bash
python create_ground_truth_sessions.py \
    --model_path path/to/model.pt \
    --output_dir ground_truth_output \
    --num_points 100 \
    --latent_dim 2
```

### Parameters

- `--model_path` (required): Path to the trained 2D model file
- `--output_dir` (optional): Output directory for results (default: `ground_truth_output`)
- `--num_points` (optional): Number of points to sample (default: 100)
- `--latent_dim` (optional): Latent dimension of the model (default: 2)

### Example

```bash
# Using a model from the artifacts/models directory
python create_ground_truth_sessions.py \
    --model_path ../../artifacts/models/COLL10/heldout_obs/variationalNN_relevant_only_latentdim2_ablaze-sweetheart-426.pt \
    --output_dir my_ground_truth_sessions \
    --num_points 50
```

## Output Structure

The script creates the following directory structure:

```
output_dir/
├── latent_space_scatter.pdf          # 2D scatter plot of sampled points
├── combined_marginals.pdf            # Combined PDF of all marginal plots
├── create_ground_truth_sessions.log  # Log file
├── simulated_data/                   # Parameter data directory
│   ├── A001_params.pt               # Parameter dict for point A001
│   ├── A001_latent_coords.pt        # Latent coordinates for point A001
│   ├── A002_params.pt               # Parameter dict for point A002
│   ├── A002_latent_coords.pt        # Latent coordinates for point A002
│   └── ...
│   └── all_simulated_data.pt        # Summary file with all data
└── marginal/                        # Individual marginal plots
    ├── A001_marginal.pdf            # Marginal fits for point A001
    ├── A002_marginal.pdf            # Marginal fits for point A002
    └── ...
```

## Point ID System

Each sampled point is assigned a systematic ID:
- **A001, A002, A003, ..., A100** for 100 points
- IDs are used consistently across all files
- Format: `A` + 3-digit zero-padded number

## Testing

Run the test script to verify everything works:

```bash
python test_ground_truth_sessions.py
```

The test script will:
1. Find a suitable 2D model in the `artifacts/models/` directory
2. Run the main script with 5 points (for speed)
3. Verify that all expected files are created
4. Report success or failure

## Dependencies

The script uses existing utility functions from the codebase:

- `load_trained_model` from `utils/data_distribution_utils.py`
- `generate_grid` from `utils/grid_search_utils.py`
- `get_predictions_dicts_from_latent_points` from `utils/grid_search_utils.py`
- `visualize_marginal_fits_many_methods` from `visualization/create_marginal_fits.py`
- `combine_pdfs_in_folder` from `visualization/create_marginal_fits.py`

## Logging

The script creates a detailed log file (`create_ground_truth_sessions.log`) that includes:
- Progress information for each step
- File paths of created outputs
- Any errors or warnings
- Summary of completed work

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure the model path is correct and the file exists
2. **Import errors**: Make sure you're running from the correct directory
3. **Memory issues**: Reduce `--num_points` if you encounter memory problems
4. **PDF creation fails**: Check that matplotlib and PyPDF2 are installed

### Error Messages

- `"No suitable 2D model found for testing"`: No 2D models in `artifacts/models/`
- `"Model loaded successfully"`: Model loading worked correctly
- `"Created X simulated points"`: Successfully sampled latent points
- `"Combined PDF created"`: All marginal plots combined successfully

## Integration with Existing Workflow

This script is designed to integrate seamlessly with the existing DLVM codebase:

- Uses the same model loading functions as other scripts
- Leverages existing marginal fit visualization code
- Follows the same file organization patterns
- Compatible with existing parameter dictionaries

The generated ground truth sessions can be used for:
- Model validation and testing
- Comparison with real data
- Understanding latent space structure
- Generating synthetic datasets for analysis
