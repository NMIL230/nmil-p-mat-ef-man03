### Validation Pipeline Notes

#### `fit_dlvm_and_imle_models_to_data.py`
This file is used to fit DLVM models and IMLE baselines. If you only need DLVM gradient-descent fitting parameters, you can omit grid-search related options.

#### `plot_representative_marginal_fits.py`
This file is used to plot representative marginal-fit figures. It selects top 5, middle 5, and bottom 5 sessions to plot.

Set arguments to select the validation data and corresponding folders.

#### `plot_merged_curves.py`
This file is used for the manuscript merged-curve figure workflow. Set the dataset configuration to `COLL10_SIM` before running the validation pipeline.

`data/COLL10_SIM` is where the validation data generated on Claudius II was stored historically.

#### `fit_error_metrics.py`
This can be used to plot separate curves rather than merged curves. The outputs are based on one ground-truth input and one synthetic-data input at a time.

It was also used to compare fitted parameters from grid search and gradient descent. You can create a folder containing only the gradient-descent parameters, point the arguments at that folder, and the script will plot only those results.

### See detailed instructions by looking up the command-line arguments of each file.
