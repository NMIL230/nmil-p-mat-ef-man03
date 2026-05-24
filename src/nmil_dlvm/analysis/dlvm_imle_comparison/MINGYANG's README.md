### Document for usage of Mingyang's scripts

##### `compare_search_algorithm.py`
This file is used to fit DLVM (compare gradient descent and grid search) as well as IMLE. If you only need DLVM gradient descent's fitting params, you can drop `--compare-grid-search` .



#### `plot_marginal_median.py`
This file is used to plot the marginal figures of fitting params. It will select top 5, middle 5 and bottom 5 to plot.

Set arguments to select the validation data and corresponding folders.

#### `plot_merged_rmse_curves.py`
This file is specified for plotting the paper's figure 4. Please set the `DATASET` in configurations.json to `COLL10_SIM` and then run.

nmil-mat-ef-dlvm-nn/data/COLL10_SIM is where I put newly generated validation data on Claudius II.

#### `plot_compare_search.py`
This could be used to plot separate curves, not merged curves. That is, all curves outputted by it will be based on one ground truth and one synthetic data.

In addition, this file used to be the code to compare fitted params by grid search and gradient descent. However, it is robust with the input.

You can just create a folder and put the gradient descent params in it. Then set the folder path in the arguments. It works and plots only gradient descent.



### See detailed instructions by looking up the command line arguments of each file.