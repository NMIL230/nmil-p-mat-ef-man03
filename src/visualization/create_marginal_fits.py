import torch
import numpy as np
import pandas as pd
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))

sys.path.append(parent_dir)
import json
from utils.mle_utils import *
from utils.data_distribution_utils import (create_folder,RELEVANT_METRICS, CURR_METRICS_DICT,
                                           SUMMARIZED_METRICS_METRIC_TYPES,SUMMARIZED_METRICS,
                                           VIS_ORDER_PREFERENCE_METRICS_MAIN_ONLY,
                                           SUMMARIZED_METRICS_MAIN_MOMENTS_LABEL_DICT,
                                           get_presentable_sigmoid
                                           )
from visualization.create_plots import calculate_figure_params,extract_mle_parameters
import pdb
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from torch.distributions.binomial import Binomial
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy
from scipy.stats import lognorm, binom
import shutil
import subprocess
from PyPDF2 import PdfMerger
import math
import argparse

def visualize_marginal_fits_many_methods(models_data, show_raw_data=True, show_moments=False, show_curves=True,
                            line_thickness=3, show_grid=True, verbose=True,
                             plot_title=None, title_fontsize=14, per_task_rmse_results=None, metric_name='RMSE', metric_font_size=16,
                             show_metric_loss=False, figure_width=6.4, legend_font_size=12, x_tick_font_size=12, x_y_label_font_size=12,
                             show_legend_per_task=True, marker_size=5, show_raw_data_annotation=False,
                             override_auto_font_sizes=False):
    """
    Create visualization comparing multiple models' marginal fits.

    Args:
        models_data (list): List of dictionaries containing model data. Each dict should have:
            - params (dict): Model parameters for each task/metric
            - raw_data (pd.DataFrame): DataFrame with columns 'metric' and 'result' containing raw data points
            - label (str): Label for the model in plot legend
            - color (str): Color to use for this model's plots
            - alpha (float): Alpha/transparency value for plots
        show_raw_data (bool): Whether to plot raw data points
        show_moments (bool): Whether to show distribution moments
        show_curves (bool): Whether to show fitted distribution curves
        line_thickness (int): Thickness of plotted lines
        show_grid (bool): Whether to show grid on plots
        verbose (bool): Whether to print verbose output
        plot_title (str): Title for overall figure
        title_fontsize (int): Font size for figure title
        per_task_rmse_results (dict, optional): A nested dictionary mapping model labels to their per-task RMSEs.
        figure_width (float): Total width of the figure in inches (default: 6.4)
        override_font_sizes (bool): Whether to override the automatically computed font sizes with the provided font_sizes
    """
    plt.style.use('default')
    fig, axes, font_sizes = setup_figure(figure_width)
    x_data, x_lims = generate_x_values()
    tasks = get_tasks()

    if not override_auto_font_sizes:
        x_tick_font_size = font_sizes["x_tick"]
        x_y_label_font_size = font_sizes["x_y_label"]
        legend_font_size = font_sizes["legend"]
        marker_size = font_sizes["marker"]
        title_font_size = font_sizes["title"]
        text_scale_factor = font_sizes["text_scale_factor"]
        metric_font_size = font_sizes["metric_font_size"]
        line_thickness = font_sizes["line_thickness"]

    for i, task in enumerate(tasks):
        # ---------- build the "(R=1, Loss=…)" suffix ----------
        loss_suffix = ""
        metric_type = SUMMARIZED_METRICS_METRIC_TYPES[task]
        metric_font_size, padding = metric_font_size, figure_width * 0.0
        if show_metric_loss:
            import re
            variants = {
                task,
                task.replace("_w_length_", "_w_len_"),
                task.replace("_w_len_", "_w_length_"),
                re.sub(r"_correct(_w_(?:len|length)_\d+)?$", "", task),
            }
            parts = []
            for mdl in models_data:
                loss_map = mdl.get("metric_losses", {})
                k = next((v for v in variants if v in loss_map), None)
                if not k:
                    continue
                m = re.search(r"(\bR|Restart)=(\d+)", mdl["label"], flags=re.I)
                tag = f"R={m.group(2)}" if m else mdl["label"]
                parts.append(f"{tag}, Loss={loss_map[k]:.6f}")

            if parts:
                loss_suffix = "(" + ";\n".join(parts) + ")"
            metric_font_size, padding = metric_font_size, 0

        ax = configure_axis(axes[i], task, loss_suffix, metric_font_size, padding, metric_type,
                            x_tick_font_size=x_tick_font_size, x_y_label_font_size=x_y_label_font_size,
                            text_scale_factor=text_scale_factor)
        # -------------------------------------------------------

        for j, model in enumerate(models_data):
            plot_model_data(ax, model, task, x_data, x_lims,
                            show_raw_data, show_moments, show_curves,
                             line_thickness,verbose=verbose,
                            per_task_rmse_results=per_task_rmse_results,
                            metric_name=metric_name,
                            marker_size=marker_size,
                            show_raw_data_annotation=show_raw_data_annotation,
                            font_sizes=None if override_auto_font_sizes else font_sizes)

        # Move legend logic outside the inner loop
        if show_legend_per_task:
            ax.legend(fontsize=legend_font_size)
        elif i == len(tasks)-1:  # show legend for the last task
            # Get legend handles and labels from first axis
            handles, labels = axes[0].get_legend_handles_labels()
            ax.legend(handles, labels, fontsize=legend_font_size, loc='upper left')
        else:
            ax.legend().set_visible(False)

        if show_grid:
            ax.grid(True)

    fig.suptitle(plot_title or "Marginal Fits Visualization", fontsize=title_font_size, y=0.95)
    fig.subplots_adjust(top=0.94, bottom=0.1, left=0.1, right=0.95, hspace=0.1, wspace=0.1)
    return fig

def setup_figure(figure_width=6.4):
    fig = plt.figure(facecolor='white')
    # Set subfigure size to achieve 6.4" total width
    # Calculate subfigure width based on desired total width
    num_figures = len(SUMMARIZED_METRICS)
    num_rows, num_cols, _, _ = calculate_figure_params(num_figures)
    subfig_width = figure_width / num_cols  # Divide total width by number of columns
    subfig_height = subfig_width  # Keep square aspect ratio
    num_rows, num_cols, total_width, total_height = calculate_figure_params(num_figures, subfig_size=(subfig_width, subfig_height))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(total_width, total_height))

    # compute the sizes the different elements based on the figure width and height
    # these values are based on 6.4" - rescale appropriately to match the figure width
    text_scale_factor = figure_width / 6.4
    title_font_size = 10 * text_scale_factor
    x_tick_font_size = 5* text_scale_factor
    x_y_label_font_size = 6 * text_scale_factor
    legend_font_size = 5 * text_scale_factor
    marker_size = 2 * text_scale_factor
    metric_font_size = 7 * text_scale_factor
    line_thickness = 1 * text_scale_factor
    sigmoid_data_marker_size = 5 * text_scale_factor

    font_sizes = {
        "title": title_font_size,
        "x_tick": x_tick_font_size,
        "x_y_label": x_y_label_font_size,
        "legend": legend_font_size,
        "marker": marker_size,
        "text_scale_factor": text_scale_factor,
        "metric_font_size": metric_font_size,
        "line_thickness": line_thickness,
        "sigmoid_data_marker_size": sigmoid_data_marker_size
    }

    return fig, axes.flatten() if num_rows > 1 else [axes], font_sizes

def generate_x_values():
    # x_lognorm = np.linspace(0, 3000, 2500) # 2500 points in the range of 0 to 3000
    x_lognorm = np.linspace(0, 2000, 2000) # 2500 points in the range of 0 to 3000
    x_sigmoid = np.linspace(1, 11, 2500)
    x_binom = np.arange(0, 41, 1)

    return {
        "timing": x_lognorm,
        "binarySpan": x_sigmoid,
        "binary": x_binom
    }, {
        "timing": get_x_limits(x_lognorm),
        "binarySpan": get_x_limits(x_sigmoid),
        "binary": get_x_limits(x_binom)
    }

def get_x_limits(x_values):
    padding = 0.05 * (x_values.max() - x_values.min())
    return x_values.min() - padding, x_values.max() + padding

def get_tasks():
    return VIS_ORDER_PREFERENCE_METRICS_MAIN_ONLY or SUMMARIZED_METRICS

def configure_axis(ax, task, loss_suffix,
                   metric_font_size, padding, metric_type, x_tick_font_size=12, x_y_label_font_size=12, text_scale_factor=1):

    pretty = SUMMARIZED_METRICS_MAIN_MOMENTS_LABEL_DICT.get(task, task)

    if metric_type == "timing":
        ax.set_xlabel("Response Time (ms)", fontsize=x_y_label_font_size)
        ax.set_ylabel("Density", fontsize=x_y_label_font_size)
    elif metric_type == "binary":
        ax.set_xlabel("Number of Successes", fontsize=x_y_label_font_size)
        ax.set_ylabel("Probability", fontsize=x_y_label_font_size)
    elif metric_type == "binarySpan":
        ax.set_xlabel("Number of items", fontsize=x_y_label_font_size)
        ax.set_ylabel("Probability", fontsize=x_y_label_font_size)
    ax.set_axisbelow(True)
    ax.set_title(f"{pretty}{loss_suffix}", fontsize=metric_font_size, pad=padding)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5 * text_scale_factor)
    ax.spines["bottom"].set_linewidth(0.5 * text_scale_factor)
    
    ax.tick_params(axis='both', which='major', labelsize=x_tick_font_size)
    return ax


def plot_model_data(ax, model, task, x_data, x_lims, show_raw_data, show_moments, show_curves,
                     moment_line_thickness=1.5, line_thickness=3, verbose=True, per_task_rmse_results=None, metric_name='RMSE', marker_size=10, show_raw_data_annotation=False, font_sizes=None):
    params = model['params'].get(task, None)
    df = model.get('raw_data', None)
    method_label = model.get('label', "Model")
    color = model.get("color", "blue")
    alpha = model.get("alpha", 0.7)
    use_solid_line = model.get("use_solid_line", True) # True for solid line, False for dashed line
    show_raw_data = model.get("show_raw_data", show_raw_data)
    show_raw_data_annotation = model.get("show_raw_data_annotation", show_raw_data_annotation)

    if font_sizes is not None:
        line_thickness = font_sizes["line_thickness"]
        marker_size = font_sizes["marker"]
    else:
        line_thickness = model.get("line_thickness", line_thickness)
        marker_size = model.get("marker_size", marker_size)
    values, n = extract_values(df, task)
    task_type = SUMMARIZED_METRICS_METRIC_TYPES[task]

    if df is not None:
        tasK_rows_df= df[df["metric"].str.contains(task)]
    else:
        tasK_rows_df = None

    if verbose:
        print(f"Plotting for task {task}")

    final_label = method_label
    # check "Ground Truth" label, we don't compute or display RMSE for it
    if per_task_rmse_results and 'Ground Truth' not in method_label:
        model_rmses = per_task_rmse_results.get(method_label)
        if model_rmses:
            task_rmse = model_rmses.get(task)
            # (not a number) is a special value in pandas that indicates that the value is missing or not applicable
            if task_rmse is not None and pd.notna(task_rmse):
                final_label = f"{method_label} ({metric_name}: {task_rmse:.3f})"

    plot_functions = {
        "timing": plot_lognormal,
        "binarySpan": plot_sigmoid,
        "binary": plot_binary
    }

    plot_params = {
        "ax": ax,
        "params": params,
        "x_values": x_data[task_type],
        "color": color,
        "alpha": alpha,
        "label": final_label,
        "n": n,
        "show_curves": show_curves,
        "line_thickness": line_thickness,
        "moment_line_thickness": moment_line_thickness,
        "show_raw_data": show_raw_data,
        "values": values,
        "df": tasK_rows_df,
        "task": task,
        "use_solid_line": use_solid_line,
        "marker_size": marker_size,
        "show_raw_data_annotation": show_raw_data_annotation,
        "font_sizes": font_sizes
    }

    if task_type in plot_functions:
        plot_functions[task_type](**plot_params)

    ax.set_xlim(x_lims[task_type])


def extract_values(df, task):
    if df is not None:
        task_rows = df[df["metric"].str.contains(task)]
        values = task_rows['result'].values if not task_rows.empty else None
        return values, len(task_rows)
    return None, 0

def plot_lognormal(ax, params, x_values, color, alpha, label, n, show_curves,
                    show_raw_data, values=None,df=None, show_moments=True,direction=1, moment_jitter_percentage=0.01, task=None, 
                    use_solid_line=False, line_thickness=3, moment_line_thickness=1.5, 
                    marker_size=10, show_raw_data_annotation=False, font_sizes=None):
    if params is not None:
        mu, sigma = torch.tensor(params[0]).item(), torch.tensor(params[1]).item()
        y = lognorm.pdf(x_values, sigma, 0, np.exp(mu))
        if show_curves:
            # ax.plot(x_values, y, '-', color=color, label=f"{label}(n={n})", linewidth=line_thickness*0.8, alpha=alpha)
            label_with_n = f"{label} (n={n})" if n > 0 else label
            ax.plot(x_values, y, '-', color=color, label=label_with_n, linewidth=line_thickness * 0.8, alpha=alpha, linestyle='-' if use_solid_line else '--')
        if show_moments:
            y_range = ax.get_ylim()
            jitter = direction * moment_jitter_percentage * (y_range[1] - y_range[0])
            ax.plot(np.exp(mu), 0.002 + jitter, 'd', markersize=marker_size, color=color,alpha=alpha)  # diamond marker
            ax.hlines(0.002 + jitter, np.exp(mu - sigma), np.exp(mu + sigma), color=color, linewidth=moment_line_thickness*0.8, alpha=alpha)
    if show_raw_data and values is not None:
        ax.plot(values, np.zeros_like(values), '|', color=color, markersize=marker_size, mew=int(marker_size/2),alpha=0.5) # mew is marker edge width
    ax.set_ylim([-.0002, 0.0045])

def plot_sigmoid(ax, params, x_values, color, alpha, label, n, show_curves,
                  show_raw_data=True, df=None,task=None, values=None, 
                  use_solid_line=False, line_thickness=3, moment_line_thickness=1.5, marker_size=10, 
                  show_raw_data_annotation=False, font_sizes=None):
    if params is not None:
        psiTheta, psiSigma = torch.tensor(params[0]), torch.tensor(params[1])
        model_sigmoid = get_presentable_sigmoid(psiTheta, psiSigma)
        y = model_sigmoid(torch.tensor(x_values)).detach().numpy()
        if show_curves:
            # ax.plot(x_values, y, '-', color=color, label=f"{label}(n={n})", linewidth=line_thickness*0.8,alpha =alpha)
            label_with_n = f"{label} (n={n})" if n > 0 else label
            ax.plot(x_values, y, '-', color=color, label=label_with_n, 
                    linewidth=line_thickness * 0.8, alpha=alpha, linestyle='-' if use_solid_line else '--')

    if show_raw_data and df is not None:
        unique_lengths = df["metric"].apply(lambda x: int(x.split('_')[-1])).unique()
        for length in unique_lengths:
            length_rows = df[df["metric"].apply(lambda x: int(x.split('_')[-1])) == length]
            prop_correct = length_rows["result"].mean()
            ax.scatter([length], [prop_correct], color=color, s=font_sizes["sigmoid_data_marker_size"], alpha=0.4)
            if show_raw_data_annotation:
                ax.annotate(f"N = {len(length_rows)}", (length, prop_correct), textcoords="offset points", xytext=(0,10), ha='center', fontsize=marker_size, alpha=0.8)
    ax.set_ylim([-.05, 1.05])
    ax.set_xticks(np.arange(0, 11, 2))

def plot_binary(ax, params, x_values, color, alpha, label, n, show_curves, show_raw_data, values,df=None, task=None, 
                use_solid_line=False, line_thickness=3, moment_line_thickness=1.5, 
                marker_size=10, show_raw_data_annotation=False, font_sizes=None):
    if params is not None:
        prop = params[0]
        y = binom.pmf(x_values, 40, torch.tensor(prop).detach().numpy())
        if show_curves:
            # ax.bar(x_values, y, color=color, label=f"{label}(n={n})",alpha=alpha)
            
            label_with_n = f"{label} (n={n})" if n > 0 else label
            ax.bar(x_values, y, color=color, label=label_with_n, alpha=alpha, linestyle='-' if use_solid_line else '--')
    if show_raw_data and values is not None:
        prop_mle = np.sum(values) / len(values)
        ax.axvline(prop_mle * 40, color=color, linestyle='dashed', alpha=alpha, linewidth=line_thickness)
    ax.set_ylim([-0.05, 1.05])


def visualize_marginal_fits_deprecated(df_A,   filename,params_A = None, mle_params_A = None,
                        param_type='A', label_A ="A",  data_source_label="TB and ML", show_raw_data=True, show_moments=False, show_curves=True,
                        moment_line_thickness=1.5, line_thickness = 3, show_grid=True, verbose = True, plot_title = None, method1="DLVM", method2="MLE"):
    plt.style.use('default')  # For better style
    fig = plt.figure(facecolor='white')  # White background

    # Make the x bounds
    x_lognorm = np.linspace(0, 3000, 2500)
    x_sigmoid = np.linspace(1, 11, 2500)
    x_binom = np.arange(0, 40+1,1)

    # Calculate 5% of the range for each x array
    x_lognorm_padding = 0.05 * (x_lognorm.max() - x_lognorm.min())
    x_sigmoid_padding = 0.05 * (x_sigmoid.max() - x_sigmoid.min())
    x_binom_padding = 0.05 * (x_binom.max() - x_binom.min())
    # Set the limits
    x_lognorm_lims = (x_lognorm.min() - x_lognorm_padding, x_lognorm.max() + x_lognorm_padding)
    x_sigmoid_lims = (x_sigmoid.min() - x_sigmoid_padding, x_sigmoid.max() + x_sigmoid_padding)
    x_binom_lims = (x_binom.min() - x_binom_padding, x_binom.max() + x_binom_padding)

    num_rows, num_cols, total_width, total_height =calculate_figure_params(num_figures=len(SUMMARIZED_METRICS))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=( total_width, total_height))

    axes = axes.flatten()

    tasks = VIS_ORDER_PREFERENCE_METRICS_MAIN_ONLY if VIS_ORDER_PREFERENCE_METRICS_MAIN_ONLY else SUMMARIZED_METRICS
    task_names = VIS_ORDER_PREFERENCE_METRICS_MAIN_ONLY if VIS_ORDER_PREFERENCE_METRICS_MAIN_ONLY else SUMMARIZED_METRICS

    base_hues = [330, 30, 240, 100, 0, 200,60, 160,300,180,270,320]  # These are the hues for magenta, orange, blue, green, red, cyan in degrees
    base_saturation = 0.4  # Regular parameters will have this saturation value
    dlvm_saturation = 1  # dlvm parameters will have this saturation value
    value = 0.85  # This is the brightness value

    color_B = (0.7, 0.7, 0.7)  # Light gray color

    colors_B = [color_B for _ in tasks]

    # Ensure the colors_A list has the same length as tasks
    colors_A = [mcolors.hsv_to_rgb(((hue + 0) / 360, dlvm_saturation, value)) for hue in base_hues[:len(tasks)]]

    # Identify the data source and participant ID from filename
    participant_id = os.path.basename(filename)

    moment_jitter_percentage = 0.01

    # For each task type
    for i, task in enumerate(tasks):
        data_task = task
        task_has_mle = False
        task_has_params= False
        task_rows_A = df_A[df_A['task'].str.contains(data_task) ]

        # values for the regular and dlvm datasets
        values_A = task_rows_A['value'] if task_rows_A is not None else None

        params,mle_params = (params_A,mle_params_A)
        if verbose:
            print(f"Plotting for task {task}")

        # Choose the appropriate row and values based on the parameter set

        values = values_A.values
        task_rows = task_rows_A
        values_count = 0 if values is None else values.shape[0]

        # Append "_A" suffix to the column names for dlvm parameters
        suffix = label_A

        # Determine the direction of the jitter
        direction = 1

        # Add suffix to color for dlvm parameters
        color = colors_A[i]

        mle_color = color_B
        mle_alpha =0.9
        data_color = color_B

        n = len(task_rows_A)
        axes[i].set_axisbelow(True)
        axes[i].set_title(f"{SUMMARIZED_METRICS_MAIN_MOMENTS_LABEL_DICT[task_names[i]]}", fontsize=14)

        if SUMMARIZED_METRICS_METRIC_TYPES[task] == "timing":  # Lognormals
            task_id = task
            if params != None:
                if task_id in params:
                    task_has_params =True
                    mu = torch.tensor(params[task_id][0]).item()
                    sigma =torch.tensor(params[task_id][1]).item()
                    y = lognorm.pdf(x_lognorm, sigma, 0, np.exp(mu))
            if mle_params != None:
                if task in mle_params:
                    task_has_mle = True
                    mu_mle = torch.tensor(mle_params[task_id][0]).item()
                    sigma_mle =torch.tensor(mle_params[task_id][1]).item()
                    y_mle = lognorm.pdf(x_lognorm, sigma_mle, 0, np.exp(mu_mle))

            if show_curves:
                if params != None and task_has_params:
                    axes[i].plot(x_lognorm, y, '-', color=color, label=f"{method1}(n={n})", linewidth=line_thickness*0.8)
                if mle_params != None and task_has_mle:
                    axes[i].plot(x_lognorm, y_mle, '-', color=mle_color, label=f"{method2}(n={n})", linewidth=line_thickness,alpha=mle_alpha)
            if show_raw_data  and values_count>0:
                axes[i].plot(values, np.zeros_like(values), '|', color=data_color, markersize=10, mew=1.5, alpha=0.6)

            y_range = axes[i].get_ylim()
            moment_jitter_value = direction * moment_jitter_percentage * (y_range[1] - y_range[0])

            if show_moments:
                if params != None and task_has_params:
                    axes[i].plot(np.exp(mu), 0.002 + moment_jitter_value, 'd', markersize=10, color=color)  # diamond marker
                    axes[i].hlines(0.002 + moment_jitter_value, np.exp(mu - sigma), np.exp(mu + sigma), color=color, linewidth=moment_line_thickness*0.8)  # lines for 2nd moment

                if mle_params != None and task_has_mle:
                    axes[i].plot(np.exp(mu_mle), 0.002 + moment_jitter_value, 'd', markersize=10, color=color)  # diamond marker
                    axes[i].hlines(0.002 + moment_jitter_value, np.exp(mu_mle - sigma_mle), np.exp(mu_mle + sigma_mle), color=mle_color, linewidth=moment_line_thickness,alpha=mle_alpha)  # lines for 2nd moment
            axes[i].set_xlim(x_lognorm_lims)
            axes[i].set_ylim([-.0002, 0.0045])

        elif SUMMARIZED_METRICS_METRIC_TYPES[task] == "binarySpan":  # Sigmoids

            if params != None:
                if task in params : # missing parameters
                    task_has_params = True
                    psiTheta = torch.tensor(params[task][0])
                    psiSigma = torch.tensor(params[task][1])
                    model_sigmoid = get_presentable_sigmoid(psiTheta, psiSigma)
                    y = model_sigmoid(torch.tensor(x_sigmoid)).detach().numpy()

            if mle_params != None :
                if task in mle_params:
                    task_has_mle = True
                    mle_psiTheta = torch.tensor(mle_params[task][0])
                    mle_psiSigma = torch.tensor(mle_params[task][1])
                    # softplus if negative or 0
                    mle_model_sigmoid = get_presentable_sigmoid(mle_psiTheta, mle_psiSigma)
                    mle_y = mle_model_sigmoid(torch.tensor(x_sigmoid)).detach().numpy()

            if show_curves:
                if params != None and task_has_params:
                    axes[i].plot(x_sigmoid, y, '-', color=color, alpha=1, label=f"{method1}(n={n})", linewidth=line_thickness*0.8)
                if mle_params != None and task_has_mle:
                    axes[i].plot(x_sigmoid, mle_y, '-', color=mle_color, label=f"{method2}(n={n})", linewidth=line_thickness,alpha=mle_alpha)

            if show_raw_data and values_count>0:
                unique_lengths = task_rows['task'].apply(lambda x: int(x.split('_')[-1])).unique()
                for length in unique_lengths:
                    length_rows = task_rows[task_rows['task'].apply(lambda x: int(x.split('_')[-1])) == length]
                    prop_correct = length_rows['value'].mean()
                    axes[i].scatter([length], [prop_correct], color=data_color, s=120)
                    axes[i].annotate(f"N = {len(length_rows)}", (length, prop_correct), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
            y_range = axes[i].get_ylim()
            moment_jitter_value = direction * moment_jitter_percentage * (y_range[1] - y_range[0])
            if show_moments:
                from scipy.optimize import fsolve
                # Calculate second moment as width between y = 0.16 and y = 0.84
                if params != None and task_has_params:
                    x_low = fsolve(lambda x: model_sigmoid(torch.tensor([x])).item() - 0.16, torch.tensor(a).item())
                    x_high = fsolve(lambda x: model_sigmoid(torch.tensor([x])).item() - 0.84, torch.tensor(a).item())
                    axes[i].plot(torch.tensor(a).item(), 0.5 + moment_jitter_value, 'd', markersize=10, color=color)  # diamond marker
                    axes[i].hlines(0.5 + moment_jitter_value, x_low, x_high, color=color, linewidth=moment_line_thickness*0.8)  # lines for 2nd moment moment

                if mle_params != None and task_has_mle:
                    mle_x_low = fsolve(lambda x: mle_model_sigmoid(torch.tensor([x])).item() - 0.16, torch.tensor(mle_a).item())
                    mle_x_high = fsolve(lambda x: mle_model_sigmoid(torch.tensor([x])).item() - 0.84, torch.tensor(mle_a).item())
                    axes[i].plot(torch.tensor(mle_a).item(), 0.5 + moment_jitter_value, 'd', markersize=10, color=mle_color,alpha=0.6)  # diamond marker
                    axes[i].hlines(0.5 + moment_jitter_value, mle_x_low, mle_x_high, color=mle_color, linewidth=moment_line_thickness,alpha=0.6)  # lines for 2nd moment moment

            axes[i].set_xlim(x_sigmoid_lims)
            axes[i].set_ylim([-.05, 1.05])

        else:  # Binomials
            if params != None:
                if task in params:
                    task_has_params = True
                    prop = params[task][0]
                    y = binom.pmf( x_binom, 40, torch.tensor(prop).detach().numpy())

            if mle_params != None:
                if task in mle_params:
                    task_has_mle = True
                    mle_prop = mle_params[task][0]
                    mle_y = binom.pmf(x_binom, 40, mle_prop)

            if show_curves:
                if params != None and task_has_params:
                    axes[i].bar(x_binom, y, color=color, alpha=1, label=f"DLVM(n={n})")  # make bars semi-transparent

                if mle_params != None and task_has_mle:
                    axes[i].bar(x_binom, mle_y, color=mle_color, alpha=mle_alpha, label=f"MLE(n={n})")  # make bars semi-transparent

            if show_raw_data and values_count>0:
                prop_mle = np.sum(values)/values_count
                axes[i].axvline(prop_mle*40, color=data_color, linestyle='dashed', alpha=mle_alpha)  # Draw a vertical dashed line

            axes[i].set_xlim(x_binom_lims)
            axes[i].set_ylim([-0.05, 1.05])




        axes[i].set_ylabel(r'Density' if SUMMARIZED_METRICS_METRIC_TYPES[task] == "timing" else r'Probability')
        if show_grid:
            axes[i].grid(True)
        axes[i].legend()
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['left'].set_linewidth(1.5)  # Set the linewidth of the left spine to create the vertical line
        axes[i].spines['bottom'].set_linewidth(1.5)  # Set the linewidth of the bottom spine to create the horizontal line
        # Set the number of y-axis labels
        axes[i].yaxis.set_major_locator(ticker.MaxNLocator(6))

    if plot_title is None:
        plot_title = f"Session: {participant_id} | Data Source : {data_source_label} "
    fig.suptitle(plot_title, fontsize=24 ) #, y=1.05
    fig.tight_layout()
    return fig # Return the figure so we can save it

def extract_dlvm_params(dlvm_model_id):

    # dlvm_fit
    try:
        dlvm_params = pd.read_csv(f"./outputs/{DATASET}/{dlvm_model_id}/{dlvm_model_id}_all_moments_predictions.csv")
    except FileNotFoundError:
        print(f"System Exiting. File not found: ./outputs/{DATASET}/{dlvm_model_id}/{dlvm_model_id}_all_moments_predictions.csv")
        sys.exit(1)  # Exit the program with a non-zero status to indicate an error
    # Create the dictionary
    result = {}

    # Group by session
    grouped = dlvm_params.groupby('session')

    for session, group in grouped:
        session_dict = {}

        for col in group.columns:
            if col == 'session':
                continue

            base_name = '_'.join(col.split('_')[:-1])
            param_type = col.split('_')[-1]

            if 'param1' in param_type or 'param2' in param_type:
                if base_name not in session_dict:
                    session_dict[base_name] = [None, None]
                if 'param1' in param_type:
                    session_dict[base_name][0] = torch.tensor(group.iloc[0][col])
                if 'param2' in param_type:
                    session_dict[base_name][1] = torch.tensor(group.iloc[0][col])
            else:
                session_dict[col] = torch.tensor(group.iloc[0][col])

        # Clean up lists where there is no param2
        for key in list(session_dict.keys()):
            if isinstance(session_dict[key], list) and session_dict[key][1] is None:
                session_dict[key] = torch.tensor([session_dict[key][0]])

        result[session] = session_dict

    dlvm_params = result

    return dlvm_params


def generate_marginal_fits(dlvm_model_id = "good-energy-261"):
    best_params_path = os.path.join(os.path.dirname(__file__),
                                    '..', 'data', DATASET, 'all_data-best_mle_params_mpf100.pt')
    mle_best_params = torch.load(
        best_params_path
    )
    df = pd.read_csv(
            os.path.join(parent_dir, f"data/{DATASET}/all_data_time_not_normed.csv")
        )
    df=df.rename(columns ={"metric":"task","result":"value"})

    plot_fits(df, mle_best_params,dlvm_model_id)



    if dlvm_model_id is None:
        # Assuming you already have a DataFrame named data_df3
        processed_mle_params = f"outputs/{DATASET}/{DATASET}_mle_all_moments.csv"
        extract_mle_parameters(mle_params_file_path=f'../data/{DATASET}/all_data-best_mle_params_mpf100.pt')
        if os.path.exists(processed_mle_params):
            df = pd.read_csv(processed_mle_params)
            try:
                plot_MLE_param_histogram(
                    df,
                    filename=f"mle/{DATASET}/_mle_params.pdf",
                    title=f"MLE parameters fitted for individual sessions (n={len(mle_best_params.keys())})"
                )
            except Exception as e:
                print(e)

        output_path = f'presentations/{DATASET}/MLE_combined_fits.pdf'
        combine_pdfs_in_folder(f"mle/{DATASET}/", output_path)
    else:
        hist_file_path = f"outputs/{DATASET}/{DATASET}_{dlvm_model_id}_vs_mle_params.pdf"
        destination_file = f"outputs/{DATASET}/{dlvm_model_id}/fits/_mle_params.pdf"

        if os.path.exists(hist_file_path):
            os.makedirs(os.path.dirname(destination_file), exist_ok=True)  # Ensure the destination directory exists
            shutil.copy2(hist_file_path, destination_file)
            print(f"File copied to {destination_file}")
        else:
            print(f"File {hist_file_path} does not exist")


        output_path = f'presentations/{DATASET}/{dlvm_model_id}_combined_fits.pdf'
        combine_pdfs_in_folder(f"outputs/{DATASET}/{dlvm_model_id}/fits/", output_path)

def combine_pdfs_in_folder(folder_path, output_path):
    merger = PdfMerger()

    # Iterate over all files in the folder
    for filename in ["_mle_params.pdf"]+sorted(os.listdir(folder_path)):
        if filename.endswith('.pdf') and not "combined" in filename:
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


def plot_fits(raw_data_df, mle_best_params,dlvm_model_id):
    mle_params = {}
    vis_key = "user_session" #"user_name"
    sessions = raw_data_df[vis_key].unique()
    if dlvm_model_id is not None:
        dlvm_params = extract_dlvm_params(dlvm_model_id) #extract predictions of this model
    save_folder = f"./outputs/{DATASET}/{dlvm_model_id}/fits"
    create_folder(save_folder)
    for user_session in mle_best_params.keys():
        session_df = raw_data_df[raw_data_df[vis_key]==user_session]

        metrics = session_df["task"].unique()

        update_w_data_dict = session_df.groupby('task')['value'].agg(list).to_dict()

        for metric in RELEVANT_METRICS:
            if not (metric in update_w_data_dict.keys()):
                update_w_data_dict[metric] =[]


        mle_params[user_session]= mle_best_params[user_session]
        # Check if heldout_ids.txt exists, else use the default set
        heldout_ids_path = f"./outputs/{DATASET}/heldout_ids.txt"
        if os.path.exists(heldout_ids_path):
            with open(heldout_ids_path, "r") as heldout_ids_txt:
                heldout_ids = eval(heldout_ids_txt.read())
        else:
            heldout_ids = DEFAULT_HELDOUT_SET

        df_A=session_df
        df_B = session_df
        if not dlvm_model_id is None:
            plot_title = f"Session: {user_session} | Data Source : TB | dataset : {DATASET}| In Train set: {'Yes' if not (user_session in heldout_ids) else 'No'}"
            fig = visualize_marginal_fits_deprecated(df_A, filename = user_session,params_A = dlvm_params[user_session], mle_params_A = mle_params[user_session],
                                label_A ="TB", data_source_label="TB", show_raw_data=True,
                                show_moments=False, show_curves=True,
                                moment_line_thickness=1.5, line_thickness = 3, show_grid=True,verbose=False, plot_title = plot_title)

            fig.savefig(f"outputs/{DATASET}/{dlvm_model_id}/fits/{user_session}.pdf",format='pdf',dpi=300)
        else:
            fig = visualize_marginal_fits_deprecated(df_A,   filename = user_session,params_A = None, mle_params_A = mle_params[user_session],
                                    param_type='TB', label_A ="TB", data_source_label="TB", show_raw_data=True,
                                    show_moments=False, show_curves=True,
                                    moment_line_thickness=1.5, line_thickness = 3, show_grid=False,verbose=False)
            fig.savefig(f"mle/{DATASET}/{user_session}.pdf",format='pdf',dpi=300)
        plt.close()


def plot_MLE_param_histogram(data_df,filename="_mle_params.pdf", title = "MLE parameters"):

    labels = ALL_METRICS_MOMENTS_LABEL_DICT

    # Extract the names for histograms from the first row
    histogram_names = data_df.columns.tolist()[1:]  # Exclude the first column

    # Create subplots for histograms of each column in a 3x4 grid
    num_rows, num_cols, total_width, total_height =calculate_figure_params(num_figures=len(ALL_METRICS_MOMENTS_LABEL_DICT.keys()))
    fig, axs = plt.subplots(num_rows, num_cols, figsize=( total_width, total_height))
    axs = axs.flatten()

    for i, col in enumerate(histogram_names):
        row = i // 4  # Calculate row index for subplot
        col = i % 4   # Calculate column index for subplot

        data_class = data_df.iloc[:, i+1]  # Extract data for the current class

        # Filter out NaN and non-numeric values
        data_class_numeric = data_class[pd.to_numeric(data_class, errors='coerce').notnull()]
        axs[i].set_ylabel('Frequency')
        axs[i].set_title(f'{labels[histogram_names[i]]}')

        axs[i].hist(data_class_numeric, bins=20, alpha=1 if not ("param2" in histogram_names[i]) else 0.5, label=f'Histogram(n={len(data_class_numeric)})')  # Plot histogram
        axs[i].set_xlabel('Value Range', color='black')
        axs[i].tick_params(axis='x', labelcolor='black')
        axs[i].legend()

    fig.suptitle(title,fontsize = 24)
    plt.tight_layout()  # Adjust layout
    fig.patch.set_facecolor('white')

    plt.gca().set_facecolor('white')
    plt.savefig(filename)
    plt.close()



def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process CSV and generate marginal fits for given latent_dim.")
    parser.add_argument('--latent_dim', type=int, default=3, help='Latent dimension to filter on (default: 3)')
    parser.add_argument('--model_id', type=str, help='Specific model ID to filter on (optional)', default=None)
    parser.add_argument('--mle_only', action='store_true', help='True if MLE only should be visualized ')

    args = parser.parse_args()

    if not args.mle_only:
        latent_dim = args.latent_dim
        user_model_id = args.model_id

        # Define the fixed path to the CSV file
        runs_date_csv_path = os.path.join(parent_dir, f"model_training_analysis/{DATASET}/runs_data.csv")

        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(runs_date_csv_path)

        # Filter the DataFrame based on the given latent_dim
        df = df[df["latent_dim"] == latent_dim]

        # If a specific model ID is provided, further filter the DataFrame for that model ID
        if user_model_id:
            df = df[df["model_id"] == user_model_id]

        # Get the list of model IDs
        model_ids = df['model_id'].tolist()
        model_ids.append(None)

        # Run generate_marginal_fits for each model ID
        for model_id in model_ids:
            generate_marginal_fits(dlvm_model_id=model_id)
    else:
        generate_marginal_fits(dlvm_model_id=None)

if __name__ == '__main__':
    main()
