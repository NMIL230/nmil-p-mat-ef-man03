#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to plot DALE/IMLE/DLVM comparison data with reusable styling
and smart axis limits (auto tight bounds with margins + manual override).

Key features:
- Generic plotting with auto axis limits (linear/log) and manual overrides (--xlim/--ylim)
- Reusable method->(color, marker, label) styling outside the plotting function
- Cleaned data prep compatible with previous CSV schema
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent))
sys.path.insert(0, str(script_dir.parent.parent))

from analysis.analysis_utils.utils_plot import plot_generic_comparison

def build_method_properties(all_methods: List[str]) -> Dict[str, Dict[str, str]]:


    # Standard palette mapping (Matplotlib tab10):
    # Blue:  #1f77b4, Orange: #ff7f0e, Red: #d62728, Green: #2ca02c
    method_config = {
        # IMLE counterparts — same colors as DLVM but dashed lines
        "IMLE_TB": {
            "color": "#2ca02c",  # Green
            "marker": "o",
            "label": "IMLE + TB",
            "linestyle": "--",
            "sd_fill_alpha": 0.0,
        },
        "IMLE_DALE_PS0": {
            "color": "#1f77b4",  # Blue
            "marker": "o",
            "label": "IMLE + DALE + PS=0",
            "linestyle": "--",
            "sd_fill_alpha": 0.0,
        },
        "IMLE_DALE_PS1": {
            # Not specified by user; keep existing style but ensure consistency if present
            "color": "#38b9c0",
            "marker": "o",
            "label": "IMLE + DALE + PS=1",
            "linestyle": "--",
            "sd_fill_alpha": 0.0,
        },
        "IMLE_DALE_PS2": {
            # Not specified by user; keep existing style but ensure consistency if present
            "color": "#a455f3",
            "marker": "o",
            "label": "IMLE + DALE",
            "linestyle": "--",
            "sd_fill_alpha": 0.0,
        },
        "IMLE_DALE_PS4": {
            "color": "#ff7f0e",  # Orange
            "marker": "o",
            "label": "IMLE + DALE + PS=4",
            "linestyle": "--",
            "sd_fill_alpha": 0.0,
        },
        "IMLE_Random": {
            "color": "#d62728",  # Red
            "marker": "o",
            "label": "IMLE + Random",
            "linestyle": "--",
            "sd_fill_alpha": 0.0,
        },
        # DLVM methods — solid lines, specified colors
        "DLVM_RANDOM": {
            "color": "#d62728",  # Red
            "marker": "s",
            "label": "DLVM-2 + Random ± SD"
        },
        "DLVM_TB": {
            "color": "#2ca02c",  # Green
            "marker": "s",
            "label": "DLVM-2 + TB ± SD"
        },
        "DLVM_DALE_PS0": {
            "color": "#1f77b4",  # Blue
            "marker": "^",
            "label": "DLVM-2 + DALE + PS=0 ± SD"
        },
        "DLVM_DALE_PS1": {
            # Not specified by user; keep existing style
            "color": "#4169E1",
            "marker": "^",
            "label": "DLVM-2 + DALE + PS=1 ± SD"
        },
        "DLVM_DALE_PS2": {
            # Not specified by user; keep existing style
            "color": "#a455f3",
            "marker": "v",
            "label": "DLVM-2 + DALE ± SD"
        },
        "DLVM_DALE_PS4": {
            "color": "#ff7f0e",  # Orange
            "marker": "D",
            "label": "DLVM-2 + DALE + PS=4 ± SD"
        },
    }
    
    props = {}
    for method in all_methods:
        if method in method_config:
            props[method] = method_config[method]
        else:
            props[method] = {
                "color": "gray", 
                "marker": "x", 
                "label": method
            }
    
    return props



def generate_individual_method_plots(
    df_original: pd.DataFrame,
    methods: List[str],
    output_dir: Path,
    metric: str = "rmse",
    max_n: int = 140,
    use_all_trials: bool = True,
    x_scale: str = "linear",
    y_scale: str = "log",
    show_markers: bool = True,
    error_type: str = "SD",
    dataset: str = "Synthetic",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    x_margin: float = 0.05,
    y_margin: float = 0.08,
    min_decades: float = 0.15,
    # New: start-padding ratios
    x_start_pad_ratio: float = 0.0,
    y_start_pad_ratio: float = 0.0,
    # New: font sizes
    title_fontsize: float = 14.0,
    label_fontsize: float = 12.0,
    tick_fontsize: float = 8.0,
    legend_fontsize: float = 6.0,
    # New: figure size
    fig_size: Tuple[float, float] = (6.4, 4.8),
    # New: linear x tick step
    x_tick_step: Optional[float] = None,
    allow_negative_x_pad: bool = False,
    allow_negative_y_pad: bool = False,
    line_width: float = 0.2,
):
    """
    Generate individual plots for each method showing session-wise curves.
    Each method will have its own plot with multiple session lines.
    """
    print(f"Generating individual method plots for {len(methods)} methods...")
    
    value_col = f'{metric}_value'
    
    # Clean data
    df_clean = df_original.dropna(subset=[value_col]).copy()
    df_clean = df_clean[np.isfinite(df_clean[value_col])]
    
    # Data now standardized to 1-based num_tests at source; no per-method fix needed

    # Filter trials
    available_n_values = sorted(df_clean['num_tests'].unique())
    if use_all_trials:
        target_n = list(range(1, max_n + 1))
    else:
        target_n = [1, 2, 3, 5, 10, 20, 50, 100]
        target_n = [n for n in target_n if n <= max_n]
    
    filtered_n = [n for n in target_n if n in available_n_values]
    df_clean = df_clean[df_clean['num_tests'].isin(filtered_n)]
    
    ylabel = "KL Divergence" if metric == "kld" else "RMSE"
    xlabel = "Number of Observed Trials"
    
    # Generate plot for each method
    for method in methods:
        method_data = df_clean[df_clean['method'] == method]
        if method_data.empty:
            print(f"No data for method: {method}")
            continue
            
        # Get all sessions for this method
        sessions = sorted(method_data['session_id'].unique())
        print(f"Plotting {method} with {len(sessions)} sessions")
        
        # Prepare data in the format expected by plot_generic_comparison
        # Each session becomes a "Method" in the plotting DataFrame
        df_sessions = []
        for session in sessions:
            session_data = method_data[method_data['session_id'] == session]
            session_data = session_data.sort_values('num_tests')
            
            df_sessions.append(pd.DataFrame({
                'Method': f"Session_{session}",
                'N': session_data['num_tests'].values,
                'Value': session_data[value_col].values,
                'Error': np.zeros(len(session_data))  # No error bars for individual sessions
            }))
        
        if not df_sessions:
            continue
            
        df_plot_method = pd.concat(df_sessions, ignore_index=True)
        
        # Create method-specific properties for sessions
        session_props = {}
        colors = plt.cm.tab10(np.linspace(0, 1, len(sessions)))
        for i, session in enumerate(sessions):
            session_props[f"Session_{session}"] = {
                "color": matplotlib.colors.rgb2hex(colors[i]),
                "marker": "o",
                "label": f"Session {session}"
            }
        
        # Generate filename
        suffix = "_all_trials" if use_all_trials else "_log_subset"
        if not show_markers:
            suffix += "_lines_only"
        suffix += f"_x{x_scale}_y{y_scale}"
        
        method_filename_base = f"{dataset}_{metric.upper()}_{method}_sessions{suffix}"
        output_path_pdf = output_dir / f"{method_filename_base}.pdf"
        output_path_png = output_dir / f"{method_filename_base}.png"
        
        title = f"{dataset} {metric.upper()} - {method} (All Sessions)"
        
        # Generate PDF
        plot_generic_comparison(
            df_plot=df_plot_method,
            output_path=str(output_path_pdf),
            x_scale=x_scale,
            y_scale=y_scale,
            show_markers=show_markers,
            error_type="SD",  # No error bars for individual sessions
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            method_properties=session_props,
            xlim=xlim, ylim=ylim,
            x_margin=x_margin, y_margin=y_margin, min_decades=min_decades,
            x_start_pad_ratio=x_start_pad_ratio, y_start_pad_ratio=y_start_pad_ratio,
            title_fontsize=title_fontsize, label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize, legend_fontsize=legend_fontsize,
            fig_size=fig_size,
            allow_negative_x_pad=allow_negative_x_pad,
            allow_negative_y_pad=allow_negative_y_pad,
            line_width=line_width
            , x_tick_step=x_tick_step
        )
        
        # Generate PNG
        plot_generic_comparison(
            df_plot=df_plot_method,
            output_path=str(output_path_png),
            x_scale=x_scale,
            y_scale=y_scale,
            show_markers=show_markers,
            error_type="SD",
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            method_properties=session_props,
            xlim=xlim, ylim=ylim,
            x_margin=x_margin, y_margin=y_margin, min_decades=min_decades,
            x_start_pad_ratio=x_start_pad_ratio, y_start_pad_ratio=y_start_pad_ratio,
            title_fontsize=title_fontsize, label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize, legend_fontsize=legend_fontsize,
            fig_size=fig_size,
            allow_negative_x_pad=allow_negative_x_pad,
            allow_negative_y_pad=allow_negative_y_pad,
            line_width=line_width
            , x_tick_step=x_tick_step
        )
        
        print(f"Generated individual plot for {method}")


def parse_limits_arg(arg: Optional[str]) -> Optional[Tuple[float, float]]:

    if arg is None:
        return None
    s = str(arg).strip().lower()
    if s in ("", "auto", "none"):
        return None
    if "," not in s:
        raise ValueError(f"--xlim/--ylim  {arg}")
    a, b = s.split(",", 1)
    return (float(a.strip()), float(b.strip()))


def parse_figsize_arg(arg: Optional[str]) -> Tuple[float, float]:

    if arg is None:
        return (6.4, 4.8)
    s = str(arg).strip()
    if s == "":
        return (6.4, 4.8)
    if "," in s:
        a, b = s.split(",", 1)
        return (float(a.strip()), float(b.strip()))
    else:
        w = float(s)
        return (w, max(0.1, w * 0.75))



def prepare_data_for_plot_rmse_combined(df, metric='kld', max_n=100, use_all_trials=False, methods_filter=None):
    """
    Transform CSV data to format expected by plotting.

    Input df columns: [method, session_id, num_tests, kld_value, rmse_value]
    Output df_plot columns: [Method, N, Value, Error]
    """
    df = df.copy()
    
    # Filter methods if specified
    if methods_filter is not None:
        df = df[df['method'].isin(methods_filter)]


    value_col = f'{metric}_value'

    df_clean = df.dropna(subset=[value_col])
    df_clean = df_clean[np.isfinite(df_clean[value_col])]
    print(f"Cleaned: {len(df)} -> {len(df_clean)} rows")

    grouped = (
        df_clean
        .groupby(['method', 'num_tests'])[value_col]
        .agg(mean=lambda x: np.nanmean(x),
             std=lambda x: np.nanstd(x),
             count=lambda x: (~np.isnan(x)).sum())
        .reset_index()
    )
    grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])

    df_plot = []
    for method in grouped['method'].unique():
        md = grouped[grouped['method'] == method]
        df_plot.append(pd.DataFrame({
            'Method': method,
            'N': md['num_tests'].values,
            'Value': md['mean'].values,
            'Error': md['std'].values 
        }))
    df_plot = pd.concat(df_plot, ignore_index=True)

    available_n_values = sorted(df_clean['num_tests'].unique())
    if use_all_trials:
        target_n = list(range(1, max_n + 1))
        print(f"use_all_trials=True -> 1..{max_n}")
    else:
        target_n = [1, 2, 3, 5, 10, 20, 50, 100]
        target_n = [n for n in target_n if n <= max_n]
        print(f"use_all_trials=False -> subset={target_n}")

    filtered_n = [n for n in target_n if n in available_n_values]
    df_plot = df_plot[df_plot['N'].isin(filtered_n)]

    print(f"N used: {filtered_n}")
    return df_plot, filtered_n


def generate_selected_sessions_pdf(
    df: pd.DataFrame,
    sessions: List[str],
    methods: List[str],
    output_pdf: Path,
    *,
    metric: str = "kld",
    max_n: int = 240,
    use_all_trials: bool = True,
    x_scale: str = "linear",
    y_scale: str = "log",
    show_markers: bool = True,
    dataset: str = "Synthetic",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    x_margin: float = 0.05,
    y_margin: float = 0.08,
    min_decades: float = 0.15,
    x_start_pad_ratio: float = 0.0,
    y_start_pad_ratio: float = 0.0,
    title_fontsize: float = 14.0,
    label_fontsize: float = 12.0,
    tick_fontsize: float = 8.0,
    legend_fontsize: float = 6.0,
    fig_size: Tuple[float, float] = (6.4, 4.8),
    x_tick_step: Optional[float] = None,
    allow_negative_x_pad: bool = False,
    allow_negative_y_pad: bool = False,
    line_width: float = 1.0,
) -> None:
    """Create a multi-page PDF, one page per selected session showing method curves."""
    value_col = f"{metric}_value"
    df = df.dropna(subset=[value_col]).copy()
    df = df[np.isfinite(df[value_col])]

    available_n_values = sorted(df['num_tests'].unique())
    if use_all_trials:
        target_n = list(range(1, max_n + 1))
    else:
        target_n = [1, 2, 3, 5, 10, 20, 50, 100]
        target_n = [n for n in target_n if n <= max_n]
    filtered_n = [n for n in target_n if n in available_n_values]
    df = df[df['num_tests'].isin(filtered_n)]

    # Styling for methods
    method_props = build_method_properties(sorted(set(methods)))

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        for sid in sessions:
            sdf = df[df['session_id'] == sid]
            if sdf.empty:
                print(f"[warn] No data for session {sid}; skipping")
                continue

            # Build tidy df across methods for this session
            rows = []
            for m in methods:
                md = sdf[sdf['method'] == m].sort_values('num_tests')
                if md.empty:
                    continue
                rows.append(pd.DataFrame({
                    'Method': m,
                    'N': md['num_tests'].values,
                    'Value': md[value_col].values,
                    'Error': np.zeros(len(md)),
                }))
            if not rows:
                continue
            df_plot = pd.concat(rows, ignore_index=True)

            title = f"{dataset} {metric.upper()} — Session {sid}"
            fig = plot_generic_comparison(
                df_plot=df_plot,
                output_path=None,
                x_scale=x_scale,
                y_scale=y_scale,
                show_markers=show_markers,
                error_type="SD",
                title=title,
                xlabel="Number of Observed Trials",
                ylabel=("Normalized KL Divergence" if metric == "kld" else "Normalized RMSE"),
                method_properties=method_props,
                legend_order=methods,
                xlim=xlim, ylim=ylim,
                x_margin=x_margin, y_margin=y_margin, min_decades=min_decades,
                x_start_pad_ratio=x_start_pad_ratio, y_start_pad_ratio=y_start_pad_ratio,
                title_fontsize=title_fontsize, label_fontsize=label_fontsize,
                tick_fontsize=tick_fontsize, legend_fontsize=legend_fontsize,
                fig_size=fig_size,
                x_tick_step=x_tick_step,
                allow_negative_x_pad=allow_negative_x_pad,
                allow_negative_y_pad=allow_negative_y_pad,
                line_width=line_width,
                return_fig=True,
            )
            if fig is not None:
                pdf.savefig(fig, dpi=300, bbox_inches='tight')
                plt.close(fig)
    print(f"Selected sessions PDF saved to: {output_pdf}")



def main():
    SCRIPT_ROOT = Path(__file__).resolve().parent
    REPO_ROOT = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Plot comparison with smart axis limits and reusable styles.")
    parser.add_argument("--csv_file", type=Path,
                        default=REPO_ROOT / "analysis/dale_syn_data/plot_data/2d-final/dale_comparison_data.csv",
                        help="Path to CSV file with comparison data")
    parser.add_argument("--output_dir", type=Path,
                        default=REPO_ROOT / "analysis/dale_syn_data/plot/Figure05_v03_TestingFitEvolution",
                        help="Directory to save plots")
    parser.add_argument("--output_filename", type=str, 
                        default="Figure05_v03_TestingFitEvolution",
                        help="Output filename without extension ('.pdf' will be added)")
    parser.add_argument("--error_type", type=str, default="SD", choices=["SD", "SE"],
                        help="Error bar type")
    parser.add_argument("--dataset", type=str, default="Synthetic", help="Dataset name for title")
    parser.add_argument("--max_n", type=int, default=240, help="Max N value")
    parser.add_argument("--use_all_trials", action="store_true", default=True,
                        help="Use all trial values (1..max_n) rather than a log subset")
    parser.add_argument("--lines_only", action="store_true", default=True,
                        help="If set, do not draw markers")

    parser.add_argument("--x_scale", type=str, default="linear", choices=["linear", "log"],
                        help="X-axis scale")
    parser.add_argument("--y_scale", type=str, default="log", choices=["linear", "log"],
                        help="Y-axis scale")

    parser.add_argument("--xlim", type=str, default='-5,245',
                        help="Manual x-range 'lo,hi' (e.g., '1,100'); use 'auto' or omit for auto")
    parser.add_argument("--ylim", type=str, default='auto', #'0,0.7'
                        help="Manual y-range 'lo,hi' (e.g., '1e-3,1'); use 'auto' or omit for auto")
    parser.add_argument("--x_margin", type=float, default=0.05,
                        help="Auto x-margin ratio (linear) or fraction of log-span (log)")
    parser.add_argument("--y_margin", type=float, default=0.08,
                        help="Auto y-margin ratio (linear) or fraction of log-span (log)")
    parser.add_argument("--min_decades", type=float, default=0.15,
                        help="Minimum decades of margin on log scales")
    # New: extra start-side padding ratios
    parser.add_argument("--x_start_pad_ratio", type=float, default=0.0,
                        help="Extra left-side padding ratio (fraction of axis span)")
    parser.add_argument("--y_start_pad_ratio", type=float, default=0.0,
                        help="Extra bottom-side padding ratio (fraction of axis span)")
    parser.add_argument("--allow_negative_x_pad", action="store_true", default=False,
                        help="Allow x start-padding to push axis lower bound below 0")
    parser.add_argument("--allow_negative_y_pad", action="store_true", default=False,
                        help="Allow y start-padding to push axis lower bound below 0")
    # New: linear x tick step
    parser.add_argument("--x_tick_step", type=float, default=20.0,
                        help="Linear x-axis tick step (e.g., 20 shows ticks every 20 units)")
    # New: font sizes
    parser.add_argument("--title_fontsize", type=float, default=14.0,
                        help="Figure title font size (pt)")
    parser.add_argument("--label_fontsize", type=float, default=12.0,
                        help="Axis label font size (pt)")
    parser.add_argument("--tick_fontsize", type=float, default=8.0,
                        help="Tick label font size (pt)")
    parser.add_argument("--legend_fontsize", type=float, default=6.0,
                        help="Legend font size (pt)")
    # New: legend order
    parser.add_argument("--legend_order", type=str, default="DLVM_TB,DLVM_RANDOM,DLVM_DALE_PS2,IMLE_TB,IMLE_Random,IMLE_DALE_PS2",
                        help="Comma/space-separated method names to order legend (others appended)")
    # New: curve line width
    parser.add_argument("--line_width", type=float, default=1.0,
                        help="Curve line width (linewidth)")
    # New: figure size
    parser.add_argument("--fig_size", type=str, default='6.4',
                        help="Figure size inches 'W' or 'W,H' (default width=6.4, height=0.75*W)")
    
    parser.add_argument("--metric", type=str, default="kld", choices=["kld", "rmse"],
                        help="Metric to plot")
    
    parser.add_argument("--methods", type=str, nargs="*", 
                        # default=["IMLE_TB", "DLVM_RANDOM", "DLVM_TB", "IMLE_DALE_PS0", "IMLE_DALE_PS1","IMLE_DALE_PS2","IMLE_DALE_PS4","IMLE_Random","DLVM_DALE_PS0" ,"DLVM_DALE_PS1", "DLVM_DALE_PS2", "DLVM_DALE_PS4"],
                        default=["IMLE_TB", "DLVM_RANDOM", "DLVM_TB", "IMLE_DALE_PS2","IMLE_Random","DLVM_DALE_PS2"],

                        help="List of method names to plot")

    parser.add_argument("--title_name", type=str, 
                        default="Different Model Fits with Data Accumulation",
                        help="Plot title name.")
    
    parser.add_argument("--generate_individual_method_plots", default=False, 
                        help="Generate individual plots for each method showing session-wise curves")
    # New: selected sessions (default 9 LD2 sessions). Use '--sessions none' to disable filtering
    default_sessions = [
        "LD2-041","LD2-037","LD2-021","LD2-047","LD2-045",
        "LD2-034","LD2-086","LD2-028","LD2-042",
    ]
    parser.add_argument("--sessions", type=str, default="none",
                        help="Comma/space-separated session IDs to export as a single multi-page PDF; use 'none' or 'all' for all sessions")
    parser.add_argument("--sessions_output", type=str, default="selected_sessions_2D",
                        help="Base filename (without extension) for the selected sessions PDF")
    
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.csv_file.exists():
        print(f"[Error] CSV not found: {args.csv_file}")
        return 1

    df = pd.read_csv(args.csv_file)
    print(f"Loaded {len(df)} rows from {args.csv_file}")
    print(f"Methods: {sorted(df['method'].unique())}")

    show_markers = (not args.lines_only)

    df_plot, _ = prepare_data_for_plot_rmse_combined(
        df, metric=args.metric, max_n=args.max_n, use_all_trials=args.use_all_trials, methods_filter=args.methods
    )

    ylabel = "Normalized KL Divergence" if args.metric == "kld" else "Normalized RMSE"
    xlabel = "Number of Observed Trials"
    title = args.title_name

    if args.output_filename:
        output_path_pdf = args.output_dir / f"{args.output_filename}.pdf"
        output_path_png = args.output_dir / f"{args.output_filename}.png"
    else:
        suffix = "_all_trials" if args.use_all_trials else "_log_subset"
        if not show_markers:
            suffix += "_lines_only"
        suffix += f"_x{args.x_scale}_y{args.y_scale}"
        filename_base = f"{args.metric.upper()}_combined_{args.dataset}{suffix}"
        output_path_pdf = args.output_dir / f"{filename_base}.pdf"
        output_path_png = args.output_dir / f"{filename_base}.png"

    xlim = parse_limits_arg(args.xlim)
    ylim = parse_limits_arg(args.ylim)
    fig_size = parse_figsize_arg(args.fig_size)

    methods = sorted(df_plot['Method'].unique())
    method_props = build_method_properties(methods)
    # Parse legend order if provided
    legend_order = None
    if args.legend_order:
        parts = [p for chunk in args.legend_order.split(',') for p in chunk.split()]
        legend_order = [p.strip() for p in parts if p.strip()]

    # Generate PDF
    plot_generic_comparison(
        df_plot=df_plot,
        output_path=str(output_path_pdf),
        x_scale=args.x_scale,
        y_scale=args.y_scale,
        show_markers=show_markers,
        error_type=args.error_type,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        method_properties=method_props,   
        legend_order=legend_order,
        xlim=xlim, ylim=ylim,            
        x_margin=args.x_margin, y_margin=args.y_margin, min_decades=args.min_decades,
        x_start_pad_ratio=args.x_start_pad_ratio, y_start_pad_ratio=args.y_start_pad_ratio,
        title_fontsize=args.title_fontsize, label_fontsize=args.label_fontsize,
        tick_fontsize=args.tick_fontsize, legend_fontsize=args.legend_fontsize,
        fig_size=fig_size,
        x_tick_step=args.x_tick_step,
        allow_negative_x_pad=args.allow_negative_x_pad,
        allow_negative_y_pad=args.allow_negative_y_pad,
        line_width=args.line_width
    )

    # Generate PNG
    plot_generic_comparison(
        df_plot=df_plot,
        output_path=str(output_path_png),
        x_scale=args.x_scale,
        y_scale=args.y_scale,
        show_markers=show_markers,
        error_type=args.error_type,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        method_properties=method_props,   
        legend_order=legend_order,
        xlim=xlim, ylim=ylim,            
        x_margin=args.x_margin, y_margin=args.y_margin, min_decades=args.min_decades,
        x_start_pad_ratio=args.x_start_pad_ratio, y_start_pad_ratio=args.y_start_pad_ratio,
        title_fontsize=args.title_fontsize, label_fontsize=args.label_fontsize,
        tick_fontsize=args.tick_fontsize, legend_fontsize=args.legend_fontsize,
        fig_size=fig_size,
        x_tick_step=args.x_tick_step,
        allow_negative_x_pad=args.allow_negative_x_pad,
        allow_negative_y_pad=args.allow_negative_y_pad,
        line_width=args.line_width
    )

    # Generate individual method plots if requested
    if args.generate_individual_method_plots:
        generate_individual_method_plots(
            df_original=df,
            methods=args.methods,
            output_dir=args.output_dir,
            metric=args.metric,
            max_n=args.max_n,
            use_all_trials=args.use_all_trials,
            x_scale=args.x_scale,
            y_scale=args.y_scale,
            show_markers=show_markers,
            error_type=args.error_type,
            dataset=args.dataset,
            # xlim=parse_limits_arg('100,120'), ylim=parse_limits_arg('0.001,10'),
            xlim=xlim, ylim=ylim,
            x_margin=args.x_margin, y_margin=args.y_margin, 
            min_decades=args.min_decades,
            x_start_pad_ratio=args.x_start_pad_ratio, y_start_pad_ratio=args.y_start_pad_ratio,
            title_fontsize=args.title_fontsize, label_fontsize=args.label_fontsize,
            tick_fontsize=args.tick_fontsize, legend_fontsize=args.legend_fontsize,
            fig_size=fig_size,
            x_tick_step=args.x_tick_step
        )

    # Generate a single multi-page PDF for selected sessions (per session across methods)
    sessions_arg = (args.sessions or "").strip()
    if sessions_arg and sessions_arg.lower() not in {"none", "all"}:
        parts = [p for chunk in sessions_arg.split(',') for p in chunk.split()]
        selected_sessions = [p.strip() for p in parts if p.strip()]
        out_pdf = args.output_dir / f"{args.sessions_output}.pdf"
        generate_selected_sessions_pdf(
            df=df,
            sessions=selected_sessions,
            methods=args.methods,
            output_pdf=out_pdf,
            metric=args.metric,
            max_n=args.max_n,
            use_all_trials=args.use_all_trials,
            x_scale=args.x_scale,
            y_scale=args.y_scale,
            show_markers=show_markers,
            dataset=args.dataset,
            xlim=xlim,
            ylim=ylim,
            x_margin=args.x_margin,
            y_margin=args.y_margin,
            min_decades=args.min_decades,
            x_start_pad_ratio=args.x_start_pad_ratio,
            y_start_pad_ratio=args.y_start_pad_ratio,
            title_fontsize=args.title_fontsize,
            label_fontsize=args.label_fontsize,
            tick_fontsize=args.tick_fontsize,
            legend_fontsize=args.legend_fontsize,
            fig_size=fig_size,
            x_tick_step=args.x_tick_step,
            allow_negative_x_pad=args.allow_negative_x_pad,
            allow_negative_y_pad=args.allow_negative_y_pad,
            line_width=args.line_width,
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
