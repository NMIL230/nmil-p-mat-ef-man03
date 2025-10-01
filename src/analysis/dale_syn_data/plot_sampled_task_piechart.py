#!/usr/bin/env python3
"""
Script to analyze task sampling distribution across DALE sessions and generate charts.

This script reads IMLE metrics data from each session's final iteration and creates
either pie charts or bar charts showing the distribution of sampled tasks across all sessions.
Each chart represents one session and shows how many times each task type was sampled.

Task types are grouped according to SUMMARIZED_METRICS:
1. CorsiComplex - combines all CorsiComplex_correct_w_len_* metrics
2. SimpleSpan - combines all SimpleSpan_correct_w_len_* metrics  
3. Countermanding_reaction_time
4. D2_hit_accuracy
5. PasatPlus_correctly_answered
6. RunningSpan_correct_w_len_2
7. RunningSpan_correct_w_len_3
8. Stroop_reaction_time

Outputs:
- One pie chart per session (showing task sampling distribution)
- All charts aggregated into a single PDF file
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Import from existing codebase
from utils.data_distribution_utils import (
    RELEVANT_METRICS,
)

# Individual metrics with detailed grouping for span tasks - all dark colors
INDIVIDUAL_METRICS = {
    # CorsiComplex subtasks - dark blue shades
    'CorsiComplex_len_2': ('#0d47a1', 'CorsiComplex_correct_w_len_2'),    # Dark blue
    'CorsiComplex_len_3': ('#1565c0', 'CorsiComplex_correct_w_len_3'),    # 
    'CorsiComplex_len_4': ('#1976d2', 'CorsiComplex_correct_w_len_4'),    # 
    'CorsiComplex_len_5': ('#1e88e5', 'CorsiComplex_correct_w_len_5'),    # 
    'CorsiComplex_len_6': ('#2196f3', 'CorsiComplex_correct_w_len_6'),    # 
    'CorsiComplex_len_7': ('#42a5f5', 'CorsiComplex_correct_w_len_7'),    # 
    'CorsiComplex_len_8': ('#64b5f6', 'CorsiComplex_correct_w_len_8'),    # 
    'CorsiComplex_len_9': ('#90caf9', 'CorsiComplex_correct_w_len_9'),    # 
    'CorsiComplex_len_10': ('#bbdefb', 'CorsiComplex_correct_w_len_10'),  # Light blue but still visible
    
    # SimpleSpan subtasks - dark orange/red shades
    'SimpleSpan_len_2': ('#bf360c', 'SimpleSpan_correct_w_len_2'),        # Dark red-orange
    'SimpleSpan_len_3': ('#d84315', 'SimpleSpan_correct_w_len_3'),        # 
    'SimpleSpan_len_4': ('#e64a19', 'SimpleSpan_correct_w_len_4'),        # 
    'SimpleSpan_len_5': ('#f57c00', 'SimpleSpan_correct_w_len_5'),        # 
    'SimpleSpan_len_6': ('#ff9800', 'SimpleSpan_correct_w_len_6'),        # 
    'SimpleSpan_len_7': ('#ffb74d', 'SimpleSpan_correct_w_len_7'),        # 
    'SimpleSpan_len_8': ('#ffcc02', 'SimpleSpan_correct_w_len_8'),        # 
    'SimpleSpan_len_9': ('#ffd54f', 'SimpleSpan_correct_w_len_9'),        # 
    'SimpleSpan_len_10': ('#ffe082', 'SimpleSpan_correct_w_len_10'),      # Light orange but still visible
    
    # RunningSpan subtasks - dark purple/pink shades
    'RunningSpan_len_2': ('#6a1b9a', 'RunningSpan_correct_w_len_2'),      # Dark purple
    'RunningSpan_len_3': ('#8e24aa', 'RunningSpan_correct_w_len_3'),      # Medium purple
    
    # Single metrics - all dark colors
    'Countermanding_reaction_time': ('#2e7d32', 'Countermanding_reaction_time'),  # Dark green
    'D2_hit_accuracy': ('#c62828', 'D2_hit_accuracy'),                           # Dark red  
    'PasatPlus_correctly_answered': ('#5e35b1', 'PasatPlus_correctly_answered'), # Dark purple
    'Stroop_reaction_time': ('#424242', 'Stroop_reaction_time')                  # Dark gray
}

# Mapping from original metrics to display keys
METRIC_TO_DISPLAY = {}
for key, (color, original_metric) in INDIVIDUAL_METRICS.items():
    METRIC_TO_DISPLAY[original_metric] = key


def is_excluded_session(session_id: str) -> bool:
    """Check if a session should be excluded from analysis."""
    excluded_keywords = ['aggregate', 'summary', 'combined']
    return any(keyword in session_id.lower() for keyword in excluded_keywords)


def find_latest_data_file(data_dir: Path) -> Optional[Path]:
    """
    Find the latest (highest iteration number) data file in a session's data directory.
    
    Args:
        data_dir: Path to the session's data directory
        
    Returns:
        Path to the latest data file, or None if no valid files found
    """
    if not data_dir.exists():
        return None
        
    data_files = []
    for file_path in data_dir.glob("num_tests_run_*_update_w_data_session_*.pt"):
        try:
            parts = file_path.stem.split('_')
            iteration_idx = None
            for i, part in enumerate(parts):
                if part == "run" and i + 1 < len(parts):
                    iteration_num = int(parts[i + 1])
                    iteration_idx = iteration_num
                    break
            
            if iteration_idx is not None:
                data_files.append((iteration_idx, file_path))
        except (ValueError, IndexError):
            continue
    
    if not data_files:
        return None
        
    # Sort by iteration and return the latest one
    data_files.sort(key=lambda x: x[0])
    return data_files[-1][1]


def extract_task_sampling_counts(imle_metrics_data: Dict) -> Dict[str, int]:
    """
    Extract task sampling counts from IMLE metrics data.
    
    Args:
        imle_metrics_data: Dictionary with metric names as keys and lists of values
        
    Returns:
        Dictionary mapping display task types to their sampling counts
    """
    task_counts = {}
    
    # Count samples for each individual metric
    for original_metric, samples in imle_metrics_data.items():
        if original_metric in METRIC_TO_DISPLAY:
            display_key = METRIC_TO_DISPLAY[original_metric]
            task_counts[display_key] = len(samples)
        
    return task_counts


def create_pie_chart(task_counts: Dict[str, int], session_id: str, ax: plt.Axes) -> None:
    """
    Create a pie chart for a single session's task sampling distribution.
    
    Args:
        task_counts: Dictionary mapping task types to counts
        session_id: Session identifier for the title
        ax: Matplotlib axes to plot on
    """
    # Filter out tasks with zero samples
    non_zero_counts = {task: count for task, count in task_counts.items() if count > 0}
    
    if not non_zero_counts:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{session_id}\n(No samples)', fontsize=11, fontweight='bold')
        return
    
    # Prepare data for pie chart
    labels = list(non_zero_counts.keys())
    sizes = list(non_zero_counts.values())
    colors = [INDIVIDUAL_METRICS[label][0] for label in labels]
    
    # Create pie chart without labels (they'll go in the legend)
    _, _, autotexts = ax.pie(
        sizes, 
        colors=colors, 
        autopct='%1.0f',
        startangle=90,
        textprops={'fontsize': 9, 'fontweight': 'bold'}
    )
    
    # Improve text formatting
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    
    total_samples = sum(sizes)
    ax.set_title(f'{session_id}\n({total_samples} samples)', fontsize=11, fontweight='bold', pad=15)
    
    # Add border around the pie chart
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor('black')


def create_bar_chart(task_counts: Dict[str, int], session_id: str, ax: plt.Axes) -> None:
    """
    Create a bar chart for a single session's task sampling distribution.
    
    Args:
        task_counts: Dictionary mapping task types to counts
        session_id: Session identifier for the title
        ax: Matplotlib axes to plot on
    """
    # Filter out tasks with zero samples
    non_zero_counts = {task: count for task, count in task_counts.items() if count > 0}

    if not non_zero_counts:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{session_id}\n(No samples)', fontsize=11, fontweight='bold')
        return

    # Use the order defined in INDIVIDUAL_METRICS for consistent display
    ordered_tasks = [t for t in INDIVIDUAL_METRICS.keys() if t in non_zero_counts]
    counts = [non_zero_counts[t] for t in ordered_tasks]
    colors = [INDIVIDUAL_METRICS[t][0] for t in ordered_tasks]

    ax.bar(range(len(ordered_tasks)), counts, color=colors, edgecolor='black', linewidth=0.7)
    ax.set_xticks(range(len(ordered_tasks)))
    # Cleaner x labels
    display_labels = []
    for t in ordered_tasks:
        if 'CorsiComplex' in t:
            display_labels.append(t.replace('CorsiComplex_', 'Corsi ').replace('len_', 'L'))
        elif 'SimpleSpan' in t:
            display_labels.append(t.replace('SimpleSpan_', 'Simple ').replace('len_', 'L'))
        elif 'RunningSpan' in t:
            display_labels.append(t.replace('RunningSpan_', 'Running ').replace('len_', 'L'))
        else:
            display_labels.append(t.replace('_', ' ').title())
    ax.set_xticklabels(display_labels, rotation=60, ha='right', fontsize=8)
    ax.tick_params(axis='y', labelsize=9)

    total_samples = sum(counts)
    ax.set_title(f'{session_id}\n({total_samples} samples)', fontsize=11, fontweight='bold', pad=10)
    ax.set_ylabel('Count', fontsize=10)

    # Add border around the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor('black')


def analyze_dale_sessions(dale_dir: Path) -> Dict[str, Dict[str, int]]:
    """
    Analyze all sessions in a DALE directory and extract task sampling distributions.
    
    Args:
        dale_dir: Path to DALE results directory
        
    Returns:
        Dictionary mapping session IDs to their task sampling counts
    """
    if not dale_dir.exists():
        raise ValueError(f"DALE directory does not exist: {dale_dir}")
    
    session_results = {}
    session_dirs = [d for d in dale_dir.iterdir() if d.is_dir() and not is_excluded_session(d.name)]
    
    logger.info(f"Processing {len(session_dirs)} sessions from {dale_dir}")
    
    for session_dir in session_dirs:
        session_id = session_dir.name
        
        data_dir = session_dir / "data"
        if not data_dir.exists():
            logger.warning(f"No data directory found for session {session_id}")
            continue
        
        # Find the latest data file
        latest_data_file = find_latest_data_file(data_dir)
        if latest_data_file is None:
            logger.warning(f"No valid data files found for session {session_id}")
            continue
        
        try:
            # Load IMLE metrics data
            imle_metrics_data = torch.load(latest_data_file)
            
            # Extract task sampling counts
            task_counts = extract_task_sampling_counts(imle_metrics_data)
            
            # Only include sessions with actual samples
            if any(count > 0 for count in task_counts.values()):
                session_results[session_id] = task_counts
                logger.debug(f"Processed session {session_id}: {sum(task_counts.values())} total samples")
            else:
                logger.warning(f"Session {session_id} has no samples")
                
        except Exception as e:
            logger.error(f"Error processing session {session_id}: {e}")
            continue
    
    logger.info(f"Successfully processed {len(session_results)} sessions")
    return session_results


def create_unified_legend(all_task_counts: Dict[str, Dict[str, int]], fig: plt.Figure) -> None:
    """
    Create a unified legend showing all task types with colors and counts.
    
    Args:
        all_task_counts: Dictionary of all session task counts
        fig: The figure to add legend to
    """
    # Collect all task types that appear in any session
    all_tasks = set()
    for session_counts in all_task_counts.values():
        all_tasks.update(task for task, count in session_counts.items() if count > 0)
    
    # Sort tasks by INDIVIDUAL_METRICS declaration order for natural L2..L10
    base_order = list(INDIVIDUAL_METRICS.keys())
    sorted_tasks = [t for t in base_order if t in all_tasks]
    
    # Create legend elements
    legend_elements = []
    for task in sorted_tasks:
        color, original_metric = INDIVIDUAL_METRICS[task]
        
        # Create cleaner display names
        if 'CorsiComplex' in task:
            display_name = task.replace('CorsiComplex_', 'Corsi ').replace('len_', 'L')
        elif 'SimpleSpan' in task:
            display_name = task.replace('SimpleSpan_', 'Simple ').replace('len_', 'L')
        elif 'RunningSpan' in task:
            display_name = task.replace('RunningSpan_', 'Running ').replace('len_', 'L')
        else:
            display_name = task.replace('_', ' ').title()
        
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=display_name))
    
    # Add legend to the right side of the figure
    fig.legend(handles=legend_elements, 
              loc='center right', 
              bbox_to_anchor=(0.98, 0.5),
              fontsize=10,
              title='Task Types',
              title_fontsize=12,
              frameon=True,
              fancybox=True,
              shadow=True)


def _natural_sort_key(s: str):
    """Natural sort key to sort strings with embedded numbers by numeric order."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def create_session_pie_charts(session_results: Dict[str, Dict[str, int]], output_path: Path, page_title: str = 'Task Sampling Distribution', chart_type: str = 'bar') -> None:
    """
    Create charts for all sessions and save to a multi-page PDF.
    
    Args:
        session_results: Dictionary mapping session IDs to task counts
        output_path: Path to save the PDF file
        page_title: Custom title for each page
    """
    if not session_results:
        logger.warning("No session results to plot")
        return
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Calculate layout (2x4 grid per page = 8 charts per page)
    # Ensure consistent ordering by session id (natural sort)
    sessions = sorted(session_results.keys(), key=_natural_sort_key)
    charts_per_page = 8
    rows_per_page = 2
    cols_per_page = 4
    
    with PdfPages(output_path) as pdf:
        for page_start in range(0, len(sessions), charts_per_page):
            page_sessions = sessions[page_start:page_start + charts_per_page]
            
            # Create figure for this page with space for legend
            fig = plt.figure(figsize=(20, 12))
            
            # Create gridspec to make room for legend
            gs = fig.add_gridspec(rows_per_page, cols_per_page, 
                                left=0.05, right=0.82, top=0.88, bottom=0.1,
                                hspace=0.4, wspace=0.3)
            
            # Create charts for sessions on this page
            for i, session_id in enumerate(page_sessions):
                row = i // cols_per_page
                col = i % cols_per_page
                ax = fig.add_subplot(gs[row, col])
                
                task_counts = session_results[session_id]
                if chart_type == 'pie':
                    create_pie_chart(task_counts, session_id, ax)
                else:
                    create_bar_chart(task_counts, session_id, ax)
                
                # Add border around each subplot
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(2)
                    spine.set_edgecolor('darkgray')
            
            # Create unified legend
            create_unified_legend(session_results, fig)
            
            # Add page title with more spacing
            page_num = (page_start // charts_per_page) + 1
            total_pages = (len(sessions) - 1) // charts_per_page + 1
            fig.suptitle(
                f'{page_title} - Page {page_num}/{total_pages}',
                fontsize=18, 
                fontweight='bold',
                y=0.95
            )
            
            # Save page to PDF
            pdf.savefig(fig, dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    logger.info(f"Saved {len(sessions)} {chart_type} charts to {output_path}")


def aggregate_results_by_method(all_results: Dict[str, Dict[str, Dict[str, int]]], method_mapping: Dict[str, str]) -> Dict[str, Dict[str, int]]:
    """
    Aggregate task sampling results by method.
    
    Args:
        all_results: Dictionary mapping result_name to session_results
        method_mapping: Dictionary mapping result_name to method description
        
    Returns:
        Dictionary mapping method description to aggregated task counts
    """
    method_aggregates = {}
    
    for result_name, session_results in all_results.items():
        if result_name not in method_mapping:
            continue
            
        method = method_mapping[result_name]
        
        if method not in method_aggregates:
            method_aggregates[method] = {}
        
        # Aggregate all sessions for this method
        for session_counts in session_results.values():
            for task_type, count in session_counts.items():
                method_aggregates[method][task_type] = method_aggregates[method].get(task_type, 0) + count
    
    return method_aggregates


def print_method_comparison_table(method_aggregates: Dict[str, Dict[str, int]]) -> None:
    """Print a comprehensive table comparing task sampling across methods."""
    if not method_aggregates:
        print("No method aggregates to display")
        return
    
    print("\n" + "="*120)
    print("TASK SAMPLING COMPARISON BY METHOD")
    print("="*120)
    
    # Get all unique tasks across all methods
    all_tasks = set()
    for method_counts in method_aggregates.values():
        all_tasks.update(method_counts.keys())
    
    # Sort tasks by category for better organization
    corsi_tasks = sorted([t for t in all_tasks if 'CorsiComplex' in t])
    simple_tasks = sorted([t for t in all_tasks if 'SimpleSpan' in t])
    running_tasks = sorted([t for t in all_tasks if 'RunningSpan' in t])
    other_tasks = sorted([t for t in all_tasks if not any(x in t for x in ['CorsiComplex', 'SimpleSpan', 'RunningSpan'])])
    
    ordered_tasks = corsi_tasks + simple_tasks + running_tasks + other_tasks
    methods = sorted(method_aggregates.keys())
    
    # Calculate totals for each method
    method_totals = {}
    for method in methods:
        method_totals[method] = sum(method_aggregates[method].values())
    
    # Print header
    print(f"{'Task Type':<30}", end="")
    for method in methods:
        print(f"{method:<25}", end="")
    print()
    print("-" * 120)
    
    # Print each task category
    current_category = None
    for task in ordered_tasks:
        # Add category headers
        task_category = None
        if 'CorsiComplex' in task:
            task_category = "CorsiComplex"
        elif 'SimpleSpan' in task:
            task_category = "SimpleSpan"
        elif 'RunningSpan' in task:
            task_category = "RunningSpan"
        else:
            task_category = "Other"
        
        if task_category != current_category:
            if current_category is not None:
                print()  # Add blank line between categories
            print(f"{task_category} Tasks:")
            print("-" * 120)
            current_category = task_category
        
        # Format task name for display
        display_name = task.replace('CorsiComplex_len_', 'Corsi L').replace('SimpleSpan_len_', 'Simple L').replace('RunningSpan_len_', 'Running L').replace('_', ' ')
        print(f"  {display_name:<28}", end="")
        
        # Print counts and percentages for each method
        for method in methods:
            count = method_aggregates[method].get(task, 0)
            percentage = (count / method_totals[method] * 100) if method_totals[method] > 0 else 0
            print(f"{count:>6} ({percentage:>5.1f}%)  ", end="")
        print()
    
    # Print totals
    print()
    print("=" * 120)
    print(f"{'TOTALS':<30}", end="")
    for method in methods:
        total = method_totals[method]
        print(f"{total:>6} (100.0%)  ", end="")
    print()
    print("=" * 120)
    
    # Print method summary
    print(f"\nMETHOD SUMMARY:")
    for method in methods:
        total = method_totals[method]
        num_sessions = len([s for s in method_aggregates[method].keys() if method_aggregates[method][s] > 0])
        print(f"  {method}: {total} total samples")


def print_summary_statistics(session_results: Dict[str, Dict[str, int]]) -> None:
    """Print summary statistics about task sampling across all sessions."""
    if not session_results:
        logger.warning("No session results for summary")
        return
    
    print("\n" + "="*60)
    print("TASK SAMPLING SUMMARY")
    print("="*60)
    
    # Calculate aggregate statistics for individual metrics
    total_task_counts = {}
    for session_counts in session_results.values():
        for task_type, count in session_counts.items():
            total_task_counts[task_type] = total_task_counts.get(task_type, 0) + count
    
    total_samples = sum(total_task_counts.values())
    
    print(f"Total sessions analyzed: {len(session_results)}")
    print(f"Total samples across all sessions: {total_samples}")
    print(f"\nDetailed task sampling distribution:")
    
    # Group by task families for cleaner display
    corsi_tasks = [(k, v) for k, v in total_task_counts.items() if 'CorsiComplex' in k]
    simple_tasks = [(k, v) for k, v in total_task_counts.items() if 'SimpleSpan' in k]
    running_tasks = [(k, v) for k, v in total_task_counts.items() if 'RunningSpan' in k]
    other_tasks = [(k, v) for k, v in total_task_counts.items() if not any(x in k for x in ['CorsiComplex', 'SimpleSpan', 'RunningSpan'])]
    
    def print_task_group(tasks, group_name):
        if not tasks:
            return
        print(f"\n  {group_name}:")
        tasks.sort(key=lambda x: x[1], reverse=True)
        for task_type, total_count in tasks:
            percentage = (total_count / total_samples * 100) if total_samples > 0 else 0
            display_name = task_type.replace('_len_', ' L').replace('_', ' ')
            print(f"    {display_name:25s}: {total_count:6d} samples ({percentage:5.1f}%)")
    
    print_task_group(corsi_tasks, "CorsiComplex Tasks")
    print_task_group(simple_tasks, "SimpleSpan Tasks")
    print_task_group(running_tasks, "RunningSpan Tasks")
    print_task_group(other_tasks, "Other Tasks")
    
    # Session-level statistics
    samples_per_session = [sum(counts.values()) for counts in session_results.values()]
    if samples_per_session:
        print(f"\nSamples per session:")
        print(f"  Mean: {np.mean(samples_per_session):.1f}")
        print(f"  Std:  {np.std(samples_per_session):.1f}")
        print(f"  Min:  {np.min(samples_per_session)}")
        print(f"  Max:  {np.max(samples_per_session)}")


def main():
    """Main function to run the task sampling analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze task sampling distribution in DALE sessions and create charts (pie or bar)."
    )
    
    # Default paths based on existing structure
    script_root = Path(__file__).resolve().parent
    # Parent results directory that contains multiple experiment subfolders
    default_result_dir = script_root / "result" / "re-2d-3d-full-final"
    default_output_dir = script_root / "plot" / "2d-3d-240-session-bar-chart"
    
    parser.add_argument(
        "--resultdir", "--reusltdir",  # support common typo as alias
        dest="resultdir",
        type=Path,
        default=default_result_dir,
        help="Path to parent result directory containing experiment subfolders"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=default_output_dir, 
        help="Directory to save output files"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="task_sampling_piecharts.pdf",
        help="Name of output PDF file"
    )
    parser.add_argument(
        "--page_title",
        type=str,
        default="Sampled-tasks-all-session-hound-charge-1105-3d",
        help="Custom title for each page of the PDF"
    )
    parser.add_argument(
        "--chart_type",
        type=str,
        choices=["bar", "pie"],
        default="bar",
        help="Type of chart to generate for each session"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Result directory: {args.resultdir}")
    print(f"Output directory: {args.output_dir}")
    
    try:
        # Validate result directory
        if not args.resultdir.exists() or not args.resultdir.is_dir():
            raise ValueError(f"Result directory does not exist or is not a directory: {args.resultdir}")

        # Discover experiment subfolders (e.g., exp_c1_3d_dale_ps0)
        exp_dirs = [d for d in args.resultdir.iterdir() if d.is_dir() and not is_excluded_session(d.name)]
        if not exp_dirs:
            logger.warning(f"No experiment subfolders found under: {args.resultdir}")
            return 0

        # Process each experiment folder
        for exp_dir in sorted(exp_dirs, key=lambda p: p.name):
            exp_name = exp_dir.name

            print(f"\nProcessing experiment: {exp_name} ...")

            # Analyze sessions for this experiment folder
            logger.info(f"Starting task sampling analysis for {exp_name}...")
            session_results = analyze_dale_sessions(exp_dir)

            if not session_results:
                logger.warning(f"No valid sessions found for {exp_name}")
                continue

            # Generate output filename per experiment based on chart type
            suffix = "barcharts" if args.chart_type == "bar" else "piecharts"
            output_filename = f"{exp_name}_sampled_task_{suffix}.pdf"
            output_path = args.output_dir / output_filename

            # Create charts
            page_title = f"Sampled-tasks-all-session-{exp_name}"
            logger.info(f"Creating {args.chart_type} charts for {exp_name}...")
            create_session_pie_charts(session_results, output_path, page_title, chart_type=args.chart_type)

            # Print summary statistics
            print(f"\nSummary for {exp_name}:")
            print_summary_statistics(session_results)

            print(f"Results saved to: {output_path}")

        print(f"\nAll analyses complete! Results saved to: {args.output_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
