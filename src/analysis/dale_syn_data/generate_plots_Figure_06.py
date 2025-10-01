#!/usr/bin/env python3
"""
Compare sampled-task distributions across methods for each session using stacked bar charts.

For each session, draw 4 stacked bars (methods: ps0, ps2, ps4, random). Each bar is stacked
by task type using the existing color palette, scaled to a fixed total height of 240 (i.e.,
proportions per method). Generate two PDFs:
- 3D group: exp_c1_3d_dale_ps0, exp_c2_3d_dale_ps2, exp_c3_3d_dale_ps4, exp_c4_3d_random
- 2D group: exp_c5_2d_dale_ps0, exp_c6_2d_dale_ps2, exp_c7_2d_dale_ps4, exp_c8_2d_random

Each PDF contains 88 charts (one per session), sorted by session id, with a unified legend.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, List, Set
import re

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns


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


# Individual metrics with detailed grouping for span tasks - all dark colors
INDIVIDUAL_METRICS = {
    # CorsiComplex subtasks - dark blue shades
    'CorsiComplex_len_2': ('#0d47a1', 'CorsiComplex_correct_w_len_2'),
    'CorsiComplex_len_3': ('#1565c0', 'CorsiComplex_correct_w_len_3'),
    'CorsiComplex_len_4': ('#1976d2', 'CorsiComplex_correct_w_len_4'),
    'CorsiComplex_len_5': ('#1e88e5', 'CorsiComplex_correct_w_len_5'),
    'CorsiComplex_len_6': ('#2196f3', 'CorsiComplex_correct_w_len_6'),
    'CorsiComplex_len_7': ('#42a5f5', 'CorsiComplex_correct_w_len_7'),
    'CorsiComplex_len_8': ('#64b5f6', 'CorsiComplex_correct_w_len_8'),
    'CorsiComplex_len_9': ('#90caf9', 'CorsiComplex_correct_w_len_9'),
    'CorsiComplex_len_10': ('#bbdefb', 'CorsiComplex_correct_w_len_10'),

    # SimpleSpan subtasks - dark orange/red shades
    'SimpleSpan_len_2': ('#bf360c', 'SimpleSpan_correct_w_len_2'),
    'SimpleSpan_len_3': ('#d84315', 'SimpleSpan_correct_w_len_3'),
    'SimpleSpan_len_4': ('#e64a19', 'SimpleSpan_correct_w_len_4'),
    'SimpleSpan_len_5': ('#f57c00', 'SimpleSpan_correct_w_len_5'),
    'SimpleSpan_len_6': ('#ff9800', 'SimpleSpan_correct_w_len_6'),
    'SimpleSpan_len_7': ('#ffb74d', 'SimpleSpan_correct_w_len_7'),
    'SimpleSpan_len_8': ('#ffcc02', 'SimpleSpan_correct_w_len_8'),
    'SimpleSpan_len_9': ('#ffd54f', 'SimpleSpan_correct_w_len_9'),
    'SimpleSpan_len_10': ('#ffe082', 'SimpleSpan_correct_w_len_10'),

    # RunningSpan subtasks - dark purple/pink shades
    'RunningSpan_len_2': ('#6a1b9a', 'RunningSpan_correct_w_len_2'),
    'RunningSpan_len_3': ('#8e24aa', 'RunningSpan_correct_w_len_3'),

    # Single metrics - all dark colors
    'Countermanding_reaction_time': ('#2e7d32', 'Countermanding_reaction_time'),
    'D2_hit_accuracy': ('#c62828', 'D2_hit_accuracy'),
    'PasatPlus_correctly_answered': ('#5e35b1', 'PasatPlus_correctly_answered'),
    'Stroop_reaction_time': ('#424242', 'Stroop_reaction_time')
}

# Mapping from original metrics to display keys
METRIC_TO_DISPLAY = {v[1]: k for k, v in INDIVIDUAL_METRICS.items()}


def is_excluded_session(session_id: str) -> bool:
    excluded_keywords = ['aggregate', 'summary', 'combined']
    return any(keyword in session_id.lower() for keyword in excluded_keywords)


def _natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def find_latest_data_file(data_dir: Path) -> Optional[Path]:
    if not data_dir.exists():
        return None
    data_files = []
    for file_path in data_dir.glob("num_tests_run_*_update_w_data_session_*.pt"):
        try:
            parts = file_path.stem.split('_')
            iteration_idx = None
            for i, part in enumerate(parts):
                if part == "run" and i + 1 < len(parts):
                    iteration_idx = int(parts[i + 1])
                    break
            if iteration_idx is not None:
                data_files.append((iteration_idx, file_path))
        except (ValueError, IndexError):
            continue
    if not data_files:
        return None
    data_files.sort(key=lambda x: x[0])
    return data_files[-1][1]


def extract_task_sampling_counts(imle_metrics_data: Dict) -> Dict[str, int]:
    task_counts = {}
    for original_metric, samples in imle_metrics_data.items():
        if original_metric in METRIC_TO_DISPLAY:
            display_key = METRIC_TO_DISPLAY[original_metric]
            task_counts[display_key] = len(samples)
    return task_counts


def analyze_sessions_for_experiment(exp_dir: Path) -> Dict[str, Dict[str, int]]:
    if not exp_dir.exists() or not exp_dir.is_dir():
        raise ValueError(f"Experiment directory not found: {exp_dir}")

    session_results: Dict[str, Dict[str, int]] = {}
    session_dirs = [d for d in exp_dir.iterdir() if d.is_dir() and not is_excluded_session(d.name)]
    logger.info(f"Processing {len(session_dirs)} sessions from {exp_dir.name}")

    for session_dir in session_dirs:
        session_id = session_dir.name
        data_dir = session_dir / "data"
        if not data_dir.exists():
            logger.warning(f"No data directory for session {session_id} in {exp_dir.name}")
            continue
        latest_data_file = find_latest_data_file(data_dir)
        if latest_data_file is None:
            logger.warning(f"No valid data files for session {session_id} in {exp_dir.name}")
            continue
        try:
            imle_metrics_data = torch.load(latest_data_file)
            task_counts = extract_task_sampling_counts(imle_metrics_data)
            session_results[session_id] = task_counts
        except Exception as e:
            logger.error(f"Error loading {latest_data_file} for session {session_id}: {e}")
            continue

    return session_results


def create_unified_legend(all_tasks: List[str], fig: plt.Figure, *, anchor_x: float = 0.88) -> None:
    # Sort tasks by INDIVIDUAL_METRICS declaration order for natural L2..L10
    base_order = list(INDIVIDUAL_METRICS.keys())
    present = set(all_tasks)
    sorted_tasks = [t for t in base_order if t in present]
    legend_elements = []
    for task in sorted_tasks:
        color, _ = INDIVIDUAL_METRICS[task]
        if task.startswith('CorsiComplex_len_'):
            n = task.split('_')[-1]
            display_name = f'Corsi Complex length {n}'
        elif task.startswith('SimpleSpan_len_'):
            n = task.split('_')[-1]
            display_name = f'Corsi Simple length {n}'
        elif 'RunningSpan' in task:
            # e.g., RunningSpan_len_2 -> Running Span length 2
            n = task.split('_')[-1]
            display_name = f'Running Span length {n}'
        else:
            display_name = task.replace('_', ' ').title()
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, label=display_name))

    fig.legend(
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(anchor_x, 0.5),
        fontsize=10,
        title='Task Types',
        title_fontsize=12,
        frameon=True,
        fancybox=False,
        shadow=False,
    )


def plot_session_method_comparison(
    session_id: str,
    method_order: List[str],
    method_session_counts: Dict[str, Dict[str, int]],
    ax: plt.Axes,
    target_total: int = 240,
    label_map: Optional[Dict[str, str]] = None,
) -> None:
    """Plot a single session's 4-method stacked bar comparison on the given axes."""
    # Determine tasks present across any method
    tasks_in_any = set()
    for m in method_order:
        tasks_in_any.update(task for task, cnt in method_session_counts.get(m, {}).items() if cnt > 0)

    # Order tasks by INDIVIDUAL_METRICS declaration order
    ordered_tasks = [t for t in INDIVIDUAL_METRICS.keys() if t in tasks_in_any]

    x = np.arange(len(method_order))
    bottoms = np.zeros(len(method_order), dtype=float)

    # For consistent colors and stacking, iterate tasks in order
    for task in ordered_tasks:
        color = INDIVIDUAL_METRICS[task][0]
        heights = []
        for m in method_order:
            counts = method_session_counts.get(m, {})
            total = sum(counts.values())
            c = counts.get(task, 0)
            if total > 0:
                heights.append((c / total) * target_total)
            else:
                heights.append(0.0)
        ax.bar(x, heights, bottom=bottoms, color=color, edgecolor='none', linewidth=0.0, width=0.7)
        bottoms += np.array(heights)

    # Axes formatting
    ax.set_xticks(x)
    # Short labels for methods
    short_labels = []
    for m in method_order:
        if label_map and m in label_map:
            short_labels.append(label_map[m])
            continue
        if 'ps0' in m:
            short_labels.append('ps0')
        elif 'ps2' in m:
            short_labels.append('DALE')
        elif 'ps4' in m:
            short_labels.append('ps4')
        elif m.upper() == 'TB':
            short_labels.append('TB')
        else:
            short_labels.append('random')
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_ylim(0, target_total)
    ax.set_yticks([0, target_total//4, target_total//2, 3*target_total//4, target_total])
    ax.set_ylabel(f'Count (scaled to {target_total})', fontsize=9)
    ax.set_title(session_id, fontsize=11, fontweight='bold', pad=8)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_edgecolor('black')


def create_group_pdf(
    group_name: str,
    exp_dirs: Dict[str, Path],
    output_path: Path,
    page_title: str,
    target_total: int = 240,
    session_filter: Optional[Set[str]] = None,
    method_order: Optional[List[str]] = None,
    label_map: Optional[Dict[str, str]] = None,
    session_order: Optional[List[str]] = None,
    save_png: bool = True,
) -> None:
    """Create a multi-page PDF of stacked bar comparisons for each session in the group."""
    # Helper to build synthetic TB distribution: target_total split across 8
    # top-level groups (2 span families + 6 fixed tasks). Span families split
    # their per-group allocation evenly across their lengths.
    def _build_tb_distribution() -> Dict[str, int]:
        counts: Dict[str, int] = {}
        def even_split(keys: List[str], total: int):
            if not keys:
                return
            base, rem = divmod(total, len(keys))
            ordered = sorted(keys, key=_natural_sort_key)
            for i, k in enumerate(ordered):
                counts[k] = counts.get(k, 0) + base + (1 if i < rem else 0)
        corsi = [k for k in INDIVIDUAL_METRICS if k.startswith('CorsiComplex_len_')]
        simple = [k for k in INDIVIDUAL_METRICS if k.startswith('SimpleSpan_len_')]
        # Split target_total across 8 top-level groups
        groups = 8
        g_base, g_rem = divmod(target_total, groups)
        group_alloc = [g_base + (1 if i < g_rem else 0) for i in range(groups)]
        # Group 0: Corsi family; Group 1: SimpleSpan family
        even_split(corsi, group_alloc[0])
        even_split(simple, group_alloc[1])
        fixed = [
            'RunningSpan_len_2', 'RunningSpan_len_3',
            'Countermanding_reaction_time', 'D2_hit_accuracy',
            'PasatPlus_correctly_answered', 'Stroop_reaction_time',
        ]
        # Groups 2..7: 6 fixed tasks
        for i, k in enumerate(fixed):
            counts[k] = counts.get(k, 0) + group_alloc[2 + i]
        return {k: v for k, v in counts.items() if v > 0}

    # Analyze methods present in exp_dirs and prepare container keyed by requested order
    method_session_results: Dict[str, Dict[str, Dict[str, int]]] = {}
    if method_order is None:
        method_order = list(exp_dirs.keys())
    for m in method_order:
        if m.upper() == 'TB':
            method_session_results[m] = {}
        else:
            exp_dir = exp_dirs.get(m)
            if exp_dir is None:
                raise ValueError(f"Missing experiment directory for method '{m}' in group {group_name}")
            logger.info(f"Analyzing {group_name} - {m} from {exp_dir.name}")
            method_session_results[m] = analyze_sessions_for_experiment(exp_dir)

    # Collect union of session ids across methods
    all_sessions: Set[str] = set()
    for m in method_session_results:
        all_sessions.update(method_session_results[m].keys())
    # Determine ordered sessions
    if session_order is not None:
        if all_sessions:
            sessions_sorted = [s for s in session_order if s in all_sessions]
        else:
            sessions_sorted = list(session_order)
    else:
        if session_filter is not None:
            filtered_sessions = all_sessions.intersection(session_filter)
            sessions_sorted = sorted(filtered_sessions, key=_natural_sort_key)
        else:
            sessions_sorted = sorted(all_sessions, key=_natural_sort_key)
    if not sessions_sorted:
        logger.info(f"{group_name}: no sessions to plot after filtering; skipping PDF generation")
        return

    # Populate TB counts for all sessions only now that we know which sessions to include
    if 'TB' in method_order:
        tb_counts = _build_tb_distribution()
        method_session_results['TB'] = {sid: tb_counts for sid in sessions_sorted}

    # Prepare plotting
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use('default')
    sns.set_palette("husl")

    # Layout: if we have a session filter, switch to portrait 3x3 per page
    if session_filter is not None:
        charts_per_page = 9
        rows_per_page = 3
        cols_per_page = 3
        figsize = (14, 16)  # portrait, wider to give legend room
        gridspec_kwargs = dict(left=0.06, right=0.70, top=0.90, bottom=0.08, hspace=0.50, wspace=0.35)
        legend_anchor_x = 0.74
    else:
        charts_per_page = 8
        rows_per_page = 2
        cols_per_page = 4
        figsize = (20, 12)  # landscape
        gridspec_kwargs = dict(left=0.05, right=0.80, top=0.88, bottom=0.10, hspace=0.40, wspace=0.30)
        legend_anchor_x = 0.84

    # Keep the requested order for plotting
    method_order = method_order

    # Gather tasks present across the entire group for legend
    tasks_for_legend = []
    for m in method_order:
        for counts in method_session_results[m].values():
            tasks_for_legend.extend([t for t, c in counts.items() if c > 0])

    # Derive base name for PNGs
    png_base = output_path.stem  # base filename without .pdf suffix
    png_dir = output_path.parent

    with PdfPages(output_path) as pdf:
        for page_start in range(0, len(sessions_sorted), charts_per_page):
            page_sessions = sessions_sorted[page_start:page_start + charts_per_page]

            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(rows_per_page, cols_per_page, **gridspec_kwargs)

            for i, session_id in enumerate(page_sessions):
                row = i // cols_per_page
                col = i % cols_per_page
                ax = fig.add_subplot(gs[row, col])

                # Collect per-method counts for this session
                method_counts_for_session: Dict[str, Dict[str, int]] = {}
                for m in method_order:
                    method_counts_for_session[m] = method_session_results[m].get(session_id, {})

                plot_session_method_comparison(
                    session_id,
                    method_order,
                    method_counts_for_session,
                    ax,
                    target_total=target_total,
                    label_map=label_map,
                )

                # Border
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(2)
                    spine.set_edgecolor('darkgray')

            create_unified_legend(tasks_for_legend, fig, anchor_x=legend_anchor_x)

            fig.suptitle(
                page_title,
                fontsize=18,
                fontweight='bold',
                y=0.95,
            )

            # Save PDF page
            pdf.savefig(fig, dpi=300, bbox_inches='tight')

            # Also save PNG per page with sequential numbering
            if save_png:
                try:
                    # Determine page number for filename
                    page_num = (page_start // charts_per_page) + 1
                    png_name = f"{png_base}_page_{page_num}.png"
                    png_path = png_dir / png_name
                    fig.savefig(png_path, dpi=300, bbox_inches='tight')
                except Exception as e:
                    logger.warning(f"Failed to save PNG for page starting at {page_start}: {e}")
            plt.close(fig)

    logger.info(f"Saved {len(sessions_sorted)} session comparison charts to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare sampled task distribution across methods per session using stacked bars."
    )
    script_root = Path(__file__).resolve().parent
    default_result_dir = script_root / "result" / "re-2d-3d-full-final"
    default_output_dir = script_root / "plot" / "Figure06_v04_SampleDistributions_A_100"

    parser.add_argument(
        "--resultdir", "--reusltdir",
        dest="resultdir",
        type=Path,
        default=default_result_dir,
        help="Path to parent result directory containing experiment subfolders",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=default_output_dir,
        help="Directory to save output PDFs",
    )
    parser.add_argument(
        "--target_total",
        type=int,
        default=100,
        help="Scale each method bar to this total to show proportions",
    )
    # New: optional session filter
    # default_sessions = [
    #     "LD2-041","LD2-037","LD2-021","LD2-047","LD2-045",
    #     "LD2-034","LD2-086","LD2-028","LD2-042",
    # ]

    #A-100
    default_sessions = [
        "LD2-069","LD2-029","LD2-027","LD2-039","LD2-083",
        "LD2-075","LD2-086","LD2-003","LD2-005",
    ]

    # B-30
    # default_sessions = [ 
    #     "LD2-034","LD2-008","LD2-009","LD2-020","LD2-086",
    #     "LD2-027","LD2-002","LD2-022","LD2-078",
    # ]
    parser.add_argument(
        "--sessions",
        type=str,
        default=",".join(default_sessions),
        help=(
            "Comma/space-separated session IDs to include. "
            "Use 'none' or 'all' to include all sessions. "
            "Default selects 9 LD2 sessions."
        ),
    )
    parser.add_argument(
        "--include_3d",
        action="store_true",
        help="Generate 3D group PDF",
    )
    parser.add_argument(
        "--include_2d",
        action="store_true",
        help="Generate 2D group PDF",
    )
    # Control which methods appear on x-axis (order preserved)
    parser.add_argument(
        "--methods_x",
        type=str,
        # Default order: TB, Random, DALE
        default="tb,random,ps2",
        help="Comma/space-separated methods on x-axis (order preserved). Supported: ps0, ps2 (DALE), ps4, random, tb",
    )
    parser.add_argument(
        "--no_png",
        action="store_true",
        help="Disable saving per-page PNGs alongside PDFs (PNG is enabled by default)",
    )

    args = parser.parse_args()

    # Parse session filter and preserve order if provided
    sessions_arg = (args.sessions or "").strip()
    session_filter: Optional[Set[str]]
    ordered_sessions: Optional[List[str]]
    if sessions_arg == "" or sessions_arg.lower() in {"none", "all"}:
        session_filter = None
        ordered_sessions = None
    else:
        parts = [p.strip() for chunk in sessions_arg.split(",") for p in chunk.split()]
        ordered_sessions = [p for p in parts if p]
        session_filter = set(ordered_sessions)

    # Parse x-axis methods
    mx = (args.methods_x or "").replace(" ", ",").lower()
    method_codes = [m for m in mx.split(',') if m]
    valid_codes = {"ps0", "ps2", "ps4", "random", "tb"}
    for m in method_codes:
        if m not in valid_codes:
            raise ValueError(f"Unsupported method '{m}' in --methods_x. Choose from {sorted(valid_codes)}")

    # Label map for ticks
    label_map = {
        'TB': 'TB',
        'ps0': 'ps0',
        'ps2': 'DALE',
        'ps4': 'ps4',
        'random': 'Random',
        '2d_dale_ps0': 'ps0', '2d_dale_ps2': 'DALE', '2d_dale_ps4': 'ps4', '2d_random': 'Random',
        '3d_dale_ps0': 'ps0', '3d_dale_ps2': 'DALE', '3d_dale_ps4': 'ps4', '3d_random': 'Random',
    }

    # Default: generate both if none specified; filtering may implicitly skip a group
    include_3d = args.include_3d or (not args.include_2d)
    include_2d = args.include_2d or (not args.include_3d)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_png = not args.no_png

    try:
        if not args.resultdir.exists() or not args.resultdir.is_dir():
            raise ValueError(f"Result directory does not exist or is not a directory: {args.resultdir}")

        # Resolve experiment directories
        exp_map = {p.name: p for p in args.resultdir.iterdir() if p.is_dir()}

        if include_3d:
            group_name = "3D"
            mapping = {
                'ps0': ('3d_dale_ps0', 'exp_c1_3d_dale_ps0'),
                'ps2': ('3d_dale_ps2', 'exp_c2_3d_dale_ps2'),
                'ps4': ('3d_dale_ps4', 'exp_c3_3d_dale_ps4'),
                'random': ('3d_random', 'exp_c4_3d_random'),
            }
            exp_dirs = {}
            for code in method_codes:
                if code == 'tb':
                    continue
                key, expname = mapping[code]
                exp_dirs[key] = exp_map.get(expname)
            missing = [k for k, v in exp_dirs.items() if v is None]
            if missing:
                raise ValueError(f"Missing expected 3D experiment(s): {missing}")
            method_order = [mapping[m][0] if m != 'tb' else 'TB' for m in method_codes]
            suffix = "_selected" if session_filter is not None else ""
            output_path = args.output_dir / f"compare_methods_{group_name}_sessions{suffix}.pdf"
            page_title = "TBD"
            create_group_pdf(
                group_name,
                exp_dirs,
                output_path,
                page_title,
                target_total=args.target_total,
                session_filter=session_filter,
                method_order=method_order,
                label_map=label_map,
                session_order=ordered_sessions,
                save_png=save_png,
            )

        if include_2d:
            group_name = "2D"
            mapping = {
                'ps0': ('2d_dale_ps0', 'exp_c5_2d_dale_ps0'),
                'ps2': ('2d_dale_ps2', 'exp_c6_2d_dale_ps2'),
                'ps4': ('2d_dale_ps4', 'exp_c7_2d_dale_ps4'),
                'random': ('2d_random', 'exp_c8_2d_random'),
            }
            exp_dirs = {}
            for code in method_codes:
                if code == 'tb':
                    continue
                key, expname = mapping[code]
                exp_dirs[key] = exp_map.get(expname)
            missing = [k for k, v in exp_dirs.items() if v is None]
            if missing:
                raise ValueError(f"Missing expected 2D experiment(s): {missing}")
            method_order = [mapping[m][0] if m != 'tb' else 'TB' for m in method_codes]
            suffix = "_selected" if session_filter is not None else ""
            output_path = args.output_dir / f"compare_methods_{group_name}_sessions{suffix}.pdf"
            page_title = "Tasks Sampled for 9 Representative Sessions"
            create_group_pdf(
                group_name,
                exp_dirs,
                output_path,
                page_title,
                target_total=args.target_total,
                session_filter=session_filter,
                method_order=method_order,
                label_map=label_map,
                session_order=ordered_sessions,
                save_png=save_png,
            )

        logger.info(f"All comparisons complete. PDFs saved to: {args.output_dir}")
        return 0
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
