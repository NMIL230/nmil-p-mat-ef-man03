#!/usr/bin/env python3
"""
Script to aggregate task sampling across sessions for each experiment and
save a single bar chart (PNG) per experiment.

This reuses session parsing from plot_sampled_task_piechart.py and
the INDIVIDUAL_METRICS color scheme for consistent visuals.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

# Project root to sys.path (for implicit namespace packages and utils)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Reuse helpers and color mapping from the piechart script
from analysis.dale_syn_data.plot_sampled_task_piechart import (
    INDIVIDUAL_METRICS,
    is_excluded_session,
    analyze_dale_sessions,
)


def aggregate_task_counts(session_results: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    """Aggregate task counts across all sessions for an experiment."""
    totals: Dict[str, int] = {}
    for counts in session_results.values():
        for task, c in counts.items():
            totals[task] = totals.get(task, 0) + int(c)
    return totals


def plot_barchart_for_experiment(exp_name: str, totals: Dict[str, int], output_path: Path, aggregate_spans: bool = True) -> None:
    """Create and save a bar chart PNG for an experiment's aggregated task counts."""
    # Optionally aggregate span families (CorsiComplex, SimpleSpan, RunningSpan)
    if aggregate_spans:
        family_totals: Dict[str, int] = {}
        # Sum families
        family_totals['CorsiComplex'] = sum(v for k, v in totals.items() if k.startswith('CorsiComplex_len_'))
        family_totals['SimpleSpan'] = sum(v for k, v in totals.items() if k.startswith('SimpleSpan_len_'))
        family_totals['RunningSpan'] = sum(v for k, v in totals.items() if k.startswith('RunningSpan_len_'))
        # Keep non-span metrics as-is
        for k, v in totals.items():
            if not (k.startswith('CorsiComplex_len_') or k.startswith('SimpleSpan_len_') or k.startswith('RunningSpan_len_')):
                family_totals[k] = v
        totals = family_totals

    # Filter zero counts and order by INDIVIDUAL_METRICS (or family order) for stable display
    if aggregate_spans:
        preferred_order = [
            'CorsiComplex', 'SimpleSpan', 'RunningSpan',
            'Countermanding_reaction_time', 'D2_hit_accuracy', 'PasatPlus_correctly_answered', 'Stroop_reaction_time'
        ]
        ordered_keys = [k for k in preferred_order if totals.get(k, 0) > 0]
        # Include any unexpected keys that have counts but weren't in preferred_order
        ordered_keys.extend([k for k in totals.keys() if k not in ordered_keys and totals.get(k, 0) > 0])
    else:
        ordered_keys = [k for k in INDIVIDUAL_METRICS.keys() if totals.get(k, 0) > 0]
    if not ordered_keys:
        logger.warning(f"No non-zero task counts for experiment {exp_name}; skipping plot.")
        return

    x = np.arange(len(ordered_keys))
    y = np.array([totals[k] for k in ordered_keys], dtype=int)
    if aggregate_spans:
        # Use representative dark colors for families
        family_colors = {
            'CorsiComplex': INDIVIDUAL_METRICS['CorsiComplex_len_2'][0],
            'SimpleSpan': INDIVIDUAL_METRICS['SimpleSpan_len_2'][0],
            'RunningSpan': INDIVIDUAL_METRICS['RunningSpan_len_2'][0],
        }
        colors = [
            family_colors.get(k, INDIVIDUAL_METRICS.get(k, ('#808080', None))[0])
            for k in ordered_keys
        ]
    else:
        colors = [INDIVIDUAL_METRICS[k][0] for k in ordered_keys]

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bar(x, y, color=colors, edgecolor='black', linewidth=1.0)

    # Format x labels (shorten long names)
    def fmt_label(k: str) -> str:
        if 'CorsiComplex' in k:
            return k.replace('CorsiComplex_', 'Corsi ').replace('len_', 'L')
        if 'SimpleSpan' in k:
            return k.replace('SimpleSpan_', 'Simple ').replace('len_', 'L')
        if 'RunningSpan' in k:
            return k.replace('RunningSpan_', 'Running ').replace('len_', 'L')
        return k.replace('_', ' ').title()

    ax.set_xticks(x)
    ax.set_xticklabels([fmt_label(k) for k in ordered_keys], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Samples (count)')
    ax.set_title(f'{exp_name} — Aggregated Sampled Tasks', fontsize=14, fontweight='bold')

    # Annotate bars with counts
    for xi, yi in zip(x, y):
        ax.text(xi, yi + max(y) * 0.01, str(int(yi)), ha='center', va='bottom', fontsize=9)

    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved barchart to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Create aggregated sampled-task bar charts (one PNG per experiment).'
    )

    script_root = Path(__file__).resolve().parent
    default_result_dir = script_root / 'result' / 're-2d-3d-full-final'
    default_output_dir = script_root / "plot" / "2d-3d-240-bar-chart"

    parser.add_argument(
        '--resultdir', '--reusltdir',
        dest='resultdir',
        type=Path,
        default=default_result_dir,
        help='Path to parent result directory containing experiment subfolders'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=default_output_dir,
        help='Directory to save PNGs'
    )
    # Whether to aggregate span families into single bars
    try:
        # Python 3.9+
        parser.add_argument(
            '--aggregate_spans',
            action=argparse.BooleanOptionalAction,
            default=True,
            help='Aggregate Corsi/Simple/Running spans into single bars (default: true)'
        )
    except AttributeError:
        # Fallback for older Python versions
        parser.add_argument('--aggregate_spans', dest='aggregate_spans', action='store_true', default=True,
                            help='Aggregate Corsi/Simple/Running spans into single bars (default: true)')
        parser.add_argument('--no-aggregate_spans', dest='aggregate_spans', action='store_false')

    args = parser.parse_args()

    if not args.resultdir.exists() or not args.resultdir.is_dir():
        raise ValueError(f'Result directory does not exist or is not a directory: {args.resultdir}')

    print(f"Result directory: {args.resultdir}")
    print(f"Output directory: {args.output_dir}")

    exp_dirs = [d for d in args.resultdir.iterdir() if d.is_dir() and not is_excluded_session(d.name)]
    if not exp_dirs:
        logger.warning(f'No experiment subfolders found under: {args.resultdir}')
        return 0

    for exp_dir in sorted(exp_dirs, key=lambda p: p.name):
        exp_name = exp_dir.name
        print(f"\nProcessing experiment: {exp_name} ...")

        # Analyze sessions using the shared helper
        session_results = analyze_dale_sessions(exp_dir)
        if not session_results:
            logger.warning(f'No valid sessions found for {exp_name}')
            continue

        totals = aggregate_task_counts(session_results)
        # Create barchart
        output_path = args.output_dir / f'{exp_name}_sampled_task_barchart.png'
        plot_barchart_for_experiment(exp_name, totals, output_path, aggregate_spans=args.aggregate_spans)

    print(f"\nAll barcharts saved to: {args.output_dir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
