#!/usr/bin/env python3
"""
Quick checker for 1D experiment session completeness under DEFAULT_RES.

For each experiment directory under `analysis/dale_syn_data/result/1d-final-now`:
- Counts complete sessions (exactly 240 iterations in CSV)
- Counts partial sessions (CSV exists with < 240 iterations)
- Counts zero-iteration/broken sessions (session folder exists but CSV missing)
- Counts missing sessions (expected 88 total; list those not present at all)

Prints counts and lists of session IDs for each category.

Usage:
  python analysis/dale_syn_data/check_sessions_1d_status.py \
    [--root <path_to_1d-final-now>] \
    [--gt <path_to_ground_truth_pt>] \
    [--expected 88]

Defaults:
  - root: analysis/dale_syn_data/result/1d-final-now (relative to this file)
  - gt: data/COLL10_SIM/D1_synthetic_ground_truth_parameters_wolverine-zoom-7298.pt
  - expected: derived from gt keys if available, else 88
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Set


def find_csv_for_session(session_dir: Path, session_id: str) -> Path | None:
    candidates = [
        session_dir / "analysis" / f"performance_tracking_session_{session_id}.csv",
        session_dir / f"performance_tracking_session_{session_id}.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def count_csv_rows(csv_path: Path) -> int:
    try:
        with csv_path.open("r", newline="") as f:
            reader = csv.reader(f)
            # Subtract header row if present
            rows = list(reader)
            if not rows:
                return 0
            # Assume first row is header when it contains non-numeric strings
            # We simply return len(rows) - 1 to match pandas length behavior in project
            return max(0, len(rows) - 1)
    except Exception:
        return 0


def load_expected_ids(gt_path: Path | None, fallback_expected: int | None = 88) -> List[str]:
    ids: List[str] = []
    if gt_path and gt_path.exists():
        try:
            import torch  # lazy import
            data = torch.load(str(gt_path))
            ids = sorted(list(data.keys()))
        except Exception:
            ids = []
    if not ids and fallback_expected:
        # Fallback to LD1-001..LD1-<expected> if GT not available
        ids = [f"LD1-{i:03d}" for i in range(1, int(fallback_expected) + 1)]
    return ids


def analyze_experiment(exp_dir: Path, expected_ids: List[str], required_iters: int = 240) -> Dict[str, List[str]]:
    present_session_dirs = {d.name: d for d in exp_dir.iterdir() if d.is_dir() and "aggregate" not in d.name.lower()}

    complete: List[str] = []
    partial: List[str] = []
    broken: List[str] = []  # folder exists but CSV missing

    for sid, sdir in sorted(present_session_dirs.items()):
        csv_path = find_csv_for_session(sdir, sid)
        if not csv_path:
            broken.append(sid)
            continue
        n = count_csv_rows(csv_path)
        if n == required_iters:
            complete.append(sid)
        elif 0 < n < required_iters:
            partial.append(sid)
        else:
            # Treat 0 or >required as broken for now
            broken.append(sid)

    present_ids = set(present_session_dirs.keys())
    expected_set = set(expected_ids)
    missing: List[str] = sorted(list(expected_set - present_ids))

    return {
        "complete": sorted(complete),
        "partial": sorted(partial),
        "broken": sorted(broken),
        "missing": missing,
    }


def main():
    this_file = Path(__file__).resolve()
    script_root = this_file.parent
    repo_root = script_root.parents[2]

    default_root = script_root / "result" / "1d-final-now"
    default_gt = repo_root / "data" / "COLL10_SIM" / "D1_synthetic_ground_truth_parameters_wolverine-zoom-7298.pt"

    ap = argparse.ArgumentParser(description="Check 1D experiment session completeness.")
    ap.add_argument("--root", type=Path, default=default_root, help="Root directory containing experiment folders")
    ap.add_argument("--gt", type=Path, default=default_gt, help="Ground truth PT file to derive expected session IDs")
    ap.add_argument("--expected", type=int, default=88, help="Expected number of sessions if GT unavailable")
    ap.add_argument("--iters", type=int, default=240, help="Required iterations to count as complete")
    args = ap.parse_args()

    root: Path = args.root
    if not root.exists() or not root.is_dir():
        print(f"Error: root directory not found: {root}")
        return 2

    expected_ids = load_expected_ids(args.gt, args.expected)
    if not expected_ids:
        print("Error: could not determine expected session IDs (GT missing and no fallback)")
        return 3

    # Pick experiment folders (by default names starting with 'exp_')
    exp_dirs: List[Path] = [d for d in root.iterdir() if d.is_dir() and d.name.startswith("exp_")]
    if not exp_dirs:
        # If naming differs, include all non-aggregate dirs
        exp_dirs = [d for d in root.iterdir() if d.is_dir() and "aggregate" not in d.name.lower()]

    exp_dirs = sorted(exp_dirs, key=lambda p: p.name)

    print("\n=== Session Completeness Report (1D) ===")
    print(f"Root: {root}")
    print(f"Expected sessions: {len(expected_ids)} ({expected_ids[0]}..{expected_ids[-1]})")
    print(f"Required iterations per session: {args.iters}")
    print("".rstrip())

    for exp in exp_dirs:
        result = analyze_experiment(exp, expected_ids, required_iters=args.iters)
        complete = result["complete"]
        partial = result["partial"]
        broken = result["broken"]
        missing = result["missing"]

        print("-" * 80)
        print(f"Experiment: {exp.name}")
        print(f"  Complete (=={args.iters}): {len(complete)}")
        print(f"  Partial  (<{args.iters}): {len(partial)}")
        print(f"  Broken (no CSV):         {len(broken)}")
        print(f"  Missing (no folder):     {len(missing)}")

        if complete:
            print(f"    Complete IDs: {', '.join(complete)}")
        if partial:
            print(f"    Partial IDs:  {', '.join(partial)}")
        if broken:
            print(f"    Broken IDs:   {', '.join(broken)}")
        if missing:
            print(f"    Missing IDs:  {', '.join(missing)}")

    print("-" * 80)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
