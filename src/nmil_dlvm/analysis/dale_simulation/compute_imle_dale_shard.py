#!/usr/bin/env python3
"""Compute IMLE_DALE shard outputs for a subset of sessions on CPU only."""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from nmil_dlvm.analysis.dale_simulation.build_dale_comparison_csv import (  # noqa: E402
    SUMMARIZED_METRICS,
    build_imle_dale_reference_data,
    compute_imle_dale_data,
    load_ground_truth_with_key_matching,
    load_method_data_from_csv,
    split_cached_sessions_for_reference,
)


METHOD_SPECS: Tuple[Tuple[str, str, str], ...] = (
    ("IMLE_DALE_PS0", "dale_ps0_dir", "imle_dale_ps0_cache_csv"),
    ("IMLE_DALE_PS2", "dale_ps2_dir", "imle_dale_ps2_cache_csv"),
    ("IMLE_DALE_PS4", "dale_ps4_dir", "imle_dale_ps4_cache_csv"),
)
METHOD_NAMES = tuple(method_name for method_name, _, _ in METHOD_SPECS)


def select_method_specs(method_selector: str) -> Tuple[Tuple[str, str, str], ...]:
    if method_selector == "all":
        return METHOD_SPECS
    for spec in METHOD_SPECS:
        if spec[0] == method_selector:
            return (spec,)
    raise ValueError(f"Unsupported IMLE_DALE method selector: {method_selector}")


def configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("compute_imle_dale_shard")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


def load_session_ids(session_ids_file: Path) -> List[str]:
    sessions = []
    with session_ids_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            session_id = line.strip()
            if session_id:
                sessions.append(session_id)
    return sessions


def clone_method_data(method_data: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, List[float]]]:
    cloned: Dict[str, Dict[str, List[float]]] = {}
    for session_id, session_data in method_data.items():
        cloned[session_id] = {
            "num_tests": list(session_data.get("num_tests", [])),
            "kld_values": list(session_data.get("kld_values", [])),
            "rmse_values": list(session_data.get("rmse_values", [])),
        }
    return cloned


def merge_method_sources(*sources: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, List[float]]]:
    merged: Dict[str, Dict[str, List[float]]] = {}
    for source in sources:
        for session_id, session_data in source.items():
            existing = merged.get(session_id)
            if existing is None or len(session_data.get("num_tests", [])) > len(existing.get("num_tests", [])):
                merged[session_id] = {
                    "num_tests": list(session_data.get("num_tests", [])),
                    "kld_values": list(session_data.get("kld_values", [])),
                    "rmse_values": list(session_data.get("rmse_values", [])),
                }
    return merged


def method_rows(method_name: str, method_data: Dict[str, Dict[str, List[float]]]) -> Iterable[Dict[str, object]]:
    for session_id, session_data in method_data.items():
        num_tests = session_data.get("num_tests", [])
        kld_values = session_data.get("kld_values", [])
        rmse_values = session_data.get("rmse_values", [])
        for idx, num_tests_value in enumerate(num_tests):
            yield {
                "method": method_name,
                "session_id": session_id,
                "num_tests": int(num_tests_value),
                "kld_value": kld_values[idx] if idx < len(kld_values) else "",
                "rmse_value": rmse_values[idx] if idx < len(rmse_values) else "",
            }


def write_method_data_csv(output_csv: Path, all_method_data: Dict[str, Dict[str, Dict[str, List[float]]]]) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for method_name, method_data in all_method_data.items():
        rows.extend(method_rows(method_name, method_data))

    rows.sort(key=lambda row: (row["method"], row["session_id"], int(row["num_tests"])))
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "session_id", "num_tests", "kld_value", "rmse_value"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def count_points(method_data: Dict[str, Dict[str, List[float]]]) -> int:
    return sum(len(session_data.get("num_tests", [])) for session_data in method_data.values())


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute IMLE_DALE shard outputs for a subset of sessions.")
    parser.add_argument("--session_ids_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dale_ps0_dir", type=Path, required=True)
    parser.add_argument("--dale_ps2_dir", type=Path, required=True)
    parser.add_argument("--dale_ps4_dir", type=Path, required=True)
    parser.add_argument("--mle_params_file", type=Path, required=True)
    parser.add_argument("--max_iterations", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--imle_dale_ps0_cache_csv", type=Path, default=None)
    parser.add_argument("--imle_dale_ps2_cache_csv", type=Path, default=None)
    parser.add_argument("--imle_dale_ps4_cache_csv", type=Path, default=None)
    parser.add_argument("--base_csv", type=Path, default=None)
    parser.add_argument("--imle_dale_method", type=str, default="all", choices=("all",) + METHOD_NAMES)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(args.output_dir / "computation_progress.log")
    selected_method_specs = select_method_specs(args.imle_dale_method)

    session_ids = load_session_ids(args.session_ids_file)
    logger.info("Starting IMLE_DALE shard computation")
    logger.info("Shard output dir: %s", args.output_dir)
    logger.info("Session file: %s", args.session_ids_file)
    logger.info("Sessions requested: %d", len(session_ids))
    logger.info("Method selector: %s", args.imle_dale_method)
    if args.base_csv:
        logger.info("Existing comparison CSV fallback: %s", args.base_csv)

    ground_truth = load_ground_truth_with_key_matching(str(args.mle_params_file), session_ids)
    parameters = {metric: [0] for metric in SUMMARIZED_METRICS}

    computed_method_data: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    metadata = {
        "session_ids_file": str(args.session_ids_file),
        "session_ids": session_ids,
        "num_requested_sessions": len(session_ids),
        "num_ground_truth_sessions": len(ground_truth),
        "method_selector": args.imle_dale_method,
        "methods": {},
    }

    for method_name, dir_attr, cache_attr in selected_method_specs:
        dale_dir = getattr(args, dir_attr)
        cache_csv = getattr(args, cache_attr)

        logger.info("--- %s ---", method_name)
        reference_data = build_imle_dale_reference_data(
            str(dale_dir),
            session_list=session_ids,
            max_iterations=args.max_iterations,
        )

        cache_method_data = load_method_data_from_csv(cache_csv, method_name) if cache_csv else {}
        csv_method_data = load_method_data_from_csv(args.base_csv, method_name) if args.base_csv else {}
        existing_method_data = merge_method_sources(cache_method_data, csv_method_data)
        cached_hits, missing_reference = split_cached_sessions_for_reference(existing_method_data, reference_data)

        logger.info(
            "%s reference sessions=%d cached_complete=%d missing=%d",
            method_name,
            len(reference_data),
            len(cached_hits),
            len(missing_reference),
        )

        method_meta = {
            "reference_sessions": len(reference_data),
            "cached_complete_sessions": len(cached_hits),
            "sessions_to_compute": len(missing_reference),
            "computed_sessions": 0,
            "computed_points": 0,
        }

        if missing_reference:
            newly_computed = compute_imle_dale_data(
                str(dale_dir),
                ground_truth,
                None,
                SUMMARIZED_METRICS,
                parameters,
                logger,
                session_list=sorted(missing_reference.keys()),
                max_iterations=args.max_iterations,
                workers=args.workers,
                append_rows_fn=None,
                method_label=method_name,
            )
            if newly_computed:
                computed_method_data[method_name] = clone_method_data(newly_computed)
                method_meta["computed_sessions"] = len(newly_computed)
                method_meta["computed_points"] = count_points(newly_computed)

        metadata["methods"][method_name] = method_meta

    shard_csv = args.output_dir / "imle_dale_shard_data.csv"
    rows_written = write_method_data_csv(shard_csv, computed_method_data)
    metadata["rows_written"] = rows_written
    metadata["methods_computed"] = sorted(computed_method_data.keys())

    metadata_path = args.output_dir / "imle_dale_shard_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    logger.info("Shard computation complete")
    logger.info("Rows written: %d", rows_written)
    logger.info("CSV output: %s", shard_csv)
    logger.info("Metadata output: %s", metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
