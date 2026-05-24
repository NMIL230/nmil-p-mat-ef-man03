#!/usr/bin/env python3
"""Compute DLVM_RANDOM shard outputs for a subset of sessions."""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from nmil_dlvm.analysis.dale_simulation.build_dale_comparison_csv import (  # noqa: E402
    SUMMARIZED_METRICS,
    compute_dlvm_random_data,
    load_ground_truth_with_key_matching,
    load_method_data_from_csv,
    split_cached_sessions_for_reference,
)
from nmil_dlvm.analysis.dlvm_imle_comparison.fit_dlvm_and_imle_models_to_data import load_simulated_data  # noqa: E402


METHOD_NAME = "DLVM_RANDOM"


def configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("compute_dlvm_random_shard")
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


def build_synthetic_reference_data(session_ids: List[str], max_iterations: int) -> Dict[str, Dict[str, List[float]]]:
    num_tests = list(range(1, max_iterations + 1))
    return {
        session_id: {
            "num_tests": list(num_tests),
            "kld_values": [],
            "rmse_values": [],
        }
        for session_id in session_ids
    }


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


def write_method_data_csv(output_csv: Path, method_name: str, method_data: Dict[str, Dict[str, List[float]]]) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = list(method_rows(method_name, method_data))
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute DLVM_RANDOM shard outputs for a subset of sessions.")
    parser.add_argument("--session_ids_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--synthetic_data_path", type=Path, required=True)
    parser.add_argument("--mle_params_file", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--latent_dim", type=int, required=True)
    parser.add_argument("--max_iterations", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dlvm_random_cache_csv", type=Path, default=None)
    parser.add_argument("--base_csv", type=Path, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(args.output_dir / "computation_progress.log")

    session_ids = load_session_ids(args.session_ids_file)
    logger.info("Starting DLVM_RANDOM shard computation")
    logger.info("Shard output dir: %s", args.output_dir)
    logger.info("Session file: %s", args.session_ids_file)
    logger.info("Sessions requested: %d", len(session_ids))
    logger.info("Synthetic data: %s", args.synthetic_data_path)
    logger.info("Model path: %s", args.model_path)
    logger.info("DLVM_RANDOM cache: %s", args.dlvm_random_cache_csv)

    reference_data = build_synthetic_reference_data(session_ids, args.max_iterations)
    cache_method_data = load_method_data_from_csv(args.dlvm_random_cache_csv, METHOD_NAME) if args.dlvm_random_cache_csv else {}
    csv_method_data = load_method_data_from_csv(args.base_csv, METHOD_NAME) if args.base_csv else {}
    existing_method_data = merge_method_sources(cache_method_data, csv_method_data)
    cached_hits, missing_reference = split_cached_sessions_for_reference(existing_method_data, reference_data)

    metadata = {
        "session_ids_file": str(args.session_ids_file),
        "session_ids": session_ids,
        "num_requested_sessions": len(session_ids),
        "reference_sessions": len(reference_data),
        "cached_complete_sessions": len(cached_hits),
        "sessions_to_compute": len(missing_reference),
        "computed_sessions": 0,
        "computed_points": 0,
        "reference_mode": "synthetic_schedule",
    }

    logger.info(
        "Reference sessions=%d cached_complete=%d missing=%d",
        len(reference_data),
        len(cached_hits),
        len(missing_reference),
    )

    computed_method_data: Dict[str, Dict[str, List[float]]] = {}
    if missing_reference:
        ground_truth = load_ground_truth_with_key_matching(str(args.mle_params_file), sorted(missing_reference.keys()))
        synthetic_data = load_simulated_data(str(args.synthetic_data_path), logger)
        parameters = {metric: [0] for metric in SUMMARIZED_METRICS}
        logger.info("Computing DLVM_RANDOM for %d uncached sessions", len(missing_reference))
        computed_method_data = compute_dlvm_random_data(
            missing_reference,
            synthetic_data,
            ground_truth,
            None,
            SUMMARIZED_METRICS,
            parameters,
            logger,
            workers=args.workers,
            model_path=str(args.model_path),
            latent_dim=args.latent_dim,
            append_rows_fn=None,
            method_label=METHOD_NAME,
        )
        metadata["computed_sessions"] = len(computed_method_data)
        metadata["computed_points"] = count_points(computed_method_data)
    else:
        logger.info("All requested sessions are already complete in cache")

    shard_csv = args.output_dir / "dlvm_random_shard_data.csv"
    rows_written = write_method_data_csv(shard_csv, METHOD_NAME, computed_method_data)
    metadata["rows_written"] = rows_written

    metadata_path = args.output_dir / "dlvm_random_shard_summary.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    logger.info("Shard computation complete")
    logger.info("Rows written: %d", rows_written)
    logger.info("CSV output: %s", shard_csv)
    logger.info("Summary output: %s", metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
