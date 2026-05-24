#!/usr/bin/env python3
"""Merge sharded DLVM_RANDOM outputs into the cache and final comparison CSV."""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from nmil_dlvm.analysis.dale_simulation.build_dale_comparison_csv import (  # noqa: E402
    acquire_cache_lock,
    load_method_data_from_csv,
)


METHOD_NAME = "DLVM_RANDOM"


def configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("merge_dlvm_random_shards")
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


def read_manifest_rows(manifest_csv: Path) -> List[Dict[str, str]]:
    if not manifest_csv.exists():
        return []
    with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_generic_rows(csv_path: Path) -> List[Dict[str, object]]:
    if not csv_path or not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def sort_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(rows, key=lambda row: (str(row["method"]), str(row["session_id"]), int(row["num_tests"])))


def dedupe_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    deduped: Dict[Tuple[str, str, int], Dict[str, object]] = {}
    for row in rows:
        key = (str(row["method"]), str(row["session_id"]), int(row["num_tests"]))
        deduped[key] = row
    return sort_rows(list(deduped.values()))


def write_rows_csv_atomic(csv_path: Path, rows: List[Dict[str, object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = csv_path.with_name(f"{csv_path.name}.tmp.{os.getpid()}")
    sorted_rows = sort_rows(rows)
    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["method", "session_id", "num_tests", "kld_value", "rmse_value"],
            )
            writer.writeheader()
            writer.writerows(sorted_rows)
        os.replace(tmp_path, csv_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def count_points(method_data: Dict[str, Dict[str, List[float]]]) -> int:
    return sum(len(session_data.get("num_tests", [])) for session_data in method_data.values())


def load_metadata(metadata_path: Path) -> Dict[str, object]:
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge DLVM_RANDOM shard outputs into cache/final CSV.")
    parser.add_argument("--manifest_csv", type=Path, required=True)
    parser.add_argument("--dim_output_dir", type=Path, required=True)
    parser.add_argument("--dlvm_random_cache_csv", type=Path, required=True)
    parser.add_argument("--base_csv", type=Path, required=True)
    parser.add_argument("--metadata_json", type=Path, required=True)
    args = parser.parse_args()

    job_root = args.manifest_csv.parent
    job_root.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(job_root / "dlvm_random_merge.log")
    logger.info("Starting DLVM_RANDOM shard merge")
    logger.info("Manifest: %s", args.manifest_csv)
    logger.info("Cache CSV: %s", args.dlvm_random_cache_csv)
    logger.info("Base CSV: %s", args.base_csv)

    manifest_rows = read_manifest_rows(args.manifest_csv)
    logger.info("Manifest shard rows: %d", len(manifest_rows))

    shard_method_data: Dict[str, Dict[str, List[float]]] = {}
    shard_csv_count = 0
    for row in manifest_rows:
        shard_output_dir = Path(row["shard_output_dir"])
        shard_csv = shard_output_dir / "dlvm_random_shard_data.csv"
        if not shard_csv.exists():
            logger.error("Missing shard CSV: %s", shard_csv)
            return 1
        shard_rows = load_method_data_from_csv(shard_csv, METHOD_NAME)
        if shard_rows:
            shard_method_data = merge_method_sources(shard_method_data, shard_rows)
            shard_csv_count += 1

    with acquire_cache_lock(args.dlvm_random_cache_csv):
        cache_rows = load_method_data_from_csv(args.dlvm_random_cache_csv, METHOD_NAME)
        base_rows_method = load_method_data_from_csv(args.base_csv, METHOD_NAME)
        merged_rows = merge_method_sources(cache_rows, base_rows_method, shard_method_data)
        rows = dedupe_rows(list(method_rows(METHOD_NAME, merged_rows)))
        write_rows_csv_atomic(args.dlvm_random_cache_csv, rows)

    base_rows = read_generic_rows(args.base_csv)
    if not base_rows:
        logger.warning("Base CSV missing or empty; final CSV will contain DLVM_RANDOM rows only")
    preserved_rows = [row for row in base_rows if row.get("method") != METHOD_NAME]
    new_rows = list(method_rows(METHOD_NAME, merged_rows))
    final_rows = dedupe_rows(preserved_rows + new_rows)
    write_rows_csv_atomic(args.base_csv, final_rows)

    metadata = load_metadata(args.metadata_json)
    final_methods = sorted({str(row["method"]) for row in final_rows})
    metadata["methods_computed"] = final_methods
    metadata["dlvm_random_shard_manifest"] = str(args.manifest_csv)
    metadata["dlvm_random_shard_rows_merged"] = count_points(shard_method_data)
    metadata["DLVM_RANDOM_sessions"] = len(merged_rows)
    metadata["DLVM_RANDOM_total_points"] = count_points(merged_rows)
    with args.metadata_json.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    summary = {
        "manifest_csv": str(args.manifest_csv),
        "dlvm_random_cache_csv": str(args.dlvm_random_cache_csv),
        "base_csv": str(args.base_csv),
        "manifest_shards": len(manifest_rows),
        "shards_with_rows": shard_csv_count,
        "shard_rows_merged": count_points(shard_method_data),
        "final_sessions": len(merged_rows),
        "final_points": count_points(merged_rows),
        "final_csv_rows": len(final_rows),
    }
    summary_path = job_root / "dlvm_random_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    logger.info("DLVM_RANDOM cache updated: sessions=%d rows=%d", summary["final_sessions"], summary["final_points"])
    logger.info("Final comparison CSV updated: %s", args.base_csv)
    logger.info("Final CSV rows: %d", len(final_rows))
    logger.info("Metadata updated: %s", args.metadata_json)
    logger.info("Summary output: %s", summary_path)
    logger.info("DLVM_RANDOM shard merge complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
