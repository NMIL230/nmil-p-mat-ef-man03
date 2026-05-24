#!/usr/bin/env python3
"""Merge CPU-sharded IMLE_TB and IMLE_Random outputs into caches/final CSV."""

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
    load_method_data_from_csv,
)


METHOD_SPECS = (
    ("IMLE_TB", "imle_tb_cache_csv"),
    ("IMLE_Random", "imle_random_cache_csv"),
)
METHOD_CACHE_ATTRS = {method_name: cache_attr for method_name, cache_attr in METHOD_SPECS}


def configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("merge_imle_tb_shards")
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


def parse_methods(methods_raw: str) -> List[str]:
    normalized = methods_raw.replace("+", ",").replace(";", ",")
    methods = []
    seen = set()
    for method_name in normalized.split(","):
        method_name = method_name.strip()
        if not method_name or method_name in seen:
            continue
        if method_name not in METHOD_CACHE_ATTRS:
            raise ValueError(f"Unsupported IMLE method '{method_name}'")
        seen.add(method_name)
        methods.append(method_name)
    if not methods:
        raise ValueError("No IMLE methods selected")
    return methods


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
    def _num_tests_key(row: Dict[str, object]) -> int:
        try:
            return int(row["num_tests"])
        except Exception:
            return 0

    return sorted(rows, key=lambda row: (str(row["method"]), str(row["session_id"]), _num_tests_key(row)))


def dedupe_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    deduped: Dict[Tuple[str, str, int], Dict[str, object]] = {}
    for row in rows:
        try:
            num_tests = int(row["num_tests"])
        except Exception:
            num_tests = 0
        key = (str(row["method"]), str(row["session_id"]), num_tests)
        deduped[key] = row
    return sort_rows(list(deduped.values()))


def write_rows_csv(csv_path: Path, rows: List[Dict[str, object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "session_id", "num_tests", "kld_value", "rmse_value"],
        )
        writer.writeheader()
        writer.writerows(sort_rows(rows))


def write_method_cache(cache_csv: Path, method_name: str, method_data: Dict[str, Dict[str, List[float]]]) -> None:
    rows = dedupe_rows(list(method_rows(method_name, method_data)))
    write_rows_csv(cache_csv, rows)


def count_points(method_data: Dict[str, Dict[str, List[float]]]) -> int:
    return sum(len(session_data.get("num_tests", [])) for session_data in method_data.values())


def load_metadata(metadata_path: Path) -> Dict[str, object]:
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge IMLE_TB and IMLE_Random shard outputs.")
    parser.add_argument("--manifest_csv", type=Path, required=True)
    parser.add_argument("--dim_output_dir", type=Path, required=True)
    parser.add_argument("--base_csv", type=Path, required=True)
    parser.add_argument("--metadata_json", type=Path, required=True)
    parser.add_argument("--methods", type=str, default="IMLE_TB+IMLE_Random")
    parser.add_argument("--imle_tb_cache_csv", type=Path, required=True)
    parser.add_argument("--imle_random_cache_csv", type=Path, required=True)
    args = parser.parse_args()

    args.dim_output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(args.dim_output_dir / "imle_tb_shard_merge.log")
    selected_methods = parse_methods(args.methods)
    logger.info("Starting IMLE_TB/IMLE_Random shard merge")
    logger.info("Selected methods: %s", ",".join(selected_methods))
    logger.info("Manifest: %s", args.manifest_csv)
    logger.info("Base CSV: %s", args.base_csv)

    manifest_rows = read_manifest_rows(args.manifest_csv)
    logger.info("Manifest shard rows: %d", len(manifest_rows))

    shard_method_data: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        method_name: {} for method_name in selected_methods
    }
    for row in manifest_rows:
        shard_output_dir = Path(row["shard_output_dir"])
        shard_csv = shard_output_dir / "imle_shard_data.csv"
        if not shard_csv.exists():
            logger.error("Missing shard CSV: %s", shard_csv)
            return 1
        for method_name in selected_methods:
            shard_rows = load_method_data_from_csv(shard_csv, method_name)
            if shard_rows:
                shard_method_data[method_name] = merge_method_sources(shard_method_data[method_name], shard_rows)

    merged_method_data: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for method_name in selected_methods:
        cache_csv = getattr(args, METHOD_CACHE_ATTRS[method_name])
        cache_rows = load_method_data_from_csv(cache_csv, method_name)
        base_rows_method = load_method_data_from_csv(args.base_csv, method_name)
        merged_rows = merge_method_sources(cache_rows, base_rows_method, shard_method_data[method_name])
        merged_method_data[method_name] = merged_rows

        if merged_rows or cache_csv.exists():
            write_method_cache(cache_csv, method_name, merged_rows)
            logger.info(
                "Wrote %s cache: sessions=%d rows=%d",
                method_name,
                len(merged_rows),
                count_points(merged_rows),
            )

    base_rows = read_generic_rows(args.base_csv)
    if not base_rows:
        logger.warning("Base CSV missing or empty; final CSV will contain IMLE rows only")

    target_methods = set(selected_methods)
    preserved_rows = [row for row in base_rows if row.get("method") not in target_methods]
    new_rows = []
    for method_name, method_data in merged_method_data.items():
        new_rows.extend(method_rows(method_name, method_data))
    final_rows = dedupe_rows(preserved_rows + new_rows)
    write_rows_csv(args.base_csv, final_rows)
    logger.info("Final comparison CSV updated: %s", args.base_csv)
    logger.info("Final CSV rows: %d", len(final_rows))

    metadata = load_metadata(args.metadata_json)
    final_methods = sorted({str(row["method"]) for row in final_rows})
    metadata["methods_computed"] = final_methods
    metadata["imle_selected_methods"] = selected_methods
    metadata["imle_shard_manifest"] = str(args.manifest_csv)
    metadata["imle_shard_rows_merged"] = sum(count_points(method_data) for method_data in shard_method_data.values())
    for method_name, method_data in merged_method_data.items():
        metadata[f"{method_name}_sessions"] = len(method_data)
        metadata[f"{method_name}_total_points"] = count_points(method_data)

    with args.metadata_json.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    logger.info("Metadata updated: %s", args.metadata_json)
    logger.info("IMLE_TB/IMLE_Random shard merge complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
