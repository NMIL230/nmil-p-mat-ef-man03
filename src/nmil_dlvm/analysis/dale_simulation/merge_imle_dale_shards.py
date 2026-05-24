#!/usr/bin/env python3
"""Merge CPU-sharded IMLE_DALE outputs into caches and the final comparison CSV."""

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


METHOD_SPECS: Tuple[Tuple[str, str], ...] = (
    ("IMLE_DALE_PS0", "imle_dale_ps0_cache_csv"),
    ("IMLE_DALE_PS2", "imle_dale_ps2_cache_csv"),
    ("IMLE_DALE_PS4", "imle_dale_ps4_cache_csv"),
)
METHOD_NAMES = tuple(method_name for method_name, _ in METHOD_SPECS)


def select_method_specs(method_selector: str) -> Tuple[Tuple[str, str], ...]:
    if method_selector == "all":
        return METHOD_SPECS
    for spec in METHOD_SPECS:
        if spec[0] == method_selector:
            return (spec,)
    raise ValueError(f"Unsupported IMLE_DALE method selector: {method_selector}")


def configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("merge_imle_dale_shards")
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


def sort_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    def _num_tests_key(row: Dict[str, object]) -> int:
        try:
            return int(row["num_tests"])
        except Exception:
            return 0

    return sorted(rows, key=lambda row: (str(row["method"]), str(row["session_id"]), _num_tests_key(row)))


def dedupe_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    def _num_tests_key(row: Dict[str, object]) -> int:
        try:
            return int(row["num_tests"])
        except Exception:
            return 0

    deduped: Dict[Tuple[str, str, int], Dict[str, object]] = {}
    for row in rows:
        key = (str(row["method"]), str(row["session_id"]), _num_tests_key(row))
        deduped[key] = row
    return sort_rows(list(deduped.values()))


def write_rows_csv(csv_path: Path, rows: List[Dict[str, object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sort_rows(rows)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "session_id", "num_tests", "kld_value", "rmse_value"],
        )
        writer.writeheader()
        writer.writerows(sorted_rows)


def write_method_cache(cache_csv: Path, method_name: str, method_data: Dict[str, Dict[str, List[float]]]) -> None:
    rows = list(method_rows(method_name, method_data))
    rows = dedupe_rows(rows)
    write_rows_csv(cache_csv, rows)


def count_points(method_data: Dict[str, Dict[str, List[float]]]) -> int:
    return sum(len(session_data.get("num_tests", [])) for session_data in method_data.values())


def load_metadata(metadata_path: Path) -> Dict[str, object]:
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge IMLE_DALE shard outputs into caches and the final CSV.")
    parser.add_argument("--manifest_csv", type=Path, required=True)
    parser.add_argument("--dim_output_dir", type=Path, required=True)
    parser.add_argument("--base_csv", type=Path, required=True)
    parser.add_argument("--metadata_json", type=Path, required=True)
    parser.add_argument("--imle_dale_ps0_cache_csv", type=Path, required=True)
    parser.add_argument("--imle_dale_ps2_cache_csv", type=Path, required=True)
    parser.add_argument("--imle_dale_ps4_cache_csv", type=Path, required=True)
    parser.add_argument("--imle_dale_method", type=str, default="all", choices=("all",) + METHOD_NAMES)
    args = parser.parse_args()

    args.dim_output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(args.dim_output_dir / "imle_dale_shard_merge.log")
    selected_method_specs = select_method_specs(args.imle_dale_method)
    logger.info("Starting IMLE_DALE shard merge")
    logger.info("Manifest: %s", args.manifest_csv)
    logger.info("Base CSV: %s", args.base_csv)
    logger.info("Method selector: %s", args.imle_dale_method)

    manifest_rows = read_manifest_rows(args.manifest_csv)
    logger.info("Manifest shard rows: %d", len(manifest_rows))

    shard_method_data: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        method_name: {} for method_name, _ in selected_method_specs
    }

    for row in manifest_rows:
        shard_output_dir = Path(row["shard_output_dir"])
        shard_csv = shard_output_dir / "imle_dale_shard_data.csv"
        if not shard_csv.exists():
            logger.error("Missing shard CSV: %s", shard_csv)
            return 1
        for method_name, _ in selected_method_specs:
            shard_rows = load_method_data_from_csv(shard_csv, method_name)
            if shard_rows:
                shard_method_data[method_name] = merge_method_sources(
                    shard_method_data[method_name],
                    shard_rows,
                )

    merged_method_data: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for method_name, cache_attr in selected_method_specs:
        cache_csv = getattr(args, cache_attr)
        cache_rows = load_method_data_from_csv(cache_csv, method_name)
        base_rows = load_method_data_from_csv(args.base_csv, method_name)
        merged_rows = merge_method_sources(
            cache_rows,
            base_rows,
            shard_method_data[method_name],
        )
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
        logger.warning("Base CSV missing or empty; final CSV will contain IMLE_DALE rows only")

    target_methods = {method_name for method_name, _ in selected_method_specs}
    preserved_rows = [row for row in base_rows if row.get("method") not in target_methods]
    new_rows: List[Dict[str, object]] = []
    for method_name, method_data in merged_method_data.items():
        new_rows.extend(method_rows(method_name, method_data))

    final_rows = dedupe_rows(preserved_rows + new_rows)
    write_rows_csv(args.base_csv, final_rows)
    logger.info("Final comparison CSV updated: %s", args.base_csv)
    logger.info("Final CSV rows: %d", len(final_rows))

    metadata = load_metadata(args.metadata_json)
    final_methods = sorted({str(row["method"]) for row in final_rows})
    metadata["methods_computed"] = final_methods
    metadata["imle_dale_method_selector"] = args.imle_dale_method
    metadata["imle_dale_shard_manifest"] = str(args.manifest_csv)
    metadata["imle_dale_shard_rows_merged"] = sum(count_points(data) for data in shard_method_data.values())
    for method_name, method_data in merged_method_data.items():
        metadata[f"{method_name}_sessions"] = len(method_data)
        metadata[f"{method_name}_total_points"] = count_points(method_data)

    with args.metadata_json.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    logger.info("Metadata updated: %s", args.metadata_json)
    logger.info("IMLE_DALE shard merge complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
