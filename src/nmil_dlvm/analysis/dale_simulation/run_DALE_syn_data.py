#!/usr/bin/env python3
"""Compatibility wrapper for synthetic-data DALE runs.

This entrypoint preserves the historical `run_DALE_syn_data.py` command surface
while delegating execution to the canonical DALE CLI in `nmil_dlvm.cli.run_dale`.
Synthetic-specific defaults are only injected when the caller did not already
provide them explicitly.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nmil_dlvm.cli.run_dale import main as run_dale_main

DEFAULT_FLAG_VALUES = {
    "--use_synthetic_data": "True",
    "--run_mode": "run",
    "--num_restarts": "1",
    "--create_marginal_fits_visualizations": "False",
    "--save_intermediate_updates": "True",
}


def _provided_flags(argv: list[str]) -> set[str]:
    provided = set()
    for arg in argv:
        if not arg.startswith("--"):
            continue
        if "=" in arg:
            provided.add(arg.split("=", 1)[0])
        else:
            provided.add(arg)
    return provided


def _build_argv(argv: list[str]) -> list[str]:
    normalized = list(argv)
    provided = _provided_flags(normalized)

    for flag, value in DEFAULT_FLAG_VALUES.items():
        if flag not in provided:
            normalized.extend([flag, value])

    if "--mle_params_file" not in provided:
        default_mle_params_file = REPO_ROOT / "data" / "COLL10_SIM" / "synthetic_ground_truth_parameters.pt"
        if default_mle_params_file.exists():
            normalized.extend(["--mle_params_file", str(default_mle_params_file)])

    return normalized


def main(argv: list[str] | None = None) -> None:
    run_dale_main(_build_argv(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    main()
