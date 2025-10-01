from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
_FALLBACK_DIR = REPO_ROOT / "manuscript_figures" / "generated_inputs" / "ground_truth"
_FALLBACK_FILENAME = "all_data-best_mle_params_mpf100.pt"


def fallback_ground_truth_path() -> Path:
    """Return the repository-relative path used for synthesized ground-truth parameters."""
    return _FALLBACK_DIR / _FALLBACK_FILENAME


def resolve_relative(base: Path, candidate: Union[str, Path]) -> Path:
    """Resolve a potentially relative path against the supplied base directory."""
    candidate_path = Path(candidate)
    return candidate_path if candidate_path.is_absolute() else (base / candidate_path)


def _select_reference_param_file(params_dir: Path) -> Optional[Path]:
    """Pick a representative synthetic MLE file to seed fallback ground-truth parameters."""
    candidate_orders = (500, 200, 100, 50, 20, 10, 5, 3, 2, 1)
    for n in candidate_orders:
        candidate = params_dir / f"synthetic_mle_params_N{n}.pt"
        if candidate.is_file():
            return candidate

    matches = sorted(params_dir.glob("synthetic_mle_params_N*.pt"))
    return matches[0] if matches else None


def _build_ground_truth_from_params(params_dir: Path) -> Optional[Path]:
    """Derive a fallback ground-truth .pt file from available synthetic parameter fits."""
    reference = _select_reference_param_file(params_dir)
    if reference is None:
        return None

    data = torch.load(reference, map_location="cpu")
    fallback = {}
    for run_id, metrics_dict in data.items():
        if not run_id.endswith("_sim1"):
            continue
        base_session = run_id.split("_sim")[0]
        fallback[base_session] = metrics_dict

    if not fallback:
        return None

    target_path = fallback_ground_truth_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fallback, target_path)
    return target_path


def ensure_ground_truth_file(
    candidate: Union[str, Path],
    params_dir: Path,
    *,
    allowed_sessions: Optional[list[str]] = None,
) -> Path:
    """Return an existing or synthesized ground-truth parameter file.

    If ``allowed_sessions`` is provided, the resulting file only contains the
    specified session IDs.
    """

    allowed_set = {sess for sess in (allowed_sessions or []) if sess}
    fallback_path = fallback_ground_truth_path()
    fallback_path.parent.mkdir(parents=True, exist_ok=True)

    def _filter_data(data: dict) -> dict:
        if not allowed_set:
            return data
        filtered = {k: v for k, v in data.items() if k in allowed_set}
        if not filtered:
            raise FileNotFoundError(
                "None of the requested sessions were found in the ground-truth data."
            )
        return filtered

    candidate_path = Path(candidate)
    if candidate_path.is_file() and not allowed_set:
        return candidate_path

    if candidate_path.is_file():
        data = torch.load(candidate_path, map_location="cpu")
        torch.save(_filter_data(data), fallback_path)
        return fallback_path

    if fallback_path.is_file() and not allowed_set:
        return fallback_path

    generated = _build_ground_truth_from_params(params_dir)
    if generated is None:
        raise FileNotFoundError(
            "Could not locate ground-truth MLE parameters. Provide --ground_truth_pt_file "
            "or keep src/data/COLL10_SIM/all_data-best_mle_params_mpf100.pt available."
        )

    data = torch.load(generated, map_location="cpu")
    data = _filter_data(data)
    torch.save(data, fallback_path)
    print(f"Using fallback ground-truth parameters generated from {params_dir} -> {fallback_path}")
    return fallback_path
