#!/usr/bin/env python3
"""Run every figure generator in sequence with a single command."""
from __future__ import annotations

import argparse
from typing import Callable, Dict, Iterable, List

import fig2
import fig3
import fig4
import fig5
import fig6
import fig7
import fig8


FigureRunner = Callable[[bool], None]

# Map figure labels to callables so we can look them up dynamically.
_GENERATORS: Dict[str, FigureRunner] = {
    "fig2": lambda dry: fig2.generate(dry_run=dry),
    "fig3": lambda dry: fig3.generate(dry_run=dry),
    "fig4": lambda dry: fig4.generate(dry_run=dry),
    "fig5": lambda dry: fig5.generate(dry_run=dry),
    "fig6": lambda dry: fig6.generate(dry_run=dry),
    "fig7": lambda dry: fig7.generate(dry_run=dry),
    "fig8": lambda dry: fig8.generate(dry_run=dry),
}

_DEFAULT_ORDER: List[str] = list(_GENERATORS)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate all (or selected) figures.")
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=_DEFAULT_ORDER,
        help="Subset of figures to run (defaults to all in numeric order).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands each figure would execute without running them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running remaining figures even if one fails.",
    )
    return parser.parse_args()


def _iter_figures(selection: Iterable[str] | None) -> List[str]:
    if selection is None:
        return _DEFAULT_ORDER
    # Preserve the default order while respecting the caller's subset.
    wanted = set(selection)
    return [name for name in _DEFAULT_ORDER if name in wanted]


def main() -> None:
    args = _parse_args()
    figures = _iter_figures(args.figures)
    if not figures:
        print("No figures selected; nothing to do.")
        return

    failures: List[str] = []
    for name in figures:
        print(f"=== Generating {name.upper()} ===")
        runner = _GENERATORS[name]
        try:
            runner(args.dry_run)
        except Exception as exc:  # noqa: BLE001
            failures.append(name)
            print(f"{name} failed: {exc}")
            if not args.continue_on_error:
                break

    if failures:
        failed_list = ", ".join(sorted(set(failures), key=figures.index))
        raise SystemExit(f"One or more figures failed: {failed_list}")

    print("All requested figures completed successfully.")


if __name__ == "__main__":
    main()
