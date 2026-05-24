import argparse
from collections import OrderedDict

import fig02
import fig03
import fig04
import fig05
import fig06
import fig07
import fig08
import figS01
import figS02
import figS03
import figS04
import figS05
import figS06
import figS07
import figS08
import figS09
import figS10
import figS11


GENERATORS = OrderedDict([
    ("fig02", fig02.generate),
    ("fig03", fig03.generate),
    ("fig04", fig04.generate),
    ("fig05", fig05.generate),
    ("fig06", fig06.generate),
    ("fig07", fig07.generate),
    ("fig08", fig08.generate),
    ("figS01", figS01.generate),
    ("figS02", figS02.generate),
    ("figS03", figS03.generate),
    ("figS04", figS04.generate),
    ("figS05", figS05.generate),
    ("figS06", figS06.generate),
    ("figS07", figS07.generate),
    ("figS08", figS08.generate),
    ("figS09", figS09.generate),
    ("figS10", figS10.generate),
    ("figS11", figS11.generate),
])


def parse_args():
    parser = argparse.ArgumentParser(description="Generate manuscript and supplemental figure PDFs.")
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=list(GENERATORS.keys()),
        help="Optional subset of figure wrappers to run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print low-level commands without executing them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running later figures after a failure.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    selected = args.figures or list(GENERATORS.keys())
    failures = []

    for figure_name in selected:
        print("=== Generating %s ===" % figure_name)
        try:
            GENERATORS[figure_name](dry_run=args.dry_run)
        except Exception as exc:
            failures.append((figure_name, exc))
            print("%s failed: %s" % (figure_name, exc))
            if not args.continue_on_error:
                break

    if failures:
        raise SystemExit(
            "One or more figure wrappers failed: %s"
            % ", ".join(name for name, _ in failures)
        )

    print("All requested figure wrappers completed successfully.")


if __name__ == "__main__":
    main()
