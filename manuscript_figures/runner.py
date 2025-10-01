import argparse
from typing import List, Optional

import fig2
import fig3
import fig4
import fig5
import fig6
import fig7
import fig8

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Figures 2–8")
    p.add_argument("--scripts-root", required=True, help="Folder containing the low-level figure scripts.")
    p.add_argument("--output-dir", default=None, help="Optional output directory (forward via passthrough if needed).")
    p.add_argument("--dry-run", action="store_true", help="Print command without executing.")

    sub = p.add_subparsers(dest="figure", required=True)

    # Figure 2
    fig2p = sub.add_parser("fig2", help="Figure 2: DALE latent trajectory")
    fig2p.add_argument("--dale-run-id", required=True)
    fig2p.add_argument("--max-length-to-plot", type=int, default=100)
    fig2p.add_argument("remainder", nargs=argparse.REMAINDER, help="Extra args after '--' are passed to the low-level script.")

    # Figures 3–8 (generic stubs; edit fig3.py … fig8.py to set args)
    for n in (3, 4, 5, 6, 7, 8):
        sp = sub.add_parser(f"fig{n}", help=f"Figure {n}")
        sp.add_argument("remainder", nargs=argparse.REMAINDER, help="Extra args after '--' are passed to the low-level script.")

    return p.parse_args()

def _extract_passthrough(remainder: List[str]) -> List[str]:
    if not remainder:
        return []
    # Remove leading '--' if present
    return remainder[1:] if remainder and remainder[0] == "--" else remainder

def main() -> None:
    args = _parse()
    passthrough = _extract_passthrough(getattr(args, "remainder", []))

    if args.figure == "fig2":
        fig2.generate(
            scripts_root=args.scripts_root,
            output_dir=args.output_dir,
            dale_run_id=args.dale_run_id,
            max_length_to_plot=args.max_length_to_plot,
            passthrough=passthrough,
            dry_run=args.dry_run,
        )
        return

    if args.figure == "fig3":
        fig3.generate(args.scripts_root, args.output_dir, passthrough, args.dry_run)
        return
    if args.figure == "fig4":
        fig4.generate(args.scripts_root, args.output_dir, passthrough, args.dry_run)
        return
    if args.figure == "fig5":
        fig5.generate(args.scripts_root, args.output_dir, passthrough, args.dry_run)
        return
    if args.figure == "fig6":
        fig6.generate(args.scripts_root, args.output_dir, passthrough, args.dry_run)
        return
    if args.figure == "fig7":
        fig7.generate(args.scripts_root, args.output_dir, passthrough, args.dry_run)
        return
    if args.figure == "fig8":
        fig8.generate(args.scripts_root, args.output_dir, passthrough, args.dry_run)
        return

if __name__ == "__main__":
    main()
