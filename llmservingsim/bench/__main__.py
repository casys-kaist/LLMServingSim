"""CLI dispatch: ``python -m llmservingsim.bench {run,validate} ...``"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(prog="bench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run a vLLM benchmark and record results")
    from llmservingsim.bench.core.runner import register_args as run_register
    run_register(p_run)

    p_val = sub.add_parser("validate", help="Compare a bench run against simulator output")
    from llmservingsim.bench.core.validate import register_args as val_register
    val_register(p_val)

    args = parser.parse_args()

    if args.cmd == "run":
        from llmservingsim.bench.core.runner import run as run_bench
        return run_bench(args)
    if args.cmd == "validate":
        from llmservingsim.bench.core.validate import run as run_validate
        return run_validate(args)

    parser.error(f"Unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
