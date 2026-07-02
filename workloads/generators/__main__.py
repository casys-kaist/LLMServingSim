"""CLI dispatch for workload generators.

Usage:
    python -m workloads.generators sharegpt --model <hf-id> --num-reqs 300 --sps 10 \
        --source <path-or-hf-id> --output workloads/sharegpt-<model>-<n>-sps<r>.jsonl
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(prog="workloads.generators")
    sub = parser.add_subparsers(dest="generator", required=True)

    sg = sub.add_parser("sharegpt", help="ShareGPT -> LLMServingSim JSONL")
    from workloads.generators.sharegpt import register_args as sg_register
    sg_register(sg)

    bg = sub.add_parser(
        "burstgpt",
        help="BurstGPT weekly_minute_trace.csv -> LLMServingSim JSONL",
    )
    from workloads.generators.burstgpt import register_args as bg_register
    bg_register(bg)

    args = parser.parse_args()

    if args.generator == "sharegpt":
        from workloads.generators.sharegpt import run
        return run(args)

    if args.generator == "burstgpt":
        from workloads.generators.burstgpt import run
        return run(args)

    parser.error(f"Unknown generator: {args.generator}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
